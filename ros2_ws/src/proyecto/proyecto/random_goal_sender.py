
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from nav_msgs.srv import GetPlan

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import random
import time


class RandomGoalSender(Node):
    def __init__(self):
        super().__init__("random_goal_sender")

        # --- Publisher de goals para el path_follower ---
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # --- Subscriber para saber cuándo se alcanzó el goal ---
        self.goal_reached = True  # al inicio no hay goal pendiente
        self.sub_goal_reached = self.create_subscription(
            Bool,
            "/navigation/goal_reached",
            self.callback_goal_reached,
            10,
        )

        # --- Cliente al planeador de caminos ---
        self.clt_plan_path = self.create_client(GetPlan, "/path_planning/plan_path")

        # --- obtener pose actual del robot ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- RUTA INICIAL (waypoints) ---
        # (4,-2), (4,-7), (2,-11.8), (-4,-12), (-4,-2), (-4,4), (0,5)
        self.initial_waypoints = [
            (4.0,  -2.0),
            (4.0,  -7.0),
            (2.0, -11.8),
            (-4.0, -12.0),
            (-4.0,  -2.0),
            (-4.0,   4.0),
            (0.0,    5.0),
        ]

        # --- Parámetros: límites del área donde se muestrean puntos (frame "map") ---
        # Puedes sobreescribirlos con --ros-args -p ...
        self.declare_parameter("min_x", -5.0)
        self.declare_parameter("max_x", 5.0)
        self.declare_parameter("min_y", -12.0)
        self.declare_parameter("max_y", 5.0)

        # Tiempo máx. para esperar que el robot llegue a un goal
        self.declare_parameter("goal_timeout", 120.0)   # segundos (double)
        # Número máx. de intentos para encontrar un goal alcanzable
        self.declare_parameter("max_plan_tries", 30)    # entero

        self.get_logger().info("RandomGoalSender inicializado.")

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def callback_goal_reached(self, msg: Bool):
        if msg.data:
            self.get_logger().info("✅ Goal alcanzado (/navigation/goal_reached)")
            self.goal_reached = True

    # ------------------------------------------------------------
    # Utilidades TF / servicios
    # ------------------------------------------------------------
    def wait_for_plan_service(self):
        self.get_logger().info("Esperando servicio /path_planning/plan_path ...")
        while rclpy.ok() and not self.clt_plan_path.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("  aún no está disponible...")
        self.get_logger().info("Servicio de plan de ruta disponible ✅")

    def wait_for_robot_pose(self):
        """Bloquea hasta que exista TF map->base_link."""
        self.get_logger().info("Esperando TF map->base_link ...")
        tf_ready = False
        while rclpy.ok() and not tf_ready:
            try:
                _ = self.tf_buffer.lookup_transform(
                    "map", "base_link", rclpy.time.Time()
                )
                tf_ready = True
            except TransformException:
                tf_ready = False
                self.get_logger().warn("  TF aún no está disponible...")
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        self.get_logger().info("TF map->base_link disponible ✅")

    def get_robot_pose(self):
        """Obtiene la pose del robot en 'map' usando TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            return (x, y)
        except TransformException:
            self.get_logger().warn("No pude obtener TF map->base_link")
            return None

    # ------------------------------------------------------------
    # Muestreo y planeo
    # ------------------------------------------------------------
    def sample_random_goal(self):
        """Genera una PoseStamped con posición aleatoria (sin checar aún si es alcanzable)."""
        min_x = self.get_parameter("min_x").get_parameter_value().double_value
        max_x = self.get_parameter("max_x").get_parameter_value().double_value
        min_y = self.get_parameter("min_y").get_parameter_value().double_value
        max_y = self.get_parameter("max_y").get_parameter_value().double_value

        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0  # orientación neutra

        return goal

    def is_goal_reachable(self, robot_xy, goal_pose):
        """Llama al servicio GetPlan para checar si existe un camino hasta goal_pose."""
        req = GetPlan.Request()
        req.start.header.frame_id = "map"
        req.goal.header.frame_id = "map"
        req.tolerance = 0.1

        req.start.pose.position.x = robot_xy[0]
        req.start.pose.position.y = robot_xy[1]
        req.start.pose.orientation.w = 1.0

        req.goal.pose = goal_pose.pose

        future = self.clt_plan_path.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result():
            self.get_logger().warn("El servicio GetPlan no devolvió resultado")
            return False

        plan = future.result().plan
        if len(plan.poses) == 0:
            # Sin puntos -> sin camino
            return False

        # Opcional: descartar planes demasiado cortos (p.ej. ruido muy cerca)
        if len(plan.poses) < 3:
            return False

        return True

    def make_goal_pose(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        return goal

    def send_goal(self, goal_pose):
        """Publica el goal en /goal_pose."""
        self.goal_reached = False
        self.goal_pub.publish(goal_pose)
        self.get_logger().info(
            f"📍 Nuevo goal enviado: "
            f"({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})"
        )

    # ------------------------------------------------------------
    # Lógica principal
    # ------------------------------------------------------------
    def run(self):
        # Esperar a servicios y TF antes de hacer nada serio
        self.wait_for_plan_service()
        self.wait_for_robot_pose()

        goal_timeout = self.get_parameter("goal_timeout").get_parameter_value().double_value
        max_plan_tries = self.get_parameter("max_plan_tries").get_parameter_value().integer_value

        # 1) RUTA INICIAL
        self.get_logger().info("Ejecutando RUTA INICIAL de waypoints...")
        for (gx, gy) in self.initial_waypoints:
            if not rclpy.ok():
                break

            robot_xy = self.get_robot_pose()
            if robot_xy is None:
                # TF debería estar listo, pero por si acaso...
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            goal = self.make_goal_pose(gx, gy)

            # Checar si es alcanzable
            if not self.is_goal_reachable(robot_xy, goal):
                self.get_logger().warn(
                    f"Waypoint inicial ({gx:.2f}, {gy:.2f}) no es alcanzable, lo salto."
                )
                continue

            # Enviar goal
            self.get_logger().info(
                f"➡️  Waypoint inicial: ({gx:.2f}, {gy:.2f})"
            )
            self.send_goal(goal)

            # Esperar a que llegue o se acabe el tiempo
            t0 = time.time()
            while rclpy.ok() and not self.goal_reached and (time.time() - t0) < goal_timeout:
                rclpy.spin_once(self, timeout_sec=0.1)

            if not self.goal_reached:
                self.get_logger().warn(
                    f"⏰ Tiempo agotado en waypoint inicial ({gx:.2f}, {gy:.2f}), sigo al siguiente."
                )

        # 2) GOALS ALEATORIOS
        self.get_logger().info("Ruta inicial terminada. Comenzando goals aleatorios...")
        while rclpy.ok():
            # 2.1) Pose actual
            robot_xy = self.get_robot_pose()
            if robot_xy is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            # 2.2) Buscar un goal aleatorio alcanzable
            found_goal = False
            goal = None
            for _ in range(max_plan_tries):
                candidate = self.sample_random_goal()
                if self.is_goal_reachable(robot_xy, candidate):
                    goal = candidate
                    found_goal = True
                    break

            if not found_goal:
                self.get_logger().warn(
                    "No encontré ningún goal aleatorio alcanzable. "
                    "Revisa min_x/max_x/min_y/max_y o el mapa."
                )
                rclpy.spin_once(self, timeout_sec=0.5)
                continue

            # 2.3) Enviar el goal
            self.send_goal(goal)

            # 2.4) Esperar a que llegue o timeout
            t0 = time.time()
            while rclpy.ok() and not self.goal_reached and (time.time() - t0) < goal_timeout:
                rclpy.spin_once(self, timeout_sec=0.1)

            if not self.goal_reached:
                self.get_logger().warn("⏰ Tiempo agotado para goal aleatorio, probando otro...")


def main(args=None):
    rclpy.init(args=args)
    node = RandomGoalSender()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

