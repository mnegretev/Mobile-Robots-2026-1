import rclpy
import random
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from action_msgs.msg import GoalStatusArray


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        # Publicador de metas para el path_follower
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Bandera: ya se encontraron los 3 objetos
        self.all_found = False
        self.flags_sub = self.create_subscription(
            Bool, '/all_objects_found', self.flags_callback, 10
        )

        # Estado de llegada al goal (status 3 = SUCCEEDED)
        self.arrived = True
        self.goal_reached_sub = self.create_subscription(Bool, '/navigation/goal_reached', self.goal_reached_callback, 10)

        self.goal_timeout = 10.0
        self.goal_start_time = None
        # Número de waypoints aleatorios a generar (puedes ajustar este valor)
        self.NUM_WAYPOINTS = 50  #Numero de  puntos

        self.WAYPOINTS = self.generate_waypoints(
            x_min=-4.0, x_max=6.0, y_min=-12.0, y_max=4.0, num_points=self.NUM_WAYPOINTS
        )
        self.current_idx = 0
        self.get_logger().info(f"Explorer (random): generados {len(self.WAYPOINTS)} waypoints aleatorios.")

        # Bucle principal
        self.timer = self.create_timer(0.5, self.main_loop)

    # ---------- Callbacks ----------

    def flags_callback(self, msg: Bool):
        self.all_found = msg.data

    def goal_reached_callback(self, msg: Bool):
        if msg.data:
            self.arrived = True
            self.goal_start_time = None
       

    # ---------- Generación de waypoints aleatorios ----------

    def generate_waypoints(self, x_min, x_max, y_min, y_max, num_points):
        """Genera waypoints aleatorios dentro de los límites especificados"""
        waypoints = []
        for _ in range(num_points):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            waypoints.append((x, y))
        return waypoints

    def publish_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.get_logger().info(f"Nuevo goal publicado: ({x:.2f}, {y:.2f})")
        self.arrived = False
        self.goal_start_time = self.get_clock().now()

    # ---------- Bucle principal ----------

    def main_loop(self):
        # Si ya se encontraron los 3 objetos, detener exploración
        if self.all_found:
            self.get_logger().info("Los 3 objetos ya fueron encontrados. Exploración detenida.")
            self.timer.cancel()
            return

        # Esperar a que el robot llegue al goal actual
        if not self.arrived:
            if self.goal_start_time is not None:
                now = self.get_clock().now()
                # Tiempo en segundos
                elapsed = (now.nanoseconds - self.goal_start_time.nanoseconds) / 1e9
                if elapsed > self.goal_timeout:
                    self.get_logger().warn(
                        f"Timeout ({elapsed:.1f}s) en waypoint {self.current_idx - 1}. "
                        "Saltando al siguiente waypoint.")
                    # Marcamos como 'llegado' para poder avanzar
                    self.arrived = True
                    self.goal_start_time = None
            # No publicamos un nuevo goal todavía
            return

        # Sin waypoints restantes
        if self.current_idx >= len(self.WAYPOINTS):
            self.get_logger().info("No quedan waypoints. Exploración finalizada.")
            self.timer.cancel()
            return

        # Tomar siguiente waypoint y publicarlo
        x, y = self.WAYPOINTS[self.current_idx]
        self.current_idx += 1
        self.publish_goal(x, y)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

