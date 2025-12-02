import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from action_msgs.msg import GoalStatusArray
import math


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')

        # Publicador de metas para el path_follower / Nav2
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Suscriptores
        self.all_found = False
        self.flags_sub = self.create_subscription(
            Bool, '/all_objects_found', self.flags_callback, 10
        )

        self.object_positions = []
        self.objects_sub = self.create_subscription(
            PoseArray, '/detected_objects', self.objects_callback, 10
        )

        self.status_sub = self.create_subscription(
            GoalStatusArray, '/goal_status', self.status_callback, 10
        )

        self.map_data = None
        self.map_info = None
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # Parámetros
        self.safety_radius = 0.6
        self.X_MAX = 4.0
        self.Y_MAX = 8.0
        self.STEP = 1.5

        # Generar rejilla bruta
        self.WAYPOINTS = self.generate_grid_waypoints(
            -1.0, self.X_MAX, -self.Y_MAX, self.Y_MAX, self.STEP
        )
        self.WAYPOINTS_VALIDOS = []

        self.arrived = True  # robot listo para nuevo goal
        self.current_idx = 0

        # Esperar mapa antes de filtrar
        self.get_logger().info("Esperando /map para filtrar waypoints...")
        while rclpy.ok() and self.map_info is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        self.filtrar_waypoints_por_mapa()

        self.get_logger().info(
            f"{len(self.WAYPOINTS)} waypoints generados, "
            f"{len(self.WAYPOINTS_VALIDOS)} dentro del mapa navegable."
        )

        self.timer = self.create_timer(0.5, self.main_loop)

    # ---------- Callbacks ----------

    def flags_callback(self, msg: Bool):
        self.all_found = msg.data

    def objects_callback(self, msg: PoseArray):
        self.object_positions = [(p.position.x, p.position.y) for p in msg.poses]

    def status_callback(self, msg: GoalStatusArray):
        # status 3 = SUCCEEDED
        if msg.status_list:
            last_status = msg.status_list[-1]
            if last_status.status == 3:
                self.arrived = True

    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg.data
        self.map_info = msg.info

    # ---------- Utilidades de mapa ----------

    def cell_from_world(self, x, y):
        if self.map_info is None or self.map_data is None:
            return None, None, None

        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        res = self.map_info.resolution
        w = self.map_info.width
        h = self.map_info.height

        mx = int((x - ox) / res)
        my = int((y - oy) / res)

        if 0 <= mx < w and 0 <= my < h:
            idx = my * w + mx
            value = self.map_data[idx]
            return mx, my, value
        else:
            return mx, my, None

    def es_navegable(self, x, y):
        mx, my, value = self.cell_from_world(x, y)
        if value is None:
            return False   # fuera de la imagen del mapa
        # típico: -1 desconocido, 0 libre, >=50 obstáculo/coste alto [web:21][web:31]
        if value >= 50:
            return False
        return True

    def filtrar_waypoints_por_mapa(self):
        self.WAYPOINTS_VALIDOS = []
        for (x, y) in self.WAYPOINTS:
            mx, my, value = self.cell_from_world(x, y)
            if self.es_navegable(x, y):
                self.WAYPOINTS_VALIDOS.append((x, y))
                self.get_logger().info(
                    f"KEEP WP ({x:.2f},{y:.2f}) -> celda ({mx},{my}) val={value}"
                )
            else:
                self.get_logger().info(
                    f"DROP  WP ({x:.2f},{y:.2f}) -> celda ({mx},{my}) val={value}"
                )

    # ---------- Otras utilidades ----------

    def generate_grid_waypoints(self, x_min, x_max, y_min, y_max, step):
        waypoints = []
        y = y_min
        flip = False
        while y <= y_max:
            xs = []
            x = x_min
            while x <= x_max:
                xs.append(x)
                x += step
            if flip:
                xs.reverse()
            for xg in xs:
                waypoints.append((xg, y))
            flip = not flip
            y += step
        return waypoints

    def is_near_object(self, x, y):
        for ox, oy in self.object_positions:
            if math.hypot(x - ox, y - oy) < self.safety_radius:
                return True
        return False

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

    # ---------- Bucle principal ----------

    def main_loop(self):
        if self.all_found:
            self.get_logger().info("Los 3 objetos ya fueron encontrados. Exploración detenida.")
            self.timer.cancel()
            return

        if not self.arrived:
            # Esperar a que el robot llegue al goal actual
            return

        if self.current_idx >= len(self.WAYPOINTS_VALIDOS):
            self.get_logger().info("No quedan waypoints válidos. Exploración finalizada.")
            self.timer.cancel()
            return

        x, y = self.WAYPOINTS_VALIDOS[self.current_idx]
        self.current_idx += 1

        if self.is_near_object(x, y):
            self.get_logger().info(
                f"Waypoint válido ({x:.2f}, {y:.2f}) cerca de objeto, se descarta en runtime."
            )
            return

        self.publish_goal(x, y)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
