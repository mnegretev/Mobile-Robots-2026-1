import rclpy
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
        self.status_sub = self.create_subscription(
            GoalStatusArray, '/goal_status', self.status_callback, 10
        )

        # Mapa para conocer límites (pero sin mirar valores de celdas)
        self.map_info = None
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # Paso de la rejilla
        self.STEP = 0.5  # m

        self.WAYPOINTS = []
        self.current_idx = 0

        # Esperar a tener /map
        self.get_logger().info("Esperando /map para calcular límites de la rejilla...")
        while rclpy.ok() and self.map_info is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        # Configurar rejilla sobre TODO el mapa
        self.setup_grid_from_map()

        self.get_logger().info(
            f"Explorer (grid desde OccupancyGrid): generados {len(self.WAYPOINTS)} waypoints."
        )

        # Bucle principal
        self.timer = self.create_timer(0.5, self.main_loop)

    # ---------- Callbacks ----------

    def flags_callback(self, msg: Bool):
        self.all_found = msg.data

    def status_callback(self, msg: GoalStatusArray):
        if msg.status_list:
            last_status = msg.status_list[-1]
            if last_status.status == 3:
                self.arrived = True

    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info

    # ---------- Configurar rejilla desde OccupancyGrid ----------

    def setup_grid_from_map(self):
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        res = self.map_info.resolution
        w   = self.map_info.width
        h   = self.map_info.height

        self.X_MIN = ox
        self.X_MAX = ox + w * res
        self.Y_MIN = oy
        self.Y_MAX = oy + h * res

        self.get_logger().info(
            f"Límites mapa: X[{self.X_MIN:.2f}, {self.X_MAX:.2f}], "
            f"Y[{self.Y_MIN:.2f}, {self.Y_MAX:.2f}] (res={res:.3f})"
        )

        self.WAYPOINTS = self.generate_grid_waypoints(
            self.X_MIN, self.X_MAX, self.Y_MIN, self.Y_MAX, self.STEP
        )

    # ---------- Generación de rejilla ----------

    def generate_grid_waypoints(self, x_min, x_max, y_min, y_max, step):
        """
        Rejilla serpentina sobre todo el mapa:
        y = y_min .. y_max
        x = x_min .. x_max, alternando izq→der y der→izq.
        """
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
        # Si ya se encontraron los 3 objetos, detener exploración
        if self.all_found:
            self.get_logger().info("Los 3 objetos ya fueron encontrados. Exploración detenida.")
            self.timer.cancel()
            return

        # Esperar a que el robot llegue al goal actual
        if not self.arrived:
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
