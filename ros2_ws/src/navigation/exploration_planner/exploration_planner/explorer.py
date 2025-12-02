
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import math
import numpy as np

def normalize_angle(a):
    while a > math.pi:
        a -= 2*math.pi
    while a <= -math.pi:
        a += 2*math.pi
    return a

class CellNavigator(Node):
    def __init__(self):
        super().__init__("cell_navigator")

        # Parámetros
        self.declare_parameter("v_max", 0.7)
        self.declare_parameter("w_max", 0.7)
        self.declare_parameter("pose_tol", 0.3)
        self.declare_parameter("control_freq", 1.0)
        self.declare_parameter("alpha", 0.22)
        self.declare_parameter("beta", 0.25)

        self.v_max = self.get_parameter("v_max").value
        self.w_max = self.get_parameter("w_max").value
        self.pose_tol = self.get_parameter("pose_tol").value
        self.alpha_param = self.get_parameter("alpha").value
        self.beta_param = self.get_parameter("beta").value

        # Subscribers / Publishers
        self.sub_markers = self.create_subscription(MarkerArray, "/cell_marker", self.cb_markers, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.pub_visited = self.create_publisher(MarkerArray, "/visited_points", 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Estado
        self.points = []
        self.visited = set()
        self.have_points = False
        self.current_goal = None

        period = 1.0 / self.get_parameter("control_freq").value
        self.timer = self.create_timer(period, self.control_loop)
        self.get_logger().info("CellNavigator iniciado.")

    # Recibir puntos
    def cb_markers(self, msg):
        pts = []
        for m in msg.markers:
            x = round(m.pose.position.x, 4)
            y = round(m.pose.position.y, 4)
            if (x, y) not in pts and (x, y) not in self.visited:
                pts.append((x, y))
        self.points = pts
        self.have_points = len(pts) > 0

    # Obtener pose
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except TransformException:
            return None, None
        x = t.transform.translation.x
        y = t.transform.translation.y
        qx = t.transform.rotation.x
        qy = t.transform.rotation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w
        siny_cosp = 2*(qw*qz + qx*qy)
        cosy_cosp = 1 - 2*(qy*qy + qz*qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([x, y]), yaw

    # Selección de objetivo
    def estimate_time(self, robot_pos, robot_yaw, point):
        px, py = point
        dx = px - robot_pos[0]
        dy = py - robot_pos[1]
        dist = math.hypot(dx, dy)
        t_lin = dist / self.v_max
        desired_yaw = math.atan2(dy, dx)
        dtheta = normalize_angle(desired_yaw - robot_yaw)
        t_rot = abs(dtheta) / self.w_max
        return t_lin + t_rot

    def select_best_point(self, robot_pos, robot_yaw):
        if not self.points:
            return None
        best_pt = None
        best_time = float("inf")
        for pt in self.points:
            t = self.estimate_time(robot_pos, robot_yaw, pt)
            if t < best_time:
                best_time = t
                best_pt = pt
        return best_pt

    # Enviar goal
    def send_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)

    # Control loop
    def control_loop(self):
        if not self.have_points:
            return
        robot_pos, robot_yaw = self.get_robot_pose()
        if robot_pos is None:
            return

        if self.current_goal is None:
            best_pt = self.select_best_point(robot_pos, robot_yaw)
            if best_pt is None:
                return
            self.current_goal = best_pt
            self.send_goal(best_pt[0], best_pt[1])

        gx, gy = self.current_goal
        dist = math.hypot(gx - robot_pos[0], gy - robot_pos[1])

        if dist <= self.pose_tol:
            self.visited.add(self.current_goal)
            self.points = [p for p in self.points if p != self.current_goal]
            self.current_goal = None

            # Publicar marker rojo
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "visited_marker"
            marker.id = len(self.visited)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = gx
            marker.pose.position.y = gy
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            ma = MarkerArray()
            ma.markers.append(marker)
            self.pub_visited.publish(ma)

            self.pub_cmd_vel.publish(Twist())
            return

        # Leyes del PDF
        desired_yaw = math.atan2(gy - robot_pos[1], gx - robot_pos[0])
        e_a = normalize_angle(desired_yaw - robot_yaw)
        v = self.v_max * math.exp(-abs(e_a) / self.alpha_param)
        w = self.w_max * (1.0 / (1.0 + math.exp(-e_a / self.beta_param)) - 0.5)

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = CellNavigator()
    rclpy.spin(node)
    node.destroy




