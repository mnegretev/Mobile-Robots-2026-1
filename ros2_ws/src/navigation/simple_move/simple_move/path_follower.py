#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# PATH FOLLOWING
#

# Instructions:
# Write the code necessary to move the robot along a given path.
# Consider a differential base. Max linear and angular speeds
# must be 0.8 and 1.0 respectively.
#


import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from navig_msgs.srv import ProcessPath
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from ament_index_python.packages import get_package_share_directory
import math
import numpy
import time

NAME = "Rocio Fabiola Romero Bernal"

SM_INIT = 0
SM_WAIT_FOR_NEW_GOAL = 10
SM_PLAN_PATH = 20
SM_SMOOTH_PATH = 30
SM_FOLLOWING_PATH = 40
SM_SAVE_DATA = 50


class PathFollowerNode(Node):
    def _wrap_to_pi(self, angle: float) -> float:
        # Envuelve a (-pi, pi]
        a = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return a if a != -math.pi else math.pi
    
    def calculate_control(self, robot_x, robot_y, robot_a,
                          goal_x, goal_y, alpha, beta, v_max, w_max):
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        desired_a = math.atan2(dy, dx)

        # Error de ángulo envuelto a (-pi, pi]
        error_a = self._wrap_to_pi(desired_a - robot_a)

        # Ley de control 
        v = v_max * math.exp(-(error_a * error_a) / max(alpha, 1e-6))
        w = w_max * (2.0 / (1.0 + math.exp(-(error_a) / max(beta, 1e-6))) - 1.0)

        return [v, w]

    def follow_path(self, path, alpha, beta, v_max, w_max, tol):
        """
        Sigue el path y devuelve True si termina a una distancia <= tol
        del último punto; False en caso contrario.
        """
        if path is None or len(path) == 0:
            self.get_logger().warn("Ruta vacía")
            return False

        switch_dist = 0.3
        goal_idx = 0
        goal_x, goal_y = float(path[goal_idx][0]), float(path[goal_idx][1])

        # Posición inicial del robot
        robot_p, robot_a = self.get_robot_pose()
        robot_x, robot_y = float(robot_p[0]), float(robot_p[1])

        # Función de distancia euclidiana
        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        last_point = (float(path[-1][0]), float(path[-1][1]))

        reached = False

        while rclpy.ok():
            # condición de salida principal
            if dist((robot_x, robot_y), last_point) <= tol:
                reached = True
                break

            # Control
            v, w = self.calculate_control(robot_x, robot_y, robot_a,
                                          goal_x, goal_y, alpha, beta, v_max, w_max)

            self.publish_and_save_data(robot_x, robot_y, robot_a, goal_x, goal_y, v, w)

            rp, robot_a = self.get_robot_pose()
            robot_x, robot_y = float(rp[0]), float(rp[1])

            # Avanzar al siguiente punto del path
            if dist((robot_x, robot_y), (goal_x, goal_y)) < switch_dist and goal_idx < len(path) - 1:
                goal_idx += 1
                goal_x, goal_y = float(path[goal_idx][0]), float(path[goal_idx][1])

        # Parada suave al final del intento (lo haya logrado o no)
        self.publish_and_save_data(robot_x, robot_y, robot_a,
                                   last_point[0], last_point[1], 0.0, 0.0)
        return reached

    def publish_and_save_data(self, robot_x, robot_y, robot_a, goal_x, goal_y, v, w):
        self.nav_data.append([robot_x, robot_y, robot_a, goal_x, goal_y, v, w])
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub_cmd_vel.publish(msg)
        rclpy.spin_once(self)
        time.sleep(0.001)

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
            robot_pose = numpy.asarray([robot_x, robot_y])
            robot_a = math.atan2(t.transform.rotation.z, t.transform.rotation.w) * 2
            self.robot_pose = robot_pose
            self.robot_a = robot_a
        except TransformException as ex:
            print("Could not get robot pose")
            robot_pose = numpy.asarray([0.0, 0.0])
            robot_a = 0.0
        return robot_pose, robot_a

    def callback_goal_pose(self, msg):
        self.nav_data = []
        self.goal_pose = numpy.asarray([msg.pose.position.x, msg.pose.position.y])
        print("Received new goal pose: ", self.goal_pose)
        self.new_goal_pose = True
    
    def callback_object_detected(self, msg: Bool):
        if msg.data:
            self.get_logger().warn("🟥 OBJETO DETECTADO — Deteniendo navegación")
            self.object_detected = True

    def __init__(self):
        print("INITIALIZING PATH FOLLOWER NODE ...")
        super().__init__("path_follower_node")
        self.nav_data = []
        self.data_file = get_package_share_directory('simple_move') + "/data.txt"
        self.robot_pose = numpy.asarray([0.0, 0.0])
        self.robot_a = 0.0
        self.new_goal_pose = False
        self.goal_pose = numpy.asarray([0.0, 0.0])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('w_max', 0.5)
        self.declare_parameter('alpha', 1.0)
        self.declare_parameter('beta', 1.0)
        self.declare_parameter('tol', 0.3)

        self.clt_plan_path = self.create_client(GetPlan, '/path_planning/plan_path')
        self.clt_smooth_path = self.create_client(ProcessPath, '/path_planning/smooth_path')
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_goal_reached = self.create_publisher(Bool, '/navigation/goal_reached', 1)
        self.sub_goal_pose = self.create_subscription(
            PoseStamped, '/goal_pose', self.callback_goal_pose, 1
        )

        # ---- DETECCIÓN DE OBJETOS (YOLO) ----
        self.object_detected = False
        self.sub_object_detected = self.create_subscription(
            Bool, '/object_detected', self.callback_object_detected, 1
        )

    def spin(self):
        robot_pose_tf_ready = False
        print("Waiting for plan path service...")
        while not self.clt_plan_path.wait_for_service(timeout_sec=1.0):
            print('Waiting for plan path service...')
        print("Plan path service is now available...")
        clt_timeout = 5
        print("Waiting for smooth path service...")
        while not self.clt_smooth_path.wait_for_service(timeout_sec=0.5) and clt_timeout > 0:
            print("Waiting for smooth path service...")
            clt_timeout -= 1
        if self.clt_smooth_path.wait_for_service(timeout_sec=0.5):
            print("Smooth path service is now available")
        else:
            print("Smooth path service is not available")
        print("Waiting for robot pose tf to be available")
        while rclpy.ok() and not robot_pose_tf_ready:
            try:
                self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
                robot_pose_tf_ready = True
            except TransformException as ex:
                robot_pose_tf_ready = False
            rclpy.spin_once(self)
            time.sleep(0.001)
        print("Robot pose tf is now available")

        state = SM_INIT
        path = None

        while rclpy.ok():
            robot_p, robot_a = self.get_robot_pose()

            if state == SM_INIT:
                print("Ready to execute new path. Waiting for new goal...")
                state = SM_WAIT_FOR_NEW_GOAL

            elif state == SM_WAIT_FOR_NEW_GOAL:
                if self.new_goal_pose:
                    self.new_goal_pose = False
                    state = SM_PLAN_PATH

            elif state == SM_PLAN_PATH:
                print("Trying to plan path from", self.robot_pose, "to", self.goal_pose)
                request = GetPlan.Request()
                request.start.pose.position.x = self.robot_pose[0]
                request.start.pose.position.y = self.robot_pose[1]
                request.goal.pose.position.x = self.goal_pose[0]
                request.goal.pose.position.y = self.goal_pose[1]

                future = self.clt_plan_path.call_async(request)
                rclpy.spin_until_future_complete(self, future)
                # --- manejo de errores del servicio / path vacío ---
                if future.exception() is not None or future.result() is None:
                    self.get_logger().error("Error al llamar a plan_path, no se planeó ruta")
                    path = None
                    # no publicamos goal_reached; el Explorer hará timeout y mandará otro punto
                    state = SM_SAVE_DATA
                    continue

                path = future.result().plan
                n_pts = len(path.poses)
                print("Path planned with", n_pts, "points")

                if n_pts == 0:
                    # A* no pudo encontrar camino para este goal
                    self.get_logger().warn("Planner no pudo encontrar ruta a este goal (path vacío)")
                    path = None
                    # no publicamos goal_reached; dejamos que el Explorer haga timeout
                    self.pub_cmd_vel.publish(Twist())  # detener robot por si se movió algo
                    state = SM_SAVE_DATA
                else:
                    state = SM_SMOOTH_PATH

            elif state == SM_SMOOTH_PATH:
                if path is None or len(path.poses) == 0:
                    state = SM_INIT
                    continue

                req = ProcessPath.Request()
                req.path = path
                if self.clt_smooth_path.wait_for_service(timeout_sec=0.1):
                    future = self.clt_smooth_path.call_async(req)
                    rclpy.spin_until_future_complete(self, future)
                    if future.exception() is None and future.result() is not None:
                        path = future.result().processed_path
                        print("Path smoothed succesfully")
                    else:
                        self.get_logger().warn("Fallo al suavizar path, usando path original")
                else:
                    print("Smooth path service is not available")
                state = SM_FOLLOWING_PATH

            elif state == SM_FOLLOWING_PATH:
                if path is None or len(path.poses) == 0:
                    state = SM_INIT
                    continue

                v_max = self.get_parameter('v_max').get_parameter_value().double_value
                w_max = self.get_parameter('w_max').get_parameter_value().double_value
                alpha = self.get_parameter('alpha').get_parameter_value().double_value
                beta  = self.get_parameter('beta').get_parameter_value().double_value
                tol   = self.get_parameter('tol').get_parameter_value().double_value
                print("Following path with [v_max, w_max, alpha, beta, tol] =",
                      [v_max, w_max, alpha, beta, tol])

                path_points = [numpy.asarray([p.pose.position.x, p.pose.position.y])
                               for p in path.poses]
                reached = self.follow_path(path_points, alpha, beta, v_max, w_max, tol)

                self.pub_cmd_vel.publish(Twist())

                if reached:
                    # Solo aquí mandamos True: el Explorer avanzará sin usar el timeout
                    self.pub_goal_reached.publish(Bool(data=True))
                    print("Global goal point reached (distancia <= tol)")
                else:
                    print("Global goal NOT reached (distancia > tol); no se publica goal_reached")

                state = SM_SAVE_DATA

            elif state == SM_SAVE_DATA:
                s = ""
                for d in self.nav_data:
                    s += (
                        str(d[0]) + "," + str(d[1]) + "," + str(d[2]) + "," +
                        str(d[3]) + "," + str(d[4]) + "," + str(d[5]) + "," +
                        str(d[6]) + "\n"
                    )
                f = open(self.data_file, "w")
                f.write(s)
                f.close()
                state = SM_INIT
                
            rclpy.spin_once(self)
            time.sleep(0.001)


def main(args=None):
    rclpy.init(args=args)
    path_follower_node = PathFollowerNode()
    path_follower_node.spin()
    path_follower_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

