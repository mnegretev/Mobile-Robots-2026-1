import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class PathFollowerNode(Node):
    def __init__(self):
        print("INITIALIZING PATH FOLLOWER NODE ...")
        super().__init__("path_follower_node")
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.timer = self.create_timer(0.1, self.callback_timer)
        self.obstacle_detected = False

    def callback_timer(self):
        msg = Twist()
        self.pub_cmd_vel.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    path_follower_node = PathFollowerNode()
    rclpy.spin(path_follower_node)
    path_follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
