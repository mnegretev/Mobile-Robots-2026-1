import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class RosBasicsNode(Node):
    def __init__(self):
        print("INITIALIZING SIMPLE MOVE NODE ...")
        super().__init__("ros_basics_node")
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.callback_scan, 1)
        self.timer = self.create_timer(0.1, self.callback_timer)
        self.obstacle_detected = False

    def callback_timer(self):
        msg = Twist()
        msg.linear.x = 0.0 if self.obstacle_detected else 0.3
        self.pub_cmd_vel.publish(msg)

    def callback_scan(self, msg):
        self.obstacle_detected = msg.ranges[len(msg.ranges)//2] < 1.0


def main(args=None):
    rclpy.init(args=args)
    ros_basics_node = RosBasicsNode()
    rclpy.spin(ros_basics_node)
    ros_basics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
