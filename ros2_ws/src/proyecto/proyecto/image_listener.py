import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')

        # Convertidor ROS <-> OpenCV
        self.br = CvBridge()

        # Suscripción al tópico de la cámara
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',   # TOPICO
            self.callback,
            10
        )

        self.get_logger().info("Nodo de escucha de imagen iniciado.")

    def callback(self, msg):
        # Convertir imagen ROS a OpenCV
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Mostrar imagen
        cv2.imshow("Imagen desde ROS2", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

