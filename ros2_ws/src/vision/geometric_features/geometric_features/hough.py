#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# HOUGH LINE DETECTOR
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy
import cv2
import math

FULL_NAME = "Popoca Zuñiga Daniel Ixbalanque"

class HoughNode(Node):
    def callback_img(self, msg):
        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_hough = img_bgr.copy()

        # Convertir imagen a gris
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Borde Canny
        edges = cv2.Canny(img_gray, self.canny_lower, self.canny_upper)

        # Medir tiempo de detección de líneas
        t0 = self.get_clock().now()
        lines = cv2.HoughLines(edges, self.rho, self.theta, self.hough_threshold)
        t1 = self.get_clock().now()
        dt = (t1 - t0).nanoseconds / 1e6  # ms

        # Dibujar líneas
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho

                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Escribir el tiempo en pantalla
        cv2.putText(img_hough,
                    f"Tiempo: {dt:.2f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        cv2.imshow("BGR Original", img_bgr)
        cv2.imshow("Hough", img_hough)
        cv2.waitKey(1)

    def __init__(self):
        print("INITIALIZING HOUGH NODE -", FULL_NAME)
        super().__init__("hough_node")

        self.br = CvBridge()
        self.sub_img = self.create_subscription(Image,
                                                '/camera/image_raw',
                                                self.callback_img,
                                                1)

        # Parámetros ROS2
        self.declare_parameter("canny_l", 10)
        self.declare_parameter("canny_u", 20)
        self.declare_parameter("rho", 10)
        self.declare_parameter("theta", 0.1)
        self.declare_parameter("hough_th", 20)

        self.canny_lower = self.get_parameter("canny_l").get_parameter_value().integer_value
        self.canny_upper = self.get_parameter("canny_u").get_parameter_value().integer_value
        self.rho = self.get_parameter("rho").get_parameter_value().integer_value
        self.theta = self.get_parameter("theta").get_parameter_value().double_value
        self.hough_threshold = self.get_parameter("hough_th").get_parameter_value().integer_value

        print("Starting line detection with parameters:")
        print("[Canny lower, Canny upper] =", [self.canny_lower, self.canny_upper])
        print("[rho, theta, Hough threshold] =", [self.rho, self.theta, self.hough_threshold])

def main(args=None):
    rclpy.init(args=args)
    hough_node = HoughNode()
    rclpy.spin(hough_node)
    hough_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
