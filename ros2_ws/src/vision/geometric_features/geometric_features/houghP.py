#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# PROBABILISTIC HOUGH LINE DETECTOR
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy
import cv2
import math

FULL_NAME = "Popoca Zúñiga Daniel Ixbalanque"

class HoughPNode(Node):
    def callback_img(self, msg):
        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_houghP = img_bgr.copy()

        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detector de bordes Canny
        edges = cv2.Canny(img_gray, self.canny_lower, self.canny_upper)

        # Medir tiempo SOLO del detector de líneas
        t0 = self.get_clock().now()
        linesP = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap
        )
        t1 = self.get_clock().now()
        dt = (t1 - t0).nanoseconds / 1e6  # milisegundos

        # Dibujar líneas
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_houghP, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostrar tiempo en la imagen
        cv2.putText(img_houghP,
                    f"Tiempo: {dt:.2f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        cv2.imshow("BGR Original", img_bgr)
        cv2.imshow("HoughP", img_houghP)
        cv2.waitKey(1)

    def __init__(self):
        print("INITIALIZING PROBABILISTIC HOUGH NODE - ", FULL_NAME)
        super().__init__("houghP_node")

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
        self.declare_parameter("min_len", 20)
        self.declare_parameter("max_gap", 10)

        self.canny_lower = self.get_parameter("canny_l").value
        self.canny_upper = self.get_parameter("canny_u").value
        self.rho = self.get_parameter("rho").value
        self.theta = self.get_parameter("theta").value
        self.hough_threshold = self.get_parameter("hough_th").value
        self.min_length = self.get_parameter("min_len").value
        self.max_gap = self.get_parameter("max_gap").value

        print("Starting line detection with parameters: ")
        print("[Canny lower, Canny upper] =", [self.canny_lower, self.canny_upper])
        print("[rho, theta, Hough threshold] =", [self.rho, self.theta, self.hough_threshold])
        print("[min length, max gap] =", [self.min_length, self.max_gap])

def main(args=None):
    rclpy.init(args=args)
    houghP_node = HoughPNode()
    rclpy.spin(houghP_node)
    houghP_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
