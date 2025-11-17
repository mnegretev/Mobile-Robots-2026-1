#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# HOUGH LINE DETECTOR
#
# Instructions:
# Complete the code necessary to implement the Hough line
# detector using the corresponding OpenCV function
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy
import cv2
import math

FULL_NAME = "Axel Jovani Ruiz Martínez"

class HoughNode(Node):
    def callback_img(self, msg):
        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_hough = img_bgr.copy()

        # 1) Gris + Canny (una sola vez)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, self.canny_lower, self.canny_upper)
        edge_count = int(numpy.count_nonzero(edges))
        EPR = edge_count / edges.size if edges.size > 0 else 0.0

        # Dar un poco de tolerancia a los bordes para empatar con grosor de líneas
        edges_d = cv2.dilate(edges, numpy.ones((3,3), numpy.uint8), iterations=1)

        # 2) Hough (medimos solo esta parte)
        t0 = self.get_clock().now()
        lines = cv2.HoughLines(edges, rho=self.rho, theta=self.theta, threshold=self.hough_threshold)
        t1 = self.get_clock().now()
        elapsed_ms = (t1 - t0).nanoseconds / 1e6

        # 3) Dibujar líneas + máscara de unión de líneas (una sola vez)
        line_union = numpy.zeros_like(edges_d, dtype=numpy.uint8)
        count = 0
        if lines is not None:
            count = len(lines)
            for l in lines:
                rho, theta = l[0]
                a, b = math.cos(theta), math.sin(theta)
                x0, y0 = a*rho, b*rho
                x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
                x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
                # visual
                cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # unión de líneas
                cv2.line(line_union, (x1, y1), (x2, y2), 255, 2)

        # 4) LCR: fracción de bordes cubiertos por alguna línea
        covered = cv2.countNonZero(cv2.bitwise_and(edges_d, line_union))
        LCR = covered / edge_count if edge_count > 0 else 0.0

        # 5) Overlay + log
        info = f"Hough: {elapsed_ms:.2f} ms | lines: {count} | LCR: {LCR:.3f} | EPR: {EPR:.3f}"
        cv2.putText(img_hough, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        self.get_logger().info(info)

        # Mostrar
        cv2.imshow("BGR Original", img_bgr)
        cv2.imshow("Hough", img_hough)  # (typo corregido)
        cv2.waitKey(1)

    
    def __init__(self):
        print("INITIALIZING HOUGH NODE - ", FULL_NAME)
        super().__init__("hough_node")
        self.br = CvBridge()
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.callback_img, 1)
        self.declare_parameter("canny_l",10)
        self.declare_parameter("canny_u",20)
        self.declare_parameter("rho", 10)
        self.declare_parameter("theta", 0.1)
        self.declare_parameter("hough_th", 20)
        self.canny_lower = self.get_parameter("canny_l").get_parameter_value().integer_value
        self.canny_upper = self.get_parameter("canny_u").get_parameter_value().integer_value
        self.rho   = self.get_parameter("rho").get_parameter_value().integer_value
        self.theta = self.get_parameter("theta").get_parameter_value().double_value
        self.hough_threshold = self.get_parameter("hough_th").get_parameter_value().integer_value
        print("Starting line detection with parameters: ")
        print("[Canny lower, Canny upper]=", [self.canny_lower, self.canny_upper])
        print("[rho, theta, Hough threshold]=", [self.rho, self.theta, self.hough_threshold])

def main(args=None):
    rclpy.init(args=args)
    hough_node = HoughNode()
    rclpy.spin(hough_node)
    hough_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
