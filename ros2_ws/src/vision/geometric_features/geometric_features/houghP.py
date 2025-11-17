#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# PROBABILISTIC HOUGH LINE DETECTOR
#
# Instructions:
# Complete the code necessary to implement the probabilistic Hough line
# detector using the corresponding OpenCV function
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy
import cv2
import math

FULL_NAME = "Axel Jovani Ruiz Martinez"

class HoughPNode(Node):
    def callback_img(self, msg):
        # Convertir de ROS Image a OpenCV BGR
        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_houghP = img_bgr.copy()

        # 1) Pasar a escala de grises
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2) Detectar bordes con Canny
        edges = cv2.Canny(img_gray, self.canny_lower, self.canny_upper)

        # 3) Medir tiempo de detección de líneas
        t_start = self.get_clock().now()
        linesP = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap
        )
        t_end = self.get_clock().now()

        # Duración en segundos
        dt = (t_end - t_start).nanoseconds / 1e9

        # 4) Dibujar líneas detectadas sobre img_houghP
        num_lines = 0
        if linesP is not None:
            num_lines = len(linesP)
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_houghP, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

        # 5) Escribir el tiempo de procesamiento y número de líneas en la imagen
        text = f"t = {dt*1000:.2f} ms  lines = {num_lines}"
        cv2.putText(
            img_houghP,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Mostrar ventanas
        cv2.imshow("BGR Original", img_bgr)
        cv2.imshow("Edges", edges)
        cv2.imshow("Hough P", img_houghP)
        cv2.waitKey(1)

    
    def __init__(self):
        print("INITIALIZING PROBABILISTIC HOUGH NODE - ", FULL_NAME)
        super().__init__("houghP_node")
        self.br = CvBridge()
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.callback_img, 1)
        self.declare_parameter("canny_l",10)
        self.declare_parameter("canny_u",20)
        self.declare_parameter("rho", 10)
        self.declare_parameter("theta", 0.1)
        self.declare_parameter("hough_th", 20)
        self.declare_parameter("min_len",1)
        self.declare_parameter("max_gap",1)
        self.canny_lower = self.get_parameter("canny_l").get_parameter_value().integer_value
        self.canny_upper = self.get_parameter("canny_u").get_parameter_value().integer_value
        self.rho   = self.get_parameter("rho").get_parameter_value().integer_value
        self.theta = self.get_parameter("theta").get_parameter_value().double_value
        self.hough_threshold = self.get_parameter("hough_th").get_parameter_value().integer_value
        self.min_length = self.get_parameter("min_len").get_parameter_value().integer_value
        self.max_gap = self.get_parameter("max_gap").get_parameter_value().integer_value
        print("Starting line detection with parameters: ")
        print("[Canny lower, Canny upper]=", [self.canny_lower, self.canny_upper])
        print("[rho, theta, Hough threshold]=", [self.rho, self.theta, self.hough_threshold])
        print("[min length, max gap]=",[self.min_length, self.max_gap])

def main(args=None):
    rclpy.init(args=args)
    houghP_node = HoughPNode()
    rclpy.spin(houghP_node)
    houghP_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
