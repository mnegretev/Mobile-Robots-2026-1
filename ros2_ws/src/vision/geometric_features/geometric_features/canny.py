#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# CANNY EDGE DETECTOR
#
# Instructions:
# Complete the code necessary to implement the Canny edge detector
# using the function provided by OpenCV
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy
import cv2

FULL_NAME = "Melissa Maruuati Matias Zavala"

class CannyNode(Node):
    def callback_img(self, msg):
        img_bin = numpy.zeros((480, 640))
        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #
        # TODO:
        # Change the color space of the image 'img_bgr' to grayscale
        # Get edges using the cv2.Canny function, use as parameters
        # the variables self.canny_lower and self.canny_upper
        # Store the resulting binary image in img_bin
        #
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        img_bin = cv2.Canny(img_gray, self.canny_lower, self.canny_upper)

        epr = numpy.sum(img_bin > 0) / img_bin.size
        
        contrast_sum = 0
        count = 0

        padded = numpy.pad(img_gray, ((1, 1), (1, 1)), mode='edge')
        
        for x in range(img_bin.shape[0]):
            for y in range(img_bin.shape[1]):
                if img_bin[x, y] != 0:
                   center = padded[x+1, y+1]
                   neighbors = padded[x:x+3, y:y+3]
                   neighbors = neighbors.flatten()
                   neighbors = neighbors[neighbors!=center]
                   local_contrast = numpy.mean(numpy.abs(neighbors - center))
                   contrast_sum += local_contrast
                   count += 1

        edge_contrast_index = (contrast_sum / count) if count > 0 else 0

        print(f"EPR: {epr:.4f}, Edge Contrast Index: {edge_contrast_index:.2f}")        
        #
        cv2.imshow("BGR Original", img_bgr)
        cv2.imshow("Canny", img_bin)
        cv2.waitKey(1)
    
    def __init__(self):
        print("INITIALIZING CANNY NODE - ", FULL_NAME)
        super().__init__("canny_node")
        self.br = CvBridge()
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.callback_img, 1)
        self.declare_parameter("canny_l",10)
        self.declare_parameter("canny_u",20)
        self.canny_lower = self.get_parameter("canny_l").get_parameter_value().integer_value
        self.canny_upper = self.get_parameter("canny_u").get_parameter_value().integer_value
        print("Starting border detection with parameters: ", [self.canny_lower, self.canny_upper])
        

def main(args=None):
    rclpy.init(args=args)
    canny_node = CannyNode()
    rclpy.spin(canny_node)
    canny_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
