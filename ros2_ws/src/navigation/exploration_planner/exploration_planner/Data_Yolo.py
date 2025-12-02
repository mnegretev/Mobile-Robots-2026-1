#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class YoloDatasetNode(Node):
    def __init__(self):
        super().__init__('yolo_dataset_node')

        # Carpeta de dataset
        self.image_folder = 'dataset_yolo/images'
        self.label_folder = 'dataset_yolo/labels'
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.label_folder, exist_ok=True)

        self.bridge = CvBridge()
        self.counter = 0

        # Objetos a capturar
        self.object_names = ['coke_can', 'cheezit_big_original', 'aws_robomaker_residential_Trash_01']
        self.object_classes = {name: idx for idx, name in enumerate(self.object_names)}

        # Suscripciones
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.cb_image, 10)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.cb_camera_info, 10)
        self.sub_models = self.create_subscription(ModelStates, '/gazebo/model_states', self.cb_models, 10)

        self.model_positions = {}
        self.camera_matrix = None

    def cb_camera_info(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'Camera intrinsics received: {self.camera_matrix}')

    def cb_models(self, msg: ModelStates):
        # Guardamos la posición de los objetos
        for name, pose in zip(msg.name, msg.pose):
            if name in self.object_names:
                self.model_positions[name] = pose

    def project_to_image(self, x, y, z):
        """Proyección simple de 3D a 2D usando cámara pinhole"""
        if self.camera_matrix is None:
            return None
        # Simple proyección (sin distorsión)
        point = np.array([x, y, z])
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        if z <= 0.0:
            return None
        u = fx * x / z + cx
        v = fy * y / z + cy
        return int(u), int(v)

    def cb_image(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = cv_img.shape

        label_lines = []

        for name, pose in self.model_positions.items():
            # Tomamos posición relativa a la cámara
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            pixel = self.project_to_image(x, y, z)
            if pixel is None:
                continue
            u, v = pixel

            # Bounding box fijo (ejemplo 50x50 pix)
            box_w = 50
            box_h = 50

            x_center = (u) / w
            y_center = (v) / h
            bw = box_w / w
            bh = box_h / h

            cls = self.object_classes[name]
            label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            # Dibujar para verificación
            cv2.rectangle(cv_img, (u - box_w//2, v - box_h//2), (u + box_w//2, v + box_h//2), (0, 0, 255), 2)
            cv2.putText(cv_img, name, (u - 20, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Guardar imagen
        image_path = os.path.join(self.image_folder, f'image_{self.counter:04d}.png')
        label_path = os.path.join(self.label_folder, f'image_{self.counter:04d}.txt')
        cv2.imwrite(image_path, cv_img)
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
        self.get_logger().info(f'Saved {image_path} with {len(label_lines)} labels')
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = YoloDatasetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
