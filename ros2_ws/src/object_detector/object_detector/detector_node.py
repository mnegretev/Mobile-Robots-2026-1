"""

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import os

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.get_logger().info("Nodo Object Detector iniciado - Muestra detecciones de tu modelo YOLO")
        
        # Ruta al modelo YOLO entrenado
        weights_path = os.path.join(os.path.dirname(__file__), "PF_RM_v2i2", "weights", "best.pt")
        self.get_logger().info(f"Cargando modelo: {weights_path}")
        self.model = YOLO(weights_path)
        
        # Configurar CvBridge
        self.bridge = CvBridge()
        
        # Suscribirse a la cámara (ROS2)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Cambia al topic de tu cámara
            self.image_callback,
            10
        )
        self.subscription  # para evitar warning de variable no usada

        # Publisher para los objetos detectados
        self.publisher_ = self.create_publisher(String, '/detected_objects', 10)

    def image_callback(self, msg):
        # Convertir de ROS Image a OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detección
        results = self.model(cv_image)[0]

        detected_labels = []
        for result in results.boxes.data.tolist():
            # result: [x1, y1, x2, y2, score, class]
            class_id = int(result[5])
            label = self.model.names[class_id]
            detected_labels.append(label)

        if detected_labels:
            detected_str = ','.join(detected_labels)
            self.get_logger().info(f"Objetos detectados: {detected_str}")
            self.publisher_.publish(String(data=detected_str))

        # Mostrar imagen con detecciones (opcional)
        annotated_frame = results.plot()
        cv2.imshow("Detecciones YOLO", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import os

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.get_logger().info("Nodo Object Detector iniciado - Guardará imágenes y posiciones de objetos detectados.")

        # ================================
        # Cargar modelo YOLO entrenado
        # ================================
        weights_path = os.path.join(
            os.path.dirname(__file__),
            "PF_RM_v2i2",
            "weights",
            "best.pt"
        )
        self.get_logger().info(f"Cargando modelo: {weights_path}")
        self.model = YOLO(weights_path)

        # Ruta donde guardar las imágenes detectadas y el archivo de texto
        self.save_path = "/home/catcyber02/Mobile-Robots-2026-1/ros2_ws/src/object_detector"
        self.text_file = os.path.join(self.save_path, "detecciones.txt")

        # Registrar qué objetos ya se guardaron (solo 1 vez)
        self.saved_objects = set()

        # Guardar máximo 3
        self.max_images = 3

        # Bridge ROS → OpenCV
        self.bridge = CvBridge()

        # Suscripciones
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Posición del robot inicial
        self.robot_x = 0.0
        self.robot_y = 0.0

        # Publicador de texto
        self.publisher_ = self.create_publisher(String, '/detected_objects', 10)

    # =============================================================
    #  Callback de posición del robot (topic /odom)
    # =============================================================
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    # =============================================================
    #  Callback de imagen (detección)
    # =============================================================
    def image_callback(self, msg):

        # Convertir a OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detección YOLO
        results = self.model(cv_image)[0]

        detected_labels = []

        # Procesar cajas detectadas
        for box in results.boxes:
            cls = int(box.cls[0])        # clase
            score = float(box.conf[0])   # confianza
            label = self.model.names[cls]

            detected_labels.append(label)

            # Condiciones para guardar la imagen y la posición
            if score >= 0.55 and label not in self.saved_objects and len(self.saved_objects) < self.max_images:
                
                # Crear imagen anotada con cajas, objetos y posición
                annotated = results.plot()

                # Agregar posición del robot
                pos_text = f"Robot pos: x={self.robot_x:.2f}, y={self.robot_y:.2f}"
                cv2.putText(
                    annotated,
                    pos_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                # ==========================================
                # Guardar imagen
                # ==========================================
                self.save_detection_image(annotated, label)

                # ==========================================
                # Guardar posición en archivo de texto
                # ==========================================
                self.save_detection_text(label)

                # Registrar objeto guardado
                self.saved_objects.add(label)
                self.get_logger().info(f"{label}: Imagen + posición guardadas.")

        # Publicar detecciones
        if detected_labels:
            msg_out = String()
            msg_out.data = ",".join(detected_labels)
            self.publisher_.publish(msg_out)

        # Mostrar imagen en pantalla
        annotated_frame = results.plot()
        cv2.putText(
            annotated_frame,
            f"Robot: {self.robot_x:.2f}, {self.robot_y:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )
        cv2.imshow("Detecciones YOLO", annotated_frame)
        cv2.waitKey(1)

    # =============================================================
    # Guardar imagen con nombre: objeto + posición
    # =============================================================
    def save_detection_image(self, image, label):
        x = round(self.robot_x, 2)
        y = round(self.robot_y, 2)

        filename = f"deteccion_{label}_{x}_{y}.jpg"
        filepath = os.path.join(self.save_path, filename)

        cv2.imwrite(filepath, image)
        self.get_logger().info(f"Imagen guardada en: {filepath}")

    # =============================================================
    # Guardar en archivo de texto
    # =============================================================
    def save_detection_text(self, label):
        x = round(self.robot_x, 2)
        y = round(self.robot_y, 2)

        line = f"{label} detectado en x={x}, y={y}\n"

        with open(self.text_file, "a") as f:
            f.write(line)

        self.get_logger().info(f"Posición guardada en archivo: {self.text_file}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
