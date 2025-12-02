import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker
import cv2
import os
import math
import numpy as np



class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Puente ROS <-> OpenCV
        self.bridge = CvBridge()

        # Cargar modelo YOLO
        self.model = YOLO("yolo11n.pt")
        self.model.overrides['verbose'] = False   # silenciar logs internos
        
        #Publicador a la parte de Marker
        self.marker_pub = self.create_publisher(Marker, '/detected_markers',10)
        # Publicación de bandera "todos los objetos encontrados"
        self.flags_pub = self.create_publisher(Bool, '/all_objects_found', 10)
        
        # Suscripción a la cámara del robot
        self.subscription = self.create_subscription(Image,'/camera/image_raw', self.image_callback,10)

        # TF para obtener la pose del robot en el frame "map"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Clases de interés y filtros de área
        self.target_classes = ["bottle", "book", "cup", "vase"]
        self.min_area_cereal = 2442
        self.min_area_bottle = 578

        # Banderas de detección (se activan cuando se ve cada objeto)
        self.found_cereal = False   # book
        self.found_bottle = False   # bottle
        self.found_trash = False    # cup

        # Banderas para guardar imagen solo la primera vez
        self.saved_cereal = False
        self.saved_bottle = False
        self.saved_trash = False

        # Lista en memoria con las detecciones (clase + pose robot)
        self.detected_positions = []   # (cls_name, x, y, theta)

        # Carpeta donde guardar imágenes y txt -> usar HOME del usuario
        home_dir = os.path.expanduser("~")
        self.save_dir = os.path.join(home_dir, "obj_imgs")
        os.makedirs(self.save_dir, exist_ok=True)

    # --------- Guardar una detección en TXT ---------

    def save_detection(self, cls_name, x, y, th):
        """
        Guarda en un archivo .txt la detección:
        nombre de clase y pose del robot (x, y, theta) en 'map'.
        """
        file_path = os.path.join(self.save_dir, "detecciones_robot.txt")
        with open(file_path, "a") as f:f.write(f"{cls_name},{x},{y},{th}")
        self.get_logger().info(f"Guardada detección de {cls_name} en {file_path}")

    # ---------- Utilidad: obtener pose del robot ----------
 
    def publish_pin_marker(self, cls_name, x, y, th):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detected_objects"
        marker.id = hash(cls_name) % 1000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        if cls_name == "book":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif cls_name == "bottle":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:  # cup
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

        marker.color.a = 1.0
        self.marker_pub.publish(marker)
    def get_robot_pose(self):
        """
        Devuelve (x, y, theta) del robot en el frame 'map'
        usando TF. Si falla, devuelve None.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            theta = 2.0 * math.atan2(
                t.transform.rotation.z,
                t.transform.rotation.w
            )
            return x, y, theta
        except TransformException:
            # Si TF aún no está listo, devolvemos None sin spamear logs
            return None

    # ---------- Callback de imagen ----------

    def image_callback(self, msg: Image):
        """
        Se ejecuta en cada imagen de la cámara:
        - Convierte a OpenCV
        - Corre YOLO
        - Filtra detecciones de interés
        - Dibuja bbox y texto la primera vez que ve cada objeto
        - Guarda imagen y pose del robot en .txt
        - Publica bandera /all_objects_found
        """
        objetos_detectados = False

        # 1) ROS Image -> OpenCV (BGR8)
        cv_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

        # 2) YOLO sobre la imagen
        results = self.model(cv_image)
        r = results[0]

        # 3) Recorrer todas las cajas detectadas
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])

            # Coordenadas del bounding box (en pixeles)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # --- BOOK (cereal) ---
            if cls_name == "book" and conf > 0.4 and area > self.min_area_cereal:
                self.get_logger().info(f"Objeto interes: {cls_name} (conf={conf:.2f})")
                self.found_cereal = True
                objetos_detectados = True

                # Solo la primera vez que se detecta
                if not self.saved_cereal:
                    # Dibujar bbox y texto sobre la imagen
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(cv_image, "book", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

                    # Guardar imagen en disco
                    img_path = os.path.join(self.save_dir, "book_1.png")
                    cv2.imwrite(img_path, cv_image)

                    # Obtener y guardar pose del robot
                    pose = self.get_robot_pose()
                    if pose is not None:
                        x_r, y_r, th_r = pose
                        self.detected_positions.append(
                            ("book", x_r, y_r, th_r)
                        )
                        self.save_detection("book", x_r, y_r, th_r)
                        self.publish_pin_marker("book", x_r, y_r, th_r)
                        self.get_logger().info(
                            f"book visto en pose robot: ({x_r:.2f}, {y_r:.2f})"
                        )

                    self.saved_cereal = True

            # --- BOTTLE ---
            if cls_name == "bottle" and conf > 0.3 and area > self.min_area_bottle:
                self.get_logger().info(
                    f"Objeto interes: {cls_name} (conf={conf:.2f})"
                )
                self.found_bottle = True
                objetos_detectados = True

                if not self.saved_bottle:
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2),
                                  (255, 0, 0), 2)
                    cv2.putText(cv_image, "bottle", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0), 2)

                    img_path = os.path.join(self.save_dir, "bottle_1.png")
                    cv2.imwrite(img_path, cv_image)

                    pose = self.get_robot_pose()
                    if pose is not None:
                        x_r, y_r, th_r = pose
                        self.detected_positions.append(
                            ("bottle", x_r, y_r, th_r)
                        )
                        self.save_detection("bottle", x_r, y_r, th_r)
                        self.publish_pin_marker("bottle", x_r, y_r, th_r)
                        self.get_logger().info(
                            f"bottle visto en pose robot: ({x_r:.2f}, {y_r:.2f})"
                        )

                    self.saved_bottle = True

            # --- CUP (trash) ---
            if cls_name == "cup" and conf > 0.5:
                self.get_logger().info(
                    f"Objeto interes: {cls_name} (conf={conf:.2f})"
                )
                self.found_trash = True
                objetos_detectados = True

                if not self.saved_trash:
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(cv_image, "cup", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                    img_path = os.path.join(self.save_dir, "cup_1.png")
                    cv2.imwrite(img_path, cv_image)

                    pose = self.get_robot_pose()
                    if pose is not None:
                        x_r, y_r, th_r = pose
                        self.detected_positions.append(
                            ("cup", x_r, y_r, th_r)
                        )
                        self.save_detection("cup", x_r, y_r, th_r)
                        self.publish_pin_marker("cup", x_r, y_r, th_r)
                        self.get_logger().info(
                            f"cup visto en pose robot: ({x_r:.2f}, {y_r:.2f})"
                        )

                    self.saved_trash = True

        # Loguear solo cuando se detectó al menos un objeto en este frame
        if objetos_detectados:
            status = (
                f"Cereal: {self.found_cereal}, "
                f"Bottle: {self.found_bottle}, "
                f"Trash: {self.found_trash}"
            )
            self.get_logger().info(f"OBJETOS: {status}")

        # Publicar bandera global: True cuando ya se encontraron los 3
        all_found = (
            self.found_cereal and
            self.found_bottle and
            self.found_trash
        )
        msg_flags = Bool()
        msg_flags.data = all_found
        self.flags_pub.publish(msg_flags)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
