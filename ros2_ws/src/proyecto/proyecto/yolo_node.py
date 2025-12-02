import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped   # <-- pose del robot
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from std_msgs.msg import Bool


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Bridge ROS2 <-> OpenCV
        self.br = CvBridge()

        # Suscripción a la cámara
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Suscripción a la pose del robot 
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',           # por ejemplo: /amcl_pose
            self.pose_callback,
            10
        )

        self.current_pose = None  # aquí guardamos la última pose conocida

        self.get_logger().info('Iniciando YOLO node, suscrito a /camera/image_raw y /amcl_pose')

        # Cargar modelo YOLO
        self.model = YOLO('/home/axel/yolov8n.pt')
        self.get_logger().info('Modelo YOLOv8n cargado.')

        # Clases permitidas
        self.allowed_classes = ["cup", "bottle", "book"]

        # Rutas para guardar resultados
        self.base_dir = '/home/axel/detections'
        self.img_dir = os.path.join(self.base_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)

        self.log_path = os.path.join(self.base_dir, 'detections.txt')

        self.get_logger().info(f"Guardando imágenes en: {self.img_dir}")
        self.get_logger().info(f"Guardando log en: {self.log_path}")
    
        self.object_pub = self.create_publisher(Bool, '/object_detected', 10)
        self.object_detected = False


    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Guardar la última pose del robot."""
        self.current_pose = msg.pose.pose


    def image_callback(self, msg: Image):
        # Imagen ROS2 -> OpenCV
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Inferencia YOLO
        results = self.model(frame, verbose=False)[0]

        filtered_boxes = []
        detected_names = []

        # --- Filtrar detecciones ---
        for box in results.boxes:
            cls_id = int(box.cls.cpu().item())
            name = results.names[cls_id]

            if name not in self.allowed_classes:
                continue  # ignorar todo lo demás

            # Renombrar "book" -> "cerealbox"
            if name == "book":
                name = "cerealbox"

            detected_names.append(name)
            filtered_boxes.append(box)

        # --- Si no se detectó nada de interés, solo mostrar imagen y salir ---
        annotated = frame.copy()

        if not filtered_boxes:
            cv2.imshow("YOLO detections (filtrado)", annotated)
            cv2.waitKey(1)
            return

        # --- Dibujar SOLO las cajas filtradas ---
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
            cls_id = int(box.cls.cpu().item())
            label = results.names[cls_id]

            if label == "book":
                label = "cerealbox"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # --- Imprimir en consola ---
        self.get_logger().info(f"Detecté: {detected_names}")
        # --- Avisar que se encontró al menos un objeto de interés ---
        if not self.object_detected:
            self.object_detected = True
            msg = Bool()
            msg.data = True
            self.object_pub.publish(msg)
            self.get_logger().info("PUBLICANDO /object_detected = True")
        # --- Guardar imagen + posición del robot ---
        self.save_detection(annotated, detected_names)

        # --- Mostrar en ventana ---
        cv2.imshow("YOLO detections (filtrado)", annotated)
        cv2.waitKey(1)


    def save_detection(self, annotated_img, detected_names):
        """Guardar la imagen y la posición del robot en archivos."""
        # Timestamp para nombre de archivo y log
        now = datetime.now()
        ts = now.strftime('%Y%m%d_%H%M%S_%f')

        # Nombre de imagen
        labels_str = "_".join(sorted(set(detected_names)))
        img_filename = f"detection_{labels_str}_{ts}.png"
        img_path = os.path.join(self.img_dir, img_filename)

        # Guardar imagen
        cv2.imwrite(img_path, annotated_img)
        self.get_logger().info(f"Imagen guardada: {img_path}")

        # Preparar línea de log
        if self.current_pose is not None:
            p = self.current_pose.position
            q = self.current_pose.orientation
            pose_str = (
                f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
                f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
            )
        else:
            pose_str = "pose=UNKNOWN"

        log_line = (
            f"{now.isoformat()} | objects={labels_str} | image={img_filename} | {pose_str}\n"
        )

        # Escribir en archivo de texto
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_line)
        except Exception as e:
            self.get_logger().error(f"No pude escribir en el log: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

