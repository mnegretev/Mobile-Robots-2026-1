import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from std_msgs.msg import Bool


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.br = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.current_pose = None

        self.get_logger().info('Iniciando YOLO node, suscrito a /camera/image_raw y /amcl_pose')

        self.model = YOLO('/home/axel/yolov8n.pt')
        self.get_logger().info('Modelo YOLOv8n cargado.')

        self.allowed_classes = ["cup", "bottle", "book"]

        self.base_dir = '/home/axel/detections'
        self.img_dir = os.path.join(self.base_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)

        self.log_path = os.path.join(self.base_dir, 'detections.txt')

        self.get_logger().info(f"Guardando imágenes en: {self.img_dir}")
        self.get_logger().info(f"Guardando log en: {self.log_path}")
    
        self.object_pub = self.create_publisher(Bool, '/object_detected', 10)
        self.object_detected = False

        # ---------- NUEVO: bandera por objeto ----------
        self.saved_flags = {
            "cup": False,
            "bottle": False,
            "cerealbox": False
        }


    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg.pose.pose


    def image_callback(self, msg: Image):
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, verbose=False)[0]

        filtered_boxes = []
        detected_names = []

        for box in results.boxes:
            cls_id = int(box.cls.cpu().item())
            name = results.names[cls_id]

            if name not in self.allowed_classes:
                continue

            if name == "book":
                name = "cerealbox"

            detected_names.append(name)
            filtered_boxes.append(box)

        annotated = frame.copy()

        if not filtered_boxes:
            cv2.imshow("YOLO detections", annotated)
            cv2.waitKey(1)
            return

        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
            cls_id = int(box.cls.cpu().item())
            label = results.names[cls_id]
            if label == "book":
                label = "cerealbox"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.get_logger().info(f"Detecté: {detected_names}")

        if not self.object_detected:
            self.object_detected = True
            msg = Bool()
            msg.data = True
            self.object_pub.publish(msg)

        # ---------- NUEVO: guardar solo la primera vez por objeto ----------
        for obj in detected_names:
            if not self.saved_flags[obj]:
                self.saved_flags[obj] = True
                self.save_detection(annotated, [obj])
                self.get_logger().info(f"Guardé la primera detección de: {obj}")

        cv2.imshow("YOLO detections", annotated)
        cv2.waitKey(1)


    def save_detection(self, annotated_img, detected_names):
        now = datetime.now()
        ts = now.strftime('%Y%m%d_%H%M%S_%f')

        label = detected_names[0]
        img_filename = f"detection_{label}_{ts}.png"
        img_path = os.path.join(self.img_dir, img_filename)

        cv2.imwrite(img_path, annotated_img)
        self.get_logger().info(f"Imagen guardada: {img_path}")

        if self.current_pose is not None:
            p = self.current_pose.position
            q = self.current_pose.orientation
            pose_str = f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
        else:
            pose_str = "pose=UNKNOWN"

        log_line = f"{now.isoformat()} | object={label} | image={img_filename} | {pose_str}\n"

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

