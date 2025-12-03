import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import os
import math
from datetime import datetime
from ultralytics import YOLO
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        
        print("Cargando YOLO...")
        self.model = YOLO("yolo11n.pt") 
        self.target_object = ["cup", "book"]
        self.found_objects = [False, False] 
        self.confidence_threshold = 0.2
        
        
        self.br = CvBridge()
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.callback_img, 1)
        self.pub_stop = self.create_publisher(Bool, '/stop_exploration', 1)
        
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        
        self.output_folder = 'salida_evidencia'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.txt_path = os.path.join(self.output_folder, 'hallazgos.txt')
        
        print("Buscando:", self.target_object)

    def get_pose_string(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            return f"x={x:.2f}, y={y:.2f}"
        except TransformException:
            return "Posición desconocida"

    def callback_img(self, msg):
        
        if all(self.found_objects):
            return

        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(img_bgr, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]

                if name in self.target_object and conf > self.confidence_threshold:
                    idx = self.target_object.index(name)
                    
                    if not self.found_objects[idx]:
                        
                        self.found_objects[idx] = True
                        pose_str = self.get_pose_string()
                        
                        print(f"Objeto encontrado{name} en {pose_str}")
                        
                        timestamp = datetime.now().strftime("%H-%M-%S")
                        img_name = f"{name}_{timestamp}.png"
                        cv2.imwrite(os.path.join(self.output_folder, img_name), results[0].plot())
                        
                        
                        with open(self.txt_path, "a") as f:
                            f.write(f"Objeto: {name} | {timestamp} | Pos: {pose_str}\n")
                        
                        
                        if all(self.found_objects):
                            print("Objetos encontrado, detenientdo robot")
                            self.pub_stop.publish(Bool(data=True))
                            raise SystemExit

        
        cv2.imshow("Camara Robot", results[0].plot())
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
