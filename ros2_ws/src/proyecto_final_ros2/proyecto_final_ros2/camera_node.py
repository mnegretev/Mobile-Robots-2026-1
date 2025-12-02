import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()
        self.model = YOLO("yolo11n.pt")  # mismo modelo que usaste en las pruebas
        self.model.overrides['verbose'] = False #silencia los logs internos
       #Subscripcion a la camara del robot 
        self.subscription = self.create_subscription(Image,'/camera/image_raw',self.image_callback,10)
        #Publicacion de las banderas
        self.flags_pub = self.create_publisher(Bool, '/all_objects_found', 10)
        
        # lista con los nombres EXACTOS de tus 3 clases
        self.target_classes = ["bottle","book","cup","vase"]
        self.min_area_cereal= 2442
        self.min_area_bottle= 578
        #Banderas
        self.found_cereal=False
        self.found_bottle=False
        self.found_trash=False
        
        
        
    def image_callback(self, msg):
        objetos_detectados=False
        
        # 1) ROS Image -> OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 2) YOLO sobre la imagen
        results = self.model(cv_image)
        r = results[0]

     
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            #coordenadas del boulding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            
            #self.get_logger().info(f"Detectado: {cls_name} (conf={conf:.2f}), w={width}, h={height}, area={area}")
            if cls_name =="book" and conf > 0.4 and area > self.min_area_cereal:
                self.get_logger().info(f"Objeto interes: {cls_name} (conf={conf:.2f})")
                self.found_cereal=True
                objetos_detectados=True
            if cls_name == "bottle" and conf > 0.3 and area > self.min_area_bottle:
                self.get_logger().info(f"Objeto interes: {cls_name} (conf={conf:.2f})")
                self.found_bottle=True
                objetos_detectados=True
            if cls_name == "cup" and conf > 0.5:
                self.get_logger().info(f"Objeto interes: {cls_name} (conf={conf:.2f})")
                self.found_trash=True
                objetos_detectados=True
        # LOGUEAR SOLO cuando se detecten objetos
        if objetos_detectados:
            status = f"Cereal: {self.found_cereal}, Bottle: {self.found_bottle}, Trash: {self.found_trash}"
            self.get_logger().info(f"OBJETOS: {status}")
            
        all_found = self.found_cereal and self.found_bottle and self.found_trash
        msg_flags = Bool()
        msg_flags.data = all_found
        self.flags_pub.publish(msg_flags)
                #printed = True
                # Aquí luego vas a:
                # - Dibujar cuadro / texto en la imagen
                # - Guardar imagen
                # - Leer posición del robot y guardarla en .txt  
     

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
