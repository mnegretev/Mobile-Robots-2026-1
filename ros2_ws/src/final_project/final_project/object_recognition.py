import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import os
# Para evitar el error (SPAM)
os.environ["USE_NNPACK"] = "0"
from datetime import datetime

import torch
# Desactiva el intento de usar NNPACK para evitar el spam en consola
torch.backends.nnpack.enabled = False 
from ultralytics import YOLO
# Para importar ubicacion
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import numpy
import time


FULL_NAME = "Juan Mancera Lopez"

class ObjectDetectorNode(Node):
    def callback_img(self, msg):
        if not self.sistema_listo:
            return

        # Convierte imagen de ROS a OpenCV
        img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Carga la imagen en el modelo y evita logs
        results = self.model(img_bgr, verbose=False)
        
        # Revisión sobre los resultados (puede haber múltiples detecciones)
        for r in results:
            # r.boxes contiene la información de cada caja delimitadora encontrada
            for box in r.boxes:
                # Obtiene el ID de la clase y su confianza (0.0 a 1.0)
                cls_id = int(box.cls[0])
                confianza = float(box.conf[0])
                
                # Cononer el nombre a partir del ID y del diccionario del modelo
                nombre_objeto = self.model.names[cls_id]

                #print(f"\n\nObjeto encontrado: {nombre_objeto}")
                
                # Si el objeto está en la lista de búsqueda /   pasa el mínimo de seguridad
                if (nombre_objeto in self.target_object) and (confianza > self.confidence_threshold):
                    
                    indice_objeto = self.target_object.index(nombre_objeto)
                    #Solo guardar si el objeto no había sido encontrado
                    if (self.cumplidos[indice_objeto] == 1):
                        continue

                    # Marca el objeto como encontrado
                    self.cumplidos[self.target_object.index(nombre_objeto)] = 1
                    
                    #Obtiene la posicion actual
                    rob_x, rob_y, rob_theta = self.get_robot_pose()
                    pos_str = "Desconocida"
                    if rob_x is not None:
                        pos_str = f"X:{rob_x:.2f}, Y:{rob_y:.2f}"
                    # Impresion de resultado
                    print(f"Objeto encontrado: {nombre_objeto} con confianza {confianza:.2f} en {pos_str}")
                    
                    # Guardado de imagen 
                    recognition_image = results[0].plot()
                    output_filename = f'detection_{nombre_objeto}_{indice_objeto}.png'
                    output_path = os.path.join(self.output_folder, output_filename)
                    cv2.imwrite(output_path, recognition_image, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                    print(f"Imagen guardada en -{output_path}-")
                    
                    # Guardado de ubicación en txt
                    with open(self.output_path_txt, "a") as fichero:
                        salida = f"\nObjeto: {nombre_objeto} en la ubicación: (x={rob_x:.3f}, y={rob_y:.3f})"
                        fichero.write(salida)

                    # Cuando encuentra 2 avisa 
                    if  sum(self.cumplidos) >= 2:
                        print("\n" + "="*40)
                        print(f"Meta alcanzada - Se han detectado {sum(self.cumplidos)} objetos.")
                        print("Deteniendo el nodo y cerrando el programa...")
                        print("="*40 + "\n")
                        
                        msg = Bool()
                        msg.data = True
                        self.pub_mision.publish(msg)
                        print("Aviso de parada enviado a la ruta de barrido.")

                        time.sleep(0.5)

                        # Detiene el código
                        raise SystemExit

        # Muestra de resultados de yolo (Se muestra la imagen de cámara, y si se detectaron objetos
        # muestra sus cajas con etiquetas)
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLO V11 Detection Live", annotated_frame)
        cv2.waitKey(1)
    
    def get_robot_pose(self):
        try:
            # Busca posicion de "baselink" respecto al map
            # rclpy.time.Time() obtener ultima posicion conocida
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            
            x = t.transform.translation.x
            y = t.transform.translation.y
            
            # Obtener ángulo
            theta = math.atan2(t.transform.rotation.z, t.transform.rotation.w)*2
            
            return x, y, theta
            
        except TransformException as ex:
            # Si no encontró la ubi
            self.get_logger().info(f'No se pudo obtener la posición: {ex}')
            return None, None, None

    def __init__(self):
        print("INITIALIZING OBJECT DETECTOR NODE - ", FULL_NAME)
        #Para usar tiempo del simulador
        #super().__init__("yolo_detector_node")
        super().__init__("objdet_node", 
                         parameter_overrides=[
                             rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
                         ])
        # Puente para convertir las imagenes
        self.br = CvBridge()
        # Suscripción a la cámara
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.callback_img, 1)
        
        # Cargar el modelo de YOLO.
        print("Cargando modelo YOLO...")
        self.model = YOLO("yolo11n.pt") 
        print("Modelo cargado correctamente.")
        
        # Para publicar cuando se encuentren los objetos y detener
        self.pub_mision = self.create_publisher(Bool, '/objetos_encontrados', 1)
        
        # Parámetros de configuración (qué buscar y con qué seguridad)
        # Nombres desde el dataset COCO
        self.target_object = ["cup", "book", "bottle"]  # cup = Cesto / book = caja de cereal / bottle = lata de refresco (no siempre lo detecta)
        self.cumplidos = [0,0,0]
        self.confidence_threshold = 0.3 # Mínimo 20% de seguridad

        # Parámetros para guardar la información
        self.output_folder = 'salida_ubicaciones'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        #Escritura de la hora y día en que se va a hacer la presente prueba (Para que no haya problemas con el append)
        ahora = datetime.now()
        texto_hora = ahora.strftime("%Y-%m-%d %H:%M:%S")
        output_filename = f'ubi_detection.txt'
        self.output_path_txt = os.path.join(self.output_folder, output_filename)

        with open(self.output_path_txt, "a") as fichero:
            salida = f"\n\n\nBusqueda {texto_hora}\n"
            fichero.write(salida)

        # Para la posicion
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sistema_listo = False

        print(f"Nodo listo. Buscando objetos: '{self.target_object}' con confianza > {self.confidence_threshold}")
       
    def wait_for_pose(self):
        print("Esperando a que la posición del robot esté disponible (TF)...")
        while rclpy.ok():
            try:
                # Time() pide la transformación más reciente
                t = self.tf_buffer.lookup_transform(
                    "map", 
                    "base_link", 
                    rclpy.time.Time(), # Tiempo 0 => La más reciente
                    timeout=rclpy.duration.Duration(seconds=0.1) # Espera un poco por si no está lista
                )
                
                print("¡Posición recibida! Activando detección...")
                self.sistema_listo = True
                break 
                
            except TransformException as e:
                # print(f"Aun esperando: {e}")
                pass
            
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    objdet = ObjectDetectorNode()
    # Este para que no inicie hasta que tenga acceso a la posicion
    # (Para evitar que haya problema si lo detecta al aparecer)
    objdet.wait_for_pose()

    try:
        rclpy.spin(objdet)
    except KeyboardInterrupt:
        pass
    finally:
        objdet.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()