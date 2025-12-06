#!/usr/bin/env python3
"""
DETECTOR DE OBJETOS CON YOLO
Detecta Coca-Cola, caja de cereal y bote de basura
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class ObjectDetector(Node):
    
    def __init__(self):
        super().__init__('object_detector')
        
        self.get_logger().info('=== DETECTOR DE OBJETOS INICIADO ===')
        
        # Cargar modelo YOLO
        try:
            # Intenta cargar YOLOv8 (ajusta la ruta si es necesario)
            self.model = YOLO('yolov8n.pt').to('cpu')
            self.get_logger().info('Modelo YOLO cargado correctamente')
        except Exception as e:
            self.get_logger().error(f'Error cargando YOLO: {e}')
            self.model = None
        
        # Bridge para convertir mensajes ROS a OpenCV
        self.bridge = CvBridge()
        
        # Objetos a detectar (mapeo de clases YOLO a objetos del proyecto)
        # Ajusta estos nombres según las clases que detecte tu modelo
        self.target_objects = {
            'bottle': 'coca_cola',      # Lata de Coca-Cola
            'cup': 'coca_cola',          # Alternativa para lata
            'bowl': 'cereal_box',        # Caja de cereal
            'book': 'cereal_box',        # Alternativa para caja
            'trash_can': 'trash_bin',    # Bote de basura
            'bin': 'trash_bin',           # Alternativa para bote
            'vase': 'trash_bin'           # Alternativa para bote
        }
        
        # Objetos encontrados
        self.found_objects = {
            'coca_cola': False,
            'cereal_box': False,
            'trash_bin': False
        }
        
        # Posiciones donde se encontraron los objetos
        self.object_positions = {}
        
        # TF para obtener posición del robot
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Directorio para guardar imágenes
        self.save_dir = '/home/melissa/detected_objects'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Control de detecciones
        self.detection_cooldown = {}
        self.cooldown_time = 3.0  # Segundos entre detecciones del mismo objeto
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher para notificar objetos encontrados
        self.detection_pub = self.create_publisher(
            String,
            '/object_detected',
            10
        )
        
        # Timer para verificar si ya terminamos
        self.check_timer = self.create_timer(1.0, self.check_completion)
        
        self.get_logger().info(f'Guardando imágenes en: {self.save_dir}')
        self.get_logger().info('Esperando imágenes de la cámara...')
    
    def get_robot_pose(self):
        """Obtiene la posición actual del robot"""
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
            
            return {
                'x': t.transform.translation.x,
                'y': t.transform.translation.y,
                'z': t.transform.translation.z,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        except TransformException as e:
            self.get_logger().warn(
                f'No se pudo obtener posición: {e}',
                throttle_duration_sec=5.0
            )
            return None
    
    def image_callback(self, msg):
        """Procesa imágenes de la cámara y detecta objetos"""
        if self.model is None:
            return
        
        # Verificar si ya encontramos todos los objetos
        if all(self.found_objects.values()):
            return
        
        try:
            # Convertir mensaje ROS a imagen OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Realizar detección con YOLO
            results = self.model(cv_image, conf=0.4, verbose=False)
            
            # Procesar detecciones
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Obtener información de la detección
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls_id]
                    
                    # Verificar si es un objeto que buscamos
                    if class_name in self.target_objects:
                        obj_name = self.target_objects[class_name]
                        
                        # Si ya lo encontramos, skip
                        if self.found_objects[obj_name]:
                            continue
                        
                        # Control de cooldown
                        current_time = self.get_clock().now().nanoseconds / 1e9
                        if obj_name in self.detection_cooldown:
                            if current_time - self.detection_cooldown[obj_name] < self.cooldown_time:
                                continue
                        
                        # Obtener coordenadas del bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Dibujar detección en la imagen
                        annotated_image = cv_image.copy()
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Agregar etiqueta
                        label = f'{obj_name.upper()} ({conf:.2f})'
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        
                        # Fondo para el texto
                        cv2.rectangle(
                            annotated_image,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Texto
                        cv2.putText(
                            annotated_image,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2
                        )
                        
                        # Obtener posición del robot
                        robot_pose = self.get_robot_pose()
                        
                        if robot_pose:
                            # Guardar imagen
                            image_filename = f'{obj_name}_{robot_pose["timestamp"]}.jpg'
                            image_path = os.path.join(self.save_dir, image_filename)
                            cv2.imwrite(image_path, annotated_image)
                            
                            # Marcar como encontrado
                            self.found_objects[obj_name] = True
                            self.object_positions[obj_name] = robot_pose
                            
                            # Guardar información en archivo de texto
                            self.save_detection_info(obj_name, robot_pose, conf)
                            
                            # Publicar detección
                            detection_msg = String()
                            detection_msg.data = f'{obj_name} detected'
                            self.detection_pub.publish(detection_msg)
                            
                            # Log
                            self.get_logger().info(
                                f'✓ {obj_name.upper()} DETECTADO! '
                                f'Pos: ({robot_pose["x"]:.2f}, {robot_pose["y"]:.2f}) '
                                f'Conf: {conf:.2f}'
                            )
                            
                            # Actualizar cooldown
                            self.detection_cooldown[obj_name] = current_time
                            
                            # Verificar si terminamos
                            found_count = sum(self.found_objects.values())
                            self.get_logger().info(
                                f'Progreso: {found_count}/3 objetos encontrados'
                            )
        
        except Exception as e:
            self.get_logger().error(f'Error procesando imagen: {e}')
    
    def save_detection_info(self, obj_name, robot_pose, confidence):
        """Guarda información de la detección en archivo de texto"""
        info_file = os.path.join(self.save_dir, 'detections.txt')
        
        with open(info_file, 'a') as f:
            f.write(f'\n{"="*60}\n')
            f.write(f'Objeto: {obj_name.upper()}\n')
            f.write(f'Timestamp: {robot_pose["timestamp"]}\n')
            f.write(f'Posición del robot:\n')
            f.write(f'  X: {robot_pose["x"]:.4f} m\n')
            f.write(f'  Y: {robot_pose["y"]:.4f} m\n')
            f.write(f'  Z: {robot_pose["z"]:.4f} m\n')
            f.write(f'Confianza: {confidence:.4f}\n')
            f.write(f'{"="*60}\n')
    
    def check_completion(self):
        """Verifica si se encontraron todos los objetos"""
        if all(self.found_objects.values()):
            self.get_logger().info('¡TODOS LOS OBJETOS ENCONTRADOS!')
            self.get_logger().info('Generando reporte final...')
            self.generate_final_report()
            
            # Notificar al explorador que termine
            msg = String()
            msg.data = 'all_objects_found'
            self.detection_pub.publish(msg)
            
            # Detener este nodo
            self.destroy_timer(self.check_timer)
    
    def generate_final_report(self):
        """Genera reporte final de detecciones"""
        report_file = os.path.join(self.save_dir, 'REPORTE_FINAL.txt')
        
        with open(report_file, 'w') as f:
            f.write('='*70 + '\n')
            f.write('REPORTE FINAL DE DETECCIÓN DE OBJETOS\n')
            f.write('='*70 + '\n\n')
            
            for obj_name, found in self.found_objects.items():
                f.write(f'\n{obj_name.upper()}:\n')
                f.write('-' * 40 + '\n')
                
                if found:
                    pose = self.object_positions[obj_name]
                    f.write(f'  Estado: ✓ ENCONTRADO\n')
                    f.write(f'  Posición del robot:\n')
                    f.write(f'    X: {pose["x"]:.4f} m\n')
                    f.write(f'    Y: {pose["y"]:.4f} m\n')
                    f.write(f'    Z: {pose["z"]:.4f} m\n')
                    f.write(f'  Timestamp: {pose["timestamp"]}\n')
                else:
                    f.write(f'  Estado: ✗ NO ENCONTRADO\n')
            
            f.write('\n' + '='*70 + '\n')
            f.write(f'Total encontrados: {sum(self.found_objects.values())}/3\n')
            f.write('='*70 + '\n')
        
        self.get_logger().info(f'Reporte guardado en: {report_file}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ObjectDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[INFO] Detector interrumpido')
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
        print('[INFO] Detector finalizado')


if __name__ == '__main__':
    main()