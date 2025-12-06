#!/usr/bin/env python3
"""
EXPLORADOR INTEGRADO CON DETECCIÓN DE OBJETOS
Se detiene cuando se encuentran los 3 objetos
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import numpy as np
import time

class IntegratedExplorer(Node):
    
    def __init__(self):
        super().__init__('integrated_explorer')
        
        self.get_logger().info('=== EXPLORADOR INTEGRADO INICIADO ===')
        
        # Estado de búsqueda
        self.objects_found = {
            'coca_cola': False,
            'cereal_box': False,
            'trash_bin': False
        }
        self.exploration_complete = False
        
        # Velocidades
        self.max_linear_speed = 0.5
        self.max_angular_speed = 0.7
        
        # Distancias críticas
        self.critical_distance = 0.4
        self.safe_distance = 0.5
        self.comfort_distance = 0.65
        
        # Estado del robot
        self.current_pose = None
        self.laser_data = None
        self.front_clear = True
        self.min_front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        
        # TF para localización
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_ready = False
        
        # Estrategia de exploración
        self.exploration_strategy = "WALL_FOLLOW"
        self.wall_follow_side = "RIGHT"
        self.spiral_angle = 0
        self.turn_timer = 0
        self.spiral_radius = 0.5
        self.spiral_growth_rate = 0.01
        self.last_strategy_change = time.time()
        self.strategy_change_interval = 15.0
        
        # ESCAPE DE ESQUINAS - NUEVO
        self.stuck_counter = 0
        self.max_stuck_count = 15  # 1.5 segundos atascado
        self.escape_mode = False
        self.escape_timer = 0
        self.escape_duration = 20  # 2 segundos de escape
        self.last_position = None
        self.position_timer = 0
        self.position_check_interval = 30  # Verificar movimiento cada 3 segundos
        
        # Estadísticas
        self.exploration_start = time.time()
        
        # Publishers y Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        
        self.detection_sub = self.create_subscription(
            String, '/object_detected', self.detection_callback, 10
        )
        
        # Timer de control
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.stats_timer = self.create_timer(10.0, self.report_stats)
        
        self.get_logger().info('Esperando localización...')
        self.get_logger().info('Buscando: Coca-Cola, Caja de cereal, Bote de basura')
    
    def detection_callback(self, msg):
        """Recibe notificaciones de objetos detectados"""
        data = msg.data.lower()
        
        if 'coca_cola' in data:
            self.objects_found['coca_cola'] = True
        elif 'cereal_box' in data:
            self.objects_found['cereal_box'] = True
        elif 'trash_bin' in data:
            self.objects_found['trash_bin'] = True
        elif 'all_objects_found' in data:
            self.exploration_complete = True
            self.get_logger().info('¡MISIÓN COMPLETADA!')
    
    def get_robot_pose(self):
        """Obtiene pose del robot usando TF"""
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
            q = t.transform.rotation
            robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            self.current_pose = {
                'x': robot_x,
                'y': robot_y,
                'yaw': robot_yaw
            }
            
            self.tf_ready = True
            return True
            
        except TransformException:
            if self.tf_ready:
                self.get_logger().warn('TF perdido temporalmente', throttle_duration_sec=5.0)
            return False
    
    def check_if_stuck(self):
        """Verifica si el robot está atascado en una esquina"""
        # Verificar por distancia láser
        corner_condition = (
            self.min_front_distance < 0.5 and 
            self.left_distance < 0.5 and 
            self.right_distance < 0.5
        )
        
        # Verificar por falta de movimiento (si TF está disponible)
        if self.current_pose and not self.escape_mode:
            self.position_timer += 1
            
            if self.position_timer >= self.position_check_interval:
                if self.last_position is None:
                    self.last_position = (self.current_pose['x'], self.current_pose['y'])
                    self.position_timer = 0
                else:
                    dx = self.current_pose['x'] - self.last_position[0]
                    dy = self.current_pose['y'] - self.last_position[1]
                    distance_moved = math.sqrt(dx*dx + dy*dy)
                    
                    if distance_moved < 0.1:  # Menos de 10 cm movidos
                        self.stuck_counter += 1
                        if self.stuck_counter >= 5:  # 1.5 segundos sin moverse
                            self.get_logger().warn(f'Robot posiblemente atascado (movimiento: {distance_moved:.2f}m)')
                    else:
                        self.stuck_counter = max(0, self.stuck_counter - 1)
                    
                    self.last_position = (self.current_pose['x'], self.current_pose['y'])
                    self.position_timer = 0
        
        # Condición de esquina: todas las distancias son cortas
        if corner_condition:
            self.stuck_counter += 1
            if self.stuck_counter >= self.max_stuck_count:
                self.enter_escape_mode()
                return True
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        return False
    
    def enter_escape_mode(self):
        """Activa modo de escape de esquina"""
        self.escape_mode = True
        self.escape_timer = self.escape_duration
        self.stuck_counter = 0
        self.get_logger().warn('¡ATASCADO EN ESQUINA! Activando modo de escape')
    
    def exit_escape_mode(self):
        """Desactiva modo de escape"""
        self.escape_mode = False
        self.escape_timer = 0
        self.get_logger().info('Modo de escape completado')
    
    def escape_behavior(self):
        """Comportamiento para escapar de esquinas"""
        twist = Twist()
        
        if self.escape_timer > 0:
            # Secuencia de escape: retroceder, girar, avanzar
            if self.escape_timer > 15:  # Fase 1: Retroceder
                twist.linear.x = -0.3
                twist.angular.z = 0.0
            elif self.escape_timer > 10:  # Fase 2: Girar
                twist.linear.x = 0.0
                # Girar hacia el lado con más espacio
                if self.left_distance > self.right_distance:
                    twist.angular.z = self.max_angular_speed * 0.8  # Girar izquierda
                else:
                    twist.angular.z = -self.max_angular_speed * 0.8  # Girar derecha
            elif self.escape_timer > 5:  # Fase 3: Avanzar con giro
                twist.linear.x = 0.4
                twist.angular.z = 0.3 if self.left_distance > self.right_distance else -0.3
            else:  # Fase 4: Solo avanzar
                twist.linear.x = 0.3
                twist.angular.z = 0.0
            
            self.escape_timer -= 1
        else:
            self.exit_escape_mode()
            # Volver a comportamiento normal
            twist = self.wall_follow_behavior()
        
        return twist
    
    def laser_callback(self, msg):
        """Procesa datos del laser"""
        self.laser_data = msg
        
        ranges = np.array(msg.ranges)
        ranges = np.where(
            (ranges > msg.range_max) | (ranges < msg.range_min) | np.isnan(ranges),
            msg.range_max,
            ranges
        )
        
        num_readings = len(ranges)
        
        # Sector frontal (más amplio para detectar esquinas)
        front_start = int(num_readings * 0.3)
        front_end = int(num_readings * 0.7)
        front_sector = ranges[front_start:front_end]
        self.min_front_distance = np.min(front_sector)
        
        # Sector izquierdo
        left_start = int(num_readings * 0.1)
        left_end = int(num_readings * 0.3)
        left_sector = ranges[left_start:left_end]
        self.left_distance = np.min(left_sector)
        
        # Sector derecho
        right_start = int(num_readings * 0.7)
        right_end = int(num_readings * 0.9)
        right_sector = ranges[right_start:right_end]
        self.right_distance = np.min(right_sector)
        
        # Sector trasero para detectar callejones sin salida
        rear_start = int(num_readings * 0.85)
        rear_end = num_readings
        rear_sector = np.concatenate([ranges[rear_start:], ranges[:int(num_readings * 0.15)]])
        self.rear_distance = np.min(rear_sector)
        
        # Histeresis para evitar oscilaciones
        if self.front_clear:
            self.front_clear = self.min_front_distance > self.safe_distance
        else:
            self.front_clear = self.min_front_distance > self.comfort_distance
        
        # Verificar si estamos en un espacio abierto para cambiar a espiral
        if not self.escape_mode:
            if self.min_front_distance > 2.0 and self.left_distance > 1.5 and self.right_distance > 1.5:
                self.exploration_strategy = "SPIRAL"
                self.get_logger().info('Espacio abierto detectado, cambiando a espiral')
            elif self.min_front_distance < 1.0 or self.left_distance < 0.8 or self.right_distance < 0.8:
                self.exploration_strategy = "WALL_FOLLOW"
    
    def wall_follow_behavior(self):
        """Comportamiento de seguir paredes"""
        twist = Twist()
        target_wall_distance = 0.45
        
        # Si estamos muy cerca de una pared frontal, girar más agresivamente
        if self.min_front_distance < 0.4:
            twist.linear.x = 0.0
            twist.angular.z = -self.max_angular_speed if self.wall_follow_side == "RIGHT" else self.max_angular_speed
            self.turn_timer = 10
        elif self.turn_timer > 0:
            twist.angular.z = -self.max_angular_speed if self.wall_follow_side == "RIGHT" else self.max_angular_speed
            twist.linear.x = 0.1
            self.turn_timer -= 1
        else:
            if self.wall_follow_side == "RIGHT":
                wall_distance = self.right_distance
                error = wall_distance - target_wall_distance
            else:
                wall_distance = self.left_distance
                error = wall_distance - target_wall_distance
            
            # Ajuste más agresivo cuando estamos cerca de paredes
            if wall_distance > 1.5:
                twist.linear.x = self.max_linear_speed * 0.8
                twist.angular.z = -0.5 if self.wall_follow_side == "RIGHT" else 0.5
            elif wall_distance < 0.3:
                twist.linear.x = 0.1
                twist.angular.z = 0.3 if self.wall_follow_side == "RIGHT" else -0.3
            else:
                twist.linear.x = self.max_linear_speed * 0.8
                twist.angular.z = -0.8 * error  # Ganancia aumentada
        
        return twist
    
    def spiral_behavior(self):
        """Comportamiento en espiral para espacios abiertos"""
        twist = Twist()
        
        # Si detectamos obstáculo cerca, volver a seguir paredes
        if self.min_front_distance < 0.8 or self.left_distance < 0.6 or self.right_distance < 0.6:
            self.exploration_strategy = "WALL_FOLLOW"
            self.get_logger().info('Obstáculo detectado, volviendo a seguir paredes')
            return self.wall_follow_behavior()
        
        # Espiral más compacta
        self.spiral_radius += self.spiral_growth_rate
        
        # Velocidad adaptativa
        linear_speed = min(self.max_linear_speed * 0.6, 0.15 + self.spiral_radius * 0.08)
        angular_speed = self.max_angular_speed * 0.4 / (self.spiral_radius + 0.3)
        
        twist.linear.x = linear_speed
        twist.angular.z = -angular_speed  # Girar derecha
        
        return twist
    
    def compute_velocity_command(self):
        """Calcula comando de velocidad según estrategia"""
        # PRIMERO verificar si estamos atascados
        if self.check_if_stuck():
            self.get_logger().warn('Ejecutando escape de esquina...')
            return self.escape_behavior()
        
        # SI NO estamos atascados, comportamiento normal
        twist = Twist()
        
        if self.exploration_strategy == "WALL_FOLLOW":
            twist = self.wall_follow_behavior()
        elif self.exploration_strategy == "SPIRAL":
            twist = self.spiral_behavior()
        
        # Ajuste de seguridad (reducido durante escape)
        if not self.escape_mode and self.min_front_distance < self.comfort_distance:
            safety_factor = self.min_front_distance / self.comfort_distance
            twist.linear.x *= max(0.2, safety_factor)
        
        # Limitar velocidades
        twist.linear.x = np.clip(twist.linear.x, -self.max_linear_speed * 0.5 if self.escape_mode else 0.0, self.max_linear_speed)
        twist.angular.z = np.clip(twist.angular.z,
                                  -self.max_angular_speed,
                                  self.max_angular_speed)
        
        return twist
    
    def control_loop(self):
        """Bucle de control principal"""
        # Si ya encontramos todo, detener
        if self.exploration_complete:
            self.finish_exploration()
            return
        
        if not self.get_robot_pose():
            return
        
        if self.laser_data is None:
            return
        
        # Cambiar estrategia periódicamente si no está atascado
        if not self.escape_mode:
            current_time = time.time()
            if current_time - self.last_strategy_change > self.strategy_change_interval:
                if self.exploration_strategy == "WALL_FOLLOW":
                    self.wall_follow_side = "LEFT" if self.wall_follow_side == "RIGHT" else "RIGHT"
                    self.get_logger().info(f'Cambiando lado de seguimiento: {self.wall_follow_side}')
                    self.last_strategy_change = current_time
                elif self.exploration_strategy == "SPIRAL":
                    self.spiral_radius = 0.5
                    self.get_logger().info('Reiniciando patrón de espiral')
                    self.last_strategy_change = current_time
        
        # Calcular y publicar velocidad
        twist = self.compute_velocity_command()
        self.cmd_vel_pub.publish(twist)
        
        # Timeout de seguridad
        elapsed = time.time() - self.exploration_start
        if elapsed > 600:  # 10 minutos
            self.get_logger().warn('Tiempo límite alcanzado')
            self.finish_exploration()
    
    def report_stats(self):
        """Reporta estadísticas"""
        elapsed = time.time() - self.exploration_start
        found_count = sum(self.objects_found.values())
        
        objects_status = []
        for obj, found in self.objects_found.items():
            status = '✓' if found else '✗'
            objects_status.append(f'{obj}: {status}')
        
        mode = "ESCAPE" if self.escape_mode else self.exploration_strategy
        self.get_logger().info(
            f'[BÚSQUEDA] Tiempo: {elapsed:.0f}s | '
            f'Encontrados: {found_count}/3 | '
            f'Modo: {mode} | '
            f'Atascado: {self.stuck_counter}/{self.max_stuck_count} | '
            f'{" | ".join(objects_status)}'
        )
    
    def finish_exploration(self):
        """Finaliza exploración"""
        self.get_logger().info('=== EXPLORACIÓN FINALIZADA ===')
        
        elapsed = time.time() - self.exploration_start
        found_count = sum(self.objects_found.values())
        
        self.get_logger().info(f'Objetos encontrados: {found_count}/3')
        self.get_logger().info(f'Tiempo total: {elapsed:.1f}s')
        
        for obj, found in self.objects_found.items():
            status = 'ENCONTRADO ✓' if found else 'NO ENCONTRADO ✗'
            self.get_logger().info(f'  {obj}: {status}')
        
        # Detener robot
        self.cmd_vel_pub.publish(Twist())
        time.sleep(1)
        
        self.destroy_node()
        rclpy.shutdown()
    
    def normalize_angle(self, angle):
        """Normaliza ángulo a [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = IntegratedExplorer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[INFO] Exploración interrumpida')
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
        print('[INFO] Programa finalizado')


if __name__ == '__main__':
    main()