import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
# Para importar ubicacion
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import numpy
import time

NAME = "Juan Mancera Lopez"

Ruta = [[-1.5,-8.5],	[-4.0,-8.5],	[-4.0,-9.5], 	    [-1.5,-9.5],	[-1.5,-10.5],	[-4.0,-10.5],
        [-2.0,-3.7],	[-4.0,-3.7],    [-4.0,-2.6], 	    [-2.0,-2.6],
        [-2.6,0.5],	    [-2.6,3.5],	    [-3.0,3.5], 	    [-3.0,0.5],	    [-3.6,0.5],	[-3.6,3.5],
        [0.4,2.5],	    [0.4,3.7],	    [-0.4,3.7], 	    [-0.4,2.5],
        [3.0,-2.5],	    [3.0,-7.0],	   	[3.8,-7.0], 	    [3.8,-2.5]]
Punt_Control = [[-1.5,-8.5], [-2,-3.7], [-2.6,0.5], [0.4,2.5], [3.0,-2.5]]

class RutaBarridoNode(Node):

    def seguimiento_metas(self):
        if self.num_destino >= len(Ruta):
            if not self.meta_enviada: # Solo imprimirlo una vez al final
                print("--- Ruta completada ---")
                self.num_destino = 0
                #self.meta_enviada = True # Bloqueamos para que no entrar en bucle infinito
            return

        """
        if self.seguir_buscando == False:
            print("Aviso de parada recibido.")
            raise SystemExit
            return
        """
        
        if not self.meta_enviada:
            msg = PoseStamped()
            
            # Indicar el sistema de coordenadas en el que está el punto
            msg.header.frame_id = "map" 
            msg.header.stamp = self.get_clock().now().to_msg()
            
            # Coordenadas destino (X, Y), a paritr del objetivo actual del array de rutas
            x_destino = Ruta[self.num_destino][0]
            y_destino = Ruta[self.num_destino][1]
            
            # Posicion a enviar al path planner
            msg.pose.position.x = x_destino
            msg.pose.position.y = y_destino
            msg.pose.orientation.w = 1.0

            print(f"Enviando al robot al punto {self.num_destino}: (x={x_destino}, y={y_destino})")
            
            self.goal_pub.publish(msg)
            self.meta_enviada = True

    def obtener_punto_cercano(self):
        print("Buscando el punto más cercano")
        # Posicion actual
        rob_x, rob_y, rob_theta = self.get_robot_pose()
        if rob_x is None:
            print("Error: No se detectó la posicion")
            return

        dist_min = 100
        pnt_cerc = [0,0]
        pos_rob = (rob_x, rob_y)

        # Busca en la lista Punt_Control cual es el "área" en la que se encuentra
        for ref in Punt_Control:
            distancia = math.dist(pos_rob, ref)
            print(f"Referencia: {ref}   - DIstancia: {distancia}")
            if  distancia < dist_min:
                dist_min = distancia
                pnt_cerc = ref
        print(f"El área más cercana es {Punt_Control.index(pnt_cerc)}, con dist {dist_min} y las coords {pnt_cerc}")

        print("Iniciando seguimiento de ruta...") # 0-4
        if (Punt_Control.index(pnt_cerc) == 0):
            self.num_destino = 5
        elif (Punt_Control.index(pnt_cerc) == 1):
            self.num_destino = 6
        elif (Punt_Control.index(pnt_cerc) == 2):
            self.num_destino = 10
        elif (Punt_Control.index(pnt_cerc) == 3):
            self.num_destino = 16
        elif (Punt_Control.index(pnt_cerc) == 4):
            self.num_destino = 20
        print(f"El punto a empezar debe ser: {self.num_destino}") 
        self.timer = self.create_timer(1.0, self.seguimiento_metas)

    def callback_llegada(self, msg):
        # Cuando path follower avisa que llegó, enviar al siguiente punto de la ruta
        if msg.data == True:
            x_llegada = Ruta[self.num_destino][0]
            y_llegada = Ruta[self.num_destino][1]
            print(f"-- El robot confirmó llegada a: ({x_llegada}, {y_llegada})")

            self.num_destino += 1
            
            self.meta_enviada = False

    def callback_stop(self, msg):
        if msg.data == True:
            print("Objetos encontrados. Deteniendo robot...")
            self.seguir_buscando = False
            raise SystemExit

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
            # Si el sistema de navegación aún no ha arrancado o no hay mapa
            self.get_logger().info(f'No se pudo obtener la posición: {ex}')
            return None, None, None

    def __init__(self):
        print("INITIALIZING RUTA BARRIDO NODE - ", NAME)
        super().__init__('ruta_barrido_node')
        
        # Para publicar en '/goal_pose' y asì decirle al robot a dónde ir (lo recibe el path_follower)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)
        
        # Escucha '/navigation/goal_reached' para saber si ya llegó (Se suscribe al tema, y path follower le avisa cuando llegò)
        self.status_sub = self.create_subscription(Bool, '/navigation/goal_reached', self.callback_llegada, 1)
        
        self.meta_enviada = False

        # Contador para la ruta establecida
        self.num_destino = 0

        self.sub_mision = self.create_subscription(Bool, '/objetos_encontrados', self.callback_stop, 1)
        # Bandera para controlar el bucle principal
        self.seguir_buscando = True

        # Para la posicion
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sistema_listo = False

        # Timer para enviar la meta
        #self.timer = self.create_timer(1.0, self.seguimiento_metas)
        #self.timer = self.create_timer(1.0, self.obtener_punto_cercano)
        self.timer = None

    def wait_for_pose(self):
        print("Esperando a que la posición del robot esté disponible (TF)...")
        while rclpy.ok():
            try:
                # Time() pide la transformación más reciente
                t = self.tf_buffer.lookup_transform(
                    "map", 
                    "base_link", 
                    rclpy.time.Time(), # Tiempo 0 => La más reciente
                    timeout=rclpy.duration.Duration(seconds=0.1) # Espera un poco si no está lista
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
    ruta_barrido = RutaBarridoNode()
    # Este para que no inicie hasta que tenga acceso a la posicion
    # (Para evitar que haya problema si lo detecta al aparecer)
    ruta_barrido.wait_for_pose()
    # Cuando tiene la posicion disponible llama al punto cercano, y este luego iniciará el bucle
    # De seguimiento
    ruta_barrido.obtener_punto_cercano()
    try:
        rclpy.spin(ruta_barrido)
    except KeyboardInterrupt:
        pass
    finally:
        ruta_barrido.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()