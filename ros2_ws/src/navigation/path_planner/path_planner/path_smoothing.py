#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# PATH SMOOTHING BY GRADIENT DESCEND
#
# Instructions:
# Write the code necessary to smooth a path using the gradient descend algorithm.
# MODIFY ONLY THE SECTIONS MARKED WITH THE 'TODO' COMMENT
#
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point
from navig_msgs.srv import ProcessPath
import numpy
import matplotlib.pyplot as plt
import os

NAME = "Daniel Ixbalanque Popoca Zuñiga"

class PathSmoothingNode(Node):
    
    def __init__(self):
        print("INITIALIZING PATH SMOOTHING NODE - ", NAME)
        super().__init__("path_smoothing_node")
        self.declare_parameter('w1', 0.9)
        self.declare_parameter('w2', 0.1)
        self.declare_parameter('steps', 10000)
        self.srv_smooth_path = self.create_service(ProcessPath, '/path_planning/smooth_path', self.callback_smooth_path)
        self.pub_smooth_path = self.create_publisher(Path, '/path_planning/smoothed_path', 10)
        self.msg_smooth_path = Path()
        self.path_counter = 0  # Contador para numerar las trayectorias
        
    def plot_and_save_paths(self, Q, P, path_number, w1, w2, max_steps):
        """
        Genera y guarda una gráfica comparando la ruta original vs la suavizada
        """
        plt.figure(figsize=(10, 8))
        
        # Plot ruta original - solo línea y puntos sin etiquetas
        plt.plot(Q[:, 0], Q[:, 1], 'ro-', linewidth=2, markersize=6, label='Ruta Original (A*)')
        
        # Plot ruta suavizada - solo línea y puntos sin etiquetas
        plt.plot(P[:, 0], P[:, 1], 'bo-', linewidth=2, markersize=6, label='Ruta Suavizada')
        
        # Configuración de la gráfica
        plt.grid(True, alpha=0.3)
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')

        plt.title(f'Ruta Original vs Suavizada \nw1={w1}, w2={w2}, N={len(Q)} puntos, Max Steps={max_steps}')  
        plt.legend()
        plt.axis('equal')
        
        # Crear directorio si no existe
        os.makedirs('path_plots', exist_ok=True)
        
        # Guardar la imagen
        filename = f'smoot_vs_A_{path_number}.png'
        filepath = os.path.join('path_plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfica guardada como: {filepath}")
        
    def smooth_path(self, Q, w1, w2, max_steps):
        P = numpy.copy(Q)
        tol = 0.00001
        nabla = numpy.full(Q.shape, float("inf"))
        epsilon = 0.1
        
        n = len(P)  # Number of points in the path
        steps = 0
        
        # Continue until convergence or max steps reached
        while steps < max_steps:
            # Calculate gradient for all points except first and last
            nabla_J = numpy.zeros_like(P)
            
            # First and last points have zero gradient (fixed endpoints)
            nabla_J[0] = 0
            nabla_J[-1] = 0
            
            # Calculate gradient for intermediate points
            for i in range(1, n - 1):
                # w1 * (2pi - pi-1 - pi+1) + w2 * (pi - qi)
                smooth_term = w1 * (2 * P[i] - P[i-1] - P[i+1])
                data_term = w2 * (P[i] - Q[i])
                nabla_J[i] = smooth_term + data_term
            
            # Check convergence (max norm of gradient)
            max_gradient_norm = numpy.max(numpy.linalg.norm(nabla_J, axis=1))
            if max_gradient_norm <= tol:
                break
            
            # Update path: P <- P - epsilon * nabla_J
            P = P - epsilon * nabla_J
            steps += 1
        
        print(f"Path smoothing completed in {steps} steps")
        return P
        
    def callback_smooth_path(self, request, response):
        w1  = self.get_parameter('w1').get_parameter_value().double_value
        w2  = self.get_parameter('w2').get_parameter_value().double_value
        steps  = self.get_parameter('steps').get_parameter_value().integer_value
        print("Smoothing path with params:", [w1, w2, steps])
        start_time = self.get_clock().now()
        Q = numpy.asarray([[p.pose.position.x, p.pose.position.y] for p in request.path.poses])
        P = self.smooth_path(Q, w1, w2, steps)
        end_time = self.get_clock().now()
        delta_ms = (end_time.nanoseconds - start_time.nanoseconds)/1e6
        print("Path smoothed after " + str(delta_ms) + " ms")
        
        # Incrementar contador y generar gráfica
        self.path_counter += 1
        self.plot_and_save_paths(Q, P, self.path_counter, w1, w2, steps)
        
    
        
        self.msg_smooth_path.header.frame_id = request.path.header.frame_id
        self.msg_smooth_path.header.stamp = self.get_clock().now().to_msg()
        self.msg_smooth_path.poses = []
        for i in range(len(request.path.poses)):
            p = PoseStamped()
            p.pose.position.x = P[i,0]
            p.pose.position.y = P[i,1]
            self.msg_smooth_path.poses.append(p)
        self.pub_smooth_path.publish(self.msg_smooth_path)
        response.processed_path = self.msg_smooth_path
        return response

def main(args=None):
    rclpy.init(args=args)
    path_smoothing_node = PathSmoothingNode()
    rclpy.spin(path_smoothing_node)
    path_smoothing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()