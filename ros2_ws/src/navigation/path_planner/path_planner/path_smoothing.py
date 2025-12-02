import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
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
        self.declare_parameter('steps', 1000)

        # Publisher de la ruta suavizada
        self.pub_smooth_path = self.create_publisher(Path, '/path_planning/smoothed_path', 10)

        # Suscripción al topic de A*
        self.sub_path = self.create_subscription(
            Path,
            '/path_planning/path',
            self.callback_path_topic,
            10
        )

        self.msg_smooth_path = Path()
        self.path_counter = 0  # Contador para numerar las trayectorias

    def callback_path_topic(self, msg):
        # Llama al smoothing automáticamente al recibir la ruta de A*
        Q = numpy.asarray([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        w1  = self.get_parameter('w1').get_parameter_value().double_value
        w2  = self.get_parameter('w2').get_parameter_value().double_value
        max_steps  = self.get_parameter('steps').get_parameter_value().integer_value

        P = self.smooth_path(Q, w1, w2, max_steps)

        # Incrementar contador y generar gráfica
        self.path_counter += 1
        self.plot_and_save_paths(Q, P, self.path_counter, w1, w2, max_steps)

        # Publicar ruta suavizada
        self.msg_smooth_path.header = msg.header
        self.msg_smooth_path.poses = []
        for i in range(len(P)):
            p = PoseStamped()
            p.pose.position.x = P[i,0]
            p.pose.position.y = P[i,1]
            self.msg_smooth_path.poses.append(p)
        self.pub_smooth_path.publish(self.msg_smooth_path)
        print(f"Published smoothed path #{self.path_counter} with {len(P)} points")

    def plot_and_save_paths(self, Q, P, path_number, w1, w2, max_steps):
        plt.figure(figsize=(10, 8))
        plt.plot(Q[:, 0], Q[:, 1], 'ro-', linewidth=2, markersize=6, label='Ruta Original (A*)')
        plt.plot(P[:, 0], P[:, 1], 'bo-', linewidth=2, markersize=6, label='Ruta Suavizada')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Ruta Original vs Suavizada - w1={w1}, w2={w2}, N={len(Q)}, Max Steps={max_steps}')
        plt.legend()
        plt.axis('equal')
        os.makedirs('path_plots', exist_ok=True)
        filepath = os.path.join('path_plots', f'smoothed_path_{path_number}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfica guardada como: {filepath}")

    def smooth_path(self, Q, w1, w2, max_steps):
        P = numpy.copy(Q)
        tol = 0.1
        epsilon = 0.1
        n = len(P)
        steps = 0

        while steps < max_steps:
            nabla_J = numpy.zeros_like(P)
            nabla_J[0] = 0
            nabla_J[-1] = 0
            for i in range(1, n-1):
                smooth_term = w1 * (2*P[i] - P[i-1] - P[i+1])
                data_term = w2 * (P[i] - Q[i])
                nabla_J[i] = smooth_term + data_term
            max_grad = numpy.max(numpy.linalg.norm(nabla_J, axis=1))
            if max_grad <= tol:
                break
            P -= epsilon * nabla_J
            steps += 1

        print(f"Path smoothing completed in {steps} steps")
        return P

def main(args=None):
    rclpy.init(args=args)
    node = PathSmoothingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
