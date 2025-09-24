#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# PATH PLANNING BY A-STAR
#
# Instructions:
# Write the code necessary to plan a path using an
# occupancy grid and the A* algorithm
# MODIFY ONLY THE SECTIONS MARKED WITH THE 'TODO' COMMENT
#
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from geometry_msgs.msg import PoseStamped, Pose, Point
from nav_msgs.msg import Path
from nav_msgs.srv import *
from builtin_interfaces.msg import Duration
from collections import deque
import numpy
import heapq
import math

import csv

NAME = "Juan Mancera Lopez"

class AStarNode(Node):

    tiempo = []
    ex_reng = []
    prev_sx = 0
    prev_sy = 0
    prev_gx = 100
    prev_gy = 100
    fallos = 0
    intentos = 0

    def a_star(self, start_r, start_c, goal_r, goal_c, grid_map, cost_map, use_diagonals):
        [height, width] = grid_map.shape
        in_open_list   = numpy.full(grid_map.shape, False)
        in_closed_list = numpy.full(grid_map.shape, False)
        g_values       = numpy.full(grid_map.shape, float("inf"))
        f_values       = numpy.full(grid_map.shape, float("inf"))
        parent_nodes   = numpy.full((grid_map.shape[0],grid_map.shape[1],2),-1)
        open_list = []
        if use_diagonals: #Every adjacent node has: [row_offset, col_offset, cost]
            adjacents = [[1,0,1],[0,1,1],[-1,0,1],[0,-1,1], [1,1,1.414], [-1,1,1.414], [-1,-1,1.414],[1,-1,1.414]]
        else:
            adjacents = [[1,0,1],[0,1,1],[-1,0,1],[0,-1,1]]

        heapq.heappush(open_list, (0, [start_r, start_c]))
        in_open_list[start_r, start_c] = True
        g_values    [start_r, start_c] = 0
        [row, col]= [start_r, start_c]   #Current node
        #
        # TODO:
        # Implement the A* algorithm for path planning
        # Map is considered to be a 2D array and start and goal positions
        # are given as row-col pairs. You can follow these steps:
        #
        # WHILE open list is not empty and current is different from goal:
        #     Get current node [row,col] from open list (see heapq.heappop function)
        #     Mark current node as 'in_closed_list'
        #     For [r,c,cost] in adjacent nodes:
        #         Get r,c indices of neighbours of current node (check content of adjacents)
        #         Discard if r,c is out of map, occupied, unknonw or in closed list, and continue
        #         get a g-value g as: g-value of current node + dist + cost of neighbour r,c
        #         Calculate heuristic 
        #         Calculate f-value
        #         IF g < g_value of neighbour r,c:
        #             set g as g_value of neighbour r,c
        #             set f as f_value of neighbour r,c
        #             SET current node row,col as parent of neighbour r,c
        #         If neighbour r,c is not in open list:
        #             mark r,c as 'in_open_list'
        #             add r,c to open list (check heapq.heappush)
        #
        while open_list != [] and [row, col] != [goal_r,goal_c]:
            [row, col] = heapq.heappop(open_list)[1]
            #print("height: ", height, "  -  wid: ", width)
            #print("row: ", row, "  - col:", col)
            in_closed_list[row, col] = True
            in_open_list[row, col] = False
            for adjacent in adjacents:
                row_neight = row+adjacent[0]
                col_neight = col+adjacent[1]
                #print("Vecino row: ", row_neight, "  - col:", col_neight)
                if (col_neight>=width or col_neight<0) or (row_neight>=height or row_neight<0):
                    continue
                if (in_closed_list[row_neight, col_neight] == True):
                    continue
                if (grid_map[row_neight,col_neight] >= 50):
                    continue
                g = g_values[row, col] + adjacent[2] + cost_map[row_neight,col_neight]
                #Traer valor heurística del mapa de costo
                h = math.sqrt((pow((row-row_neight),2)) + (pow((col-col_neight),2)))
                f = g + h
                if g < g_values[row_neight,col_neight]:
                    g_values[row_neight,col_neight] = g
                    f_values[row_neight,col_neight] = f
                    parent_nodes[row_neight,col_neight] = [row,col]
                if not in_open_list[row_neight,col_neight]:
                    in_open_list[row_neight,col_neight] = True
                    heapq.heappush(open_list, (f, [row_neight,col_neight]))
                #print(open_list, " --- ")
        #
        # END OF WHILE
        #
        path = []
        #Marcar error si no hubo
        if goal_r>=height or goal_c>=width:
            return path
        while parent_nodes[goal_r, goal_c][0] != -1:
            path.insert(0, [goal_r, goal_c])
            [goal_r, goal_c] = parent_nodes[goal_r, goal_c]
        return path

    def get_maps(self):
        print("Waiting for inflated map service...")
        while not self.clt_inflated_map.wait_for_service(timeout_sec=1.0):
            print('Waiting for inflated map service...')
        print("Inflated map service is now available...")
        print("Waiting for cost map service...")
        while not self.clt_cost_map.wait_for_service(timeout_sec=1.0):
            print('Waiting for cost map service...')
        print("Cost map service is now available...")

        print("Trying to get inflated map...")
        future = self.clt_inflated_map.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        inflated_map = future.result().map
        print("Got inflated map.")
        print("Trying to get cost map...")
        future = self.clt_cost_map.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        cost_map= future.result().map
        print("Got cost map.")
        return [inflated_map, cost_map]

    def get_path_msg(self, path, res, zx, zy):
        msg_path = Path()
        msg_path.header.frame_id = "map"
        msg_path.header.stamp = self.get_clock().now().to_msg()
        msg_path.poses = []
        for [r,c] in path:
            msg_path.poses.append(PoseStamped(pose=Pose(position=Point(x=(c*res + zx), y=(r*res + zy)))))
        return msg_path

    def callback_a_star(self, req, resp):
        info = self.inflated_map.info
        res = info.resolution
        [sx, sy] = [req.start.pose.position.x, req.start.pose.position.y]
        [gx, gy] = [req.goal .pose.position.x, req.goal .pose.position.y]
        [zx, zy] = [self.inflated_map.info.origin.position.x, self.inflated_map.info.origin.position.y]
        use_diagonals = self.get_parameter('diagonals').get_parameter_value().bool_value
        inflated_grid = numpy.reshape(numpy.asarray(self.inflated_map.data), (info.height, info.width))
        cost_grid     = numpy.reshape(numpy.asarray(self.cost_map.data)    , (info.height, info.width))
        
        print("Planning path by A* from " + str([sx, sy])+" to "+str([gx, gy]))
        start_time = self.get_clock().now()
        path = self.a_star(int((sy-zy)/res), int((sx-zx)/res), int((gy-zy)/res), int((gx-zx)/res),
                           inflated_grid, cost_grid, use_diagonals)
        end_time = self.get_clock().now()
        delta_ms = (end_time.nanoseconds - start_time.nanoseconds)/1e6
        if len(path) > 0:
            print("Path planned after " + str(delta_ms) + " ms with " +  str(len(path)) + " points")
            nuevo_fallo = 0
        else:
            print("Cannot plan path from  " + str([sx, sy])+" to "+str([gx, gy]) + " :'(")
            #self.fallos += 1
            nuevo_fallo = 1

        # Multiples intentos
        if  [sx, sy, gx, gy] == [self.prev_sx, self.prev_sy, self.prev_gx, self.prev_gy]:
            self.tiempo.append(float(delta_ms))
            self.intentos += 1
            self.fallos += nuevo_fallo
        else:
            if self.intentos > 1:
                print("Guardando dato")
                conj = ([round(self.prev_sx,3), round(self.prev_sy,3)], [round(self.prev_gx,3), round(self.prev_gy,3)], self.intentos, self.fallos, round(min(self.tiempo),3), round(max(self.tiempo),3), round(numpy.mean(self.tiempo),3))
                self.ex_reng.append(conj)
            self.intentos = 1
            self.fallos = nuevo_fallo
            self.tiempo = []
            self.tiempo.append(float(delta_ms))
        radio_cost = self.get_parameter('cost_radius').get_parameter_value().double_value

        # Comandos
        if ([sx,sy] == [0,0] and [gx,gy] == [15,15]):
            #Empezar a grabar datos
            print("Inicio almacenamiento de datos")
            self.ex_reng = []
            self.fallos = 0
            self.intentos = 0
            self.tiempo = []

            data_info = ("Diagonals=", use_diagonals, "", "Cost Radius=", radio_cost)
            self.ex_reng.append(data_info)
            data_info = ("Pos inicial", "Pos final", "Cant. Intentos", "Cant. Fallos", "MinTmp(ms)", "MaxTmp(ms)", "AvgTmp(ms)")
            self.ex_reng.append(data_info)
        if ([sx,sy] == [0,0] and [gx,gy] == [16,16]):
            #Guardar en CSV
            nombreArchivo = "SalidaDatos_D" + str(use_diagonals) + "_CoRad_" + str(radio_cost).replace('.','_') + ".csv"
            with open(nombreArchivo, mode="w", newline="", encoding="UTF-8") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerows(self.ex_reng)
            print("FINAL - Guardado de datos en: ", nombreArchivo)
            print("")
        if ([sx,sy] == [0,0] and [gx,gy] == [18,18]):
            self.ejecucion_casos()
        self.msg_path = self.get_path_msg(path, res, zx, zy)
        resp.plan = self.msg_path

        self.prev_sx = sx
        self.prev_sy = sy
        self.prev_gx = gx
        self.prev_gy = gy

        return resp

    def ejecucion_casos(self):
        Rutas = [[[0,0], [-2,4]],       [[0,0], [-1.5,5.3]],    [[2,0], [2,4.3]],           [[2,4.3], [-4.5,-1.5]],
                [[0,0], [2,5.5]],       [[2,5.5], [0,0]],       [[2,5.5], [-2,1]],          [[-2,1], [2,5.5]],
                [[2,5.5], [-4.5,-9]],   [[-4.5,-9], [2,5.5]],   [[0,0], [1, -12]],          [[2, 4.3], [1, -12]], 
                [[2.5, 5], [1, -12]],   [[0,0], [-4.5, -9]],    [[-4.5, -9], [2.5, -10]],   [[-4.5, -9], [-5, 0]], 
                [[-4.5, -9], [2, 4.3]]]
        print("-- Probando algoritmo con rutas establecidas")
        for ruta in Rutas:
            for cuenta in range(5): 
                # Crear petición
                req = GetPlan.Request()
                req.start = PoseStamped(pose=Pose(position=Point(x=ruta[0][0],y=ruta[0][1])))
                req.goal  = PoseStamped(pose=Pose(position=Point(x=ruta[1][0],y=ruta[1][1])))
                #req.start = PoseStamped(pose=Pose(position=Point(x=-5.0, y=-5.0)))
                #req.goal  = PoseStamped(pose=Pose(position=Point(x=5.0, y=-3.0)))
                # Crear respuesta vacía
                resp = GetPlan.Response()

                # Llamar callback
                resp = self.callback_a_star(req, resp)
        print("-- Fin prueba algoritmo con rutas establecidas")
        return 0

    def callback_timer(self):
        self.pub_path.publish(self.msg_path)
            
    def __init__(self):
        print("INITIALIZING RRT NODE - ", NAME)
        super().__init__("rrt_node")
        self.clt_inflated_map = self.create_client(GetMap, '/get_inflated_map')
        self.clt_cost_map     = self.create_client(GetMap, '/get_cost_map')
        
        self.declare_parameter('cost_radius', 0.05)

        [self.inflated_map, self.cost_map] = self.get_maps()
        self.declare_parameter('diagonals', False)
        self.srv_plan_path = self.create_service(GetPlan, '/path_planning/plan_path', self.callback_a_star)
        self.pub_path = self.create_publisher(Path, '/path_planning/path', 10)
        self.msg_path = Path()
        self.timer = self.create_timer(0.5, self.callback_timer)
            
def main(args=None):
    rclpy.init(args=args)
    a_star_node = AStarNode()
    rclpy.spin(a_star_node)
    a_star_node.destroy_node()
    rclpy.shutdown()


    
if __name__ == '__main__':
    main()
