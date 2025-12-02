# MOBILE ROBOTS - FI-UNAM, 2026-1
# PATH PLANNING BY A-STAR
#
# Daniel Ixbalanque Popoca Zuñiga
#
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetMap
from builtin_interfaces.msg import Duration
import numpy
import heapq
import math


NAME = "Daniel Ixbalanque Popoca Zuñiga"


class AStarNode(Node):

    # ---------------------------------------------------------
    #       A* COMPLETO Y ROBUSTO
    # ---------------------------------------------------------
    def a_star(self, start_r, start_c, goal_r, goal_c, grid_map, cost_map, use_diagonals):

        height, width = grid_map.shape

        g = numpy.full((height, width), float('inf'))
        f = numpy.full((height, width), float('inf'))
        visited = numpy.full((height, width), False)
        parent = numpy.full((height, width, 2), -1)

        if use_diagonals:
            moves = [(1,0,1), (-1,0,1), (0,1,1), (0,-1,1),
                     (1,1,1.414), (-1,1,1.414), (-1,-1,1.414), (1,-1,1.414)]
        else:
            moves = [(1,0,1), (-1,0,1), (0,1,1), (0,-1,1)]

        g[start_r, start_c] = 0
        f[start_r, start_c] = math.sqrt((start_r - goal_r)**2 + (start_c - goal_c)**2)

        open_list = []
        heapq.heappush(open_list, (f[start_r, start_c], (start_r, start_c)))

        while open_list:
            _, (r, c) = heapq.heappop(open_list)
            if visited[r, c]:
                continue

            visited[r, c] = True

            if r == goal_r and c == goal_c:
                break

            for dr, dc, cost in moves:
                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if grid_map[nr, nc] != 0:
                    continue
                if visited[nr, nc]:
                    continue

                tentative_g = g[r, c] + cost + cost_map[nr, nc]
                heuristic = math.sqrt((nr - goal_r)**2 + (nc - goal_c)**2)
                tentative_f = tentative_g + heuristic

                if tentative_g < g[nr, nc]:
                    g[nr, nc] = tentative_g
                    f[nr, nc] = tentative_f
                    parent[nr, nc] = [r, c]
                    heapq.heappush(open_list, (tentative_f, (nr, nc)))

        # Reconstrucción
        path = []
        cr, cc = goal_r, goal_c

        if parent[cr, cc][0] == -1:
            return []

        while not (cr == start_r and cc == start_c):
            path.insert(0, [cr, cc])
            cr, cc = parent[cr, cc]

        path.insert(0, [start_r, start_c])
        return path

    # ---------------------------------------------------------
    #     SMOOTH POR PROMEDIO MÓVIL (ROBUSTO)
    # ---------------------------------------------------------
    def smooth_path(self, path, k=4):
        if len(path) < k:
            return path

        smooth = []
        for i in range(len(path)):
            r_sum, c_sum, count = 0.0, 0.0, 0
            for j in range(-k, k + 1):
                idx = i + j
                if 0 <= idx < len(path):
                    r_sum += path[idx][0]
                    c_sum += path[idx][1]
                    count += 1
            smooth.append([r_sum / count, c_sum / count])

        return smooth

    # ---------------------------------------------------------
    #     OBTENER MAPAS
    # ---------------------------------------------------------
    def get_maps(self):
        print("Waiting for inflated map service...")
        while not self.clt_inflated_map.wait_for_service(timeout_sec=1.0):
            print('Waiting for inflated map service...')

        print("Waiting for cost map service...")
        while not self.clt_cost_map.wait_for_service(timeout_sec=1.0):
            print('Waiting for cost map service...')

        print("Trying to get inflated map...")
        future = self.clt_inflated_map.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        inflated_map = future.result().map

        print("Trying to get cost map...")
        future = self.clt_cost_map.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        cost_map = future.result().map

        return [inflated_map, cost_map]

    # ---------------------------------------------------------
    #     CONVERTIR PATH A MENSAJE ROS
    # ---------------------------------------------------------
    def get_path_msg(self, path, res, zx, zy):
        msg_path = Path()
        msg_path.header.frame_id = "map"
        msg_path.header.stamp = self.get_clock().now().to_msg()
        msg_path.poses = []

        for [r, c] in path:
            msg_path.poses.append(
                PoseStamped(
                    pose=Pose(
                        position=Point(x=(c * res + zx), y=(r * res + zy))
                    )
                )
            )

        return msg_path

    # ---------------------------------------------------------
    #         CALLBACK DEL SERVICIO A*
    # ---------------------------------------------------------
    def callback_a_star(self, req, resp):
        info = self.inflated_map.info
        res = info.resolution

        sx = req.start.pose.position.x
        sy = req.start.pose.position.y
        gx = req.goal.pose.position.x
        gy = req.goal.pose.position.y

        zx = info.origin.position.x
        zy = info.origin.position.y

        use_diagonals = self.get_parameter('diagonals').get_parameter_value().bool_value

        inflated_grid = numpy.reshape(numpy.asarray(self.inflated_map.data), (info.height, info.width))
        cost_grid = numpy.reshape(numpy.asarray(self.cost_map.data), (info.height, info.width))

        print("Planning path by A* from", [sx, sy], "to", [gx, gy])

        start_r = int((sy - zy) / res)
        start_c = int((sx - zx) / res)
        goal_r  = int((gy - zy) / res)
        goal_c  = int((gx - zx) / res)

        path = self.a_star(start_r, start_c, goal_r, goal_c, inflated_grid, cost_grid, use_diagonals)

        if len(path) == 0:
            print("⚠ No se pudo planear un camino.")
            resp.plan = Path()
            return resp

        # -------------------------
        # APPLY SMOOTH
        # -------------------------
        path = self.smooth_path(path, k=4)

        self.msg_path = self.get_path_msg(path, res, zx, zy)
        resp.plan = self.msg_path

        print("Path planned with", len(path), "points (smoothed)")
        return resp

    # ---------------------------------------------------------
    #         PUBLICAR PATH
    # ---------------------------------------------------------
    def callback_timer(self):
        self.pub_path.publish(self.msg_path)

    # ---------------------------------------------------------
    #                  INIT NODE
    # ---------------------------------------------------------
    def __init__(self):
        print("INITIALIZING A* NODE -", NAME)
        super().__init__("a_star_node")

        self.clt_inflated_map = self.create_client(GetMap, '/get_inflated_map')
        self.clt_cost_map = self.create_client(GetMap, '/get_cost_map')

        [self.inflated_map, self.cost_map] = self.get_maps()

        self.declare_parameter('diagonals', False)

        self.srv_plan_path = self.create_service(GetPlan, '/path_planning/plan_path', self.callback_a_star)
        self.pub_path = self.create_publisher(Path, '/path_planning/path', 10)

        self.msg_path = Path()
        self.timer = self.create_timer(0.5, self.callback_timer)


# ---------------------------------------------------------
#                   MAIN
# ---------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
