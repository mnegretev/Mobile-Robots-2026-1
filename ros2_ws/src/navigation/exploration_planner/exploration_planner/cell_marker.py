#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.srv import GetMap
import numpy as np

class GridPointsNode(Node):
    def __init__(self):
        super().__init__('grid_points_node')
        self.pub_points = self.create_publisher(MarkerArray, '/cell_marker', 10)  # topic: all points
        self.sub_visited = self.create_subscription(MarkerArray, '/visited_points', self.cb_visited, 10)

        self.visited_points = set()  # Conjunto de puntos visitados

        # Conectarse al servicio de mapa
        self.clt_map = self.create_client(GetMap, '/get_inflated_map')
        self.get_logger().info("Waiting for map service...")
        while not self.clt_map.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for map service...")
        self.get_logger().info("Map service available!")

        # Obtener mapa
        future = self.clt_map.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        self.map_data = future.result().map

        # Crear markers
        self.marker_array = self.create_marker_array()

        # Publicar cada 0.5 s
        self.timer = self.create_timer(0.5, self.publish_points)

    def cb_visited(self, msg):
        """Actualizar puntos visitados recibidos desde el navegador"""
        for m in msg.markers:
            x = round(m.pose.position.x, 2)
            y = round(m.pose.position.y, 2)
            self.visited_points.add((x, y))

    def create_marker_array(self):
        info = self.map_data.info
        width = info.width
        height = info.height
        res = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        grid = np.reshape(np.array(self.map_data.data, dtype=int), (height, width))
        marker_array = MarkerArray()
        marker_id = 0

        visited = set()       # puntos agregados
        to_visit = [(0.0, 0.0)]  # empezamos desde 0,0

        while to_visit:
            x, y = to_visit.pop(0)
            i = int((y - origin_y)/res)
            j = int((x - origin_x)/res)

            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if (x, y) in visited:
                continue

            # Área libre alrededor 0.4 m
            radius_cells = int(0.4/res)
            area_free = True
            for di in range(-radius_cells, radius_cells+1):
                for dj in range(-radius_cells, radius_cells+1):
                    ni = i + di
                    nj = j + dj
                    if ni < 0 or ni >= height or nj < 0 or nj >= width:
                        continue
                    cell_value = grid[ni, nj]
                    if cell_value != 0 or cell_value == -1:
                        area_free = False
                        break
                if not area_free:
                    break
            if not area_free:
                continue

            # Crear Marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cell_marker"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)
            visited.add((x, y))

            # Agregar vecinos a 1 m
            neighbors = [
                (x + 1.0, y), (x - 1.0, y),
                (x, y + 1.0), (x, y - 1.0)
            ]
            for nx, ny in neighbors:
                if (nx, ny) not in visited:
                    to_visit.append((nx, ny))

        return marker_array

    def publish_points(self):
        for m in self.marker_array.markers:
            x = round(m.pose.position.x, 2)
            y = round(m.pose.position.y, 2)

            # Cambiar color si visitado
            if (x, y) in self.visited_points:
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            else:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0

            m.header.stamp = self.get_clock().now().to_msg()

        self.pub_points.publish(self.marker_array)
        self.get_logger().info(f"Published {len(self.marker_array.markers)} points!")


def main(args=None):
    rclpy.init(args=args)
    node = GridPointsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
