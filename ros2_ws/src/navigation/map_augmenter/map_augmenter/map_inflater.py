#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# MAP INFLATION (non-blocking init)
#
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import numpy as np

FULL_NAME = "Popoca Zuñiga Daniel Ixbalanque"

class MapInflaterNode(Node):
    def get_inflated_map(self, static_map, inflation_cells):
        """Inflar obstáculos en 'static_map' en radio 'inflation_cells' (celdas)."""
        self.get_logger().info("Inflating map by %d cells" % (inflation_cells,))
        inflated = np.copy(static_map)
        height, width = static_map.shape

        if inflation_cells <= 0:
            return inflated

        # Recorremos cada celda ocupada (>50) y marcamos vecinos en rango [-n, n]
        for i in range(height):
            for j in range(width):
                if static_map[i, j] > 50:
                    for dx in range(-inflation_cells, inflation_cells + 1):
                        for dy in range(-inflation_cells, inflation_cells + 1):
                            ni = i + dx
                            nj = j + dy
                            if ni < 0 or ni >= height or nj < 0 or nj >= width:
                                continue
                            inflated[ni, nj] = 100
        return inflated

    def callback_inflated_map(self, request, response):
        response.map = self.inflated_map
        return response

    def callback_timer(self):
        if not hasattr(self, 'map_static') or self.map_static is None:
            return

        info = self.map_static.info
        height = info.height
        width = info.width
        res = info.resolution

        try:
            static_arr = np.reshape(np.asarray(self.map_static.data, dtype='int8'), (height, width))
        except Exception as e:
            self.get_logger().error("Error reshaping static map: " + str(e))
            return

        inflation_m = float(self.get_parameter('inflation_radius').get_parameter_value().double_value)
        inflation_cells = max(0, int(round(inflation_m / res)))

        inflated_arr = self.get_inflated_map(static_arr, inflation_cells)

        flat = np.ravel(inflated_arr).astype(int).tolist()
        self.inflated_map = OccupancyGrid(info=info, data=flat)
        self.inflated_map.header.frame_id = "map"
        self.inflated_map.header.stamp = self.get_clock().now().to_msg()
        self.pub_inflated_map.publish(self.inflated_map)

    def _map_srv_cb(self, future):
        try:
            result = future.result()
            self.map_static = result.map
            self.get_logger().info("Received static map (async). Size: %dx%d" % (self.map_static.info.width, self.map_static.info.height))
            if not getattr(self, '_ready', False):
                self.srv_inflate_map  = self.create_service(GetMap, '/get_inflated_map', self.callback_inflated_map)
                self.pub_inflated_map = self.create_publisher(OccupancyGrid, '/inflated_map', 10)
                self.timer = self.create_timer(1.0, self.callback_timer)
                self._ready = True
        except Exception as e:
            self.get_logger().error("Failed to get static map: %s" % (e,))

    def __init__(self):
        print("INITIALIZING MAP INFLATER NODE - ", FULL_NAME)
        super().__init__("map_inflater_node")

        # Client al map_server (no bloquear)
        self.clt_static_map = self.create_client(GetMap, '/map_server/map')
        self.declare_parameter('inflation_radius', 0.05)

        # flags y estados
        self._requested_map = False
        self._ready = False
        self.map_static = None
        self.inflated_map = None

        # Timer para pedir el mapa asíncronamente cuando el servicio esté listo
        self.check_timer = self.create_timer(0.5, self._try_request_map)

    def _try_request_map(self):
        if self._requested_map:
            return
        if self.clt_static_map.service_is_ready():
            fut = self.clt_static_map.call_async(GetMap.Request())
            fut.add_done_callback(self._map_srv_cb)
            self._requested_map = True
            self.get_logger().info("Requested static map (async).")
        else:
            self.get_logger().debug("Waiting for static map service...")

def main(args=None):
    rclpy.init(args=args)
    map_inflater_node = MapInflaterNode()
    rclpy.spin(map_inflater_node)
    map_inflater_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
