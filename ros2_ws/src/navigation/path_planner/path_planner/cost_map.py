#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2026-1
# COST MAPS (non-blocking init, robust)
#
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import numpy as np

FULL_NAME = "Daniel Ixbalanque Popoca Zuñiga"

class CostMapNode(Node):
    def get_cost_map(self, static_map, cost_radius):
        """Calcula el cost map a partir del mapa estático (numpy 2D) y un radio en celdas."""
        self.get_logger().info("Getting cost map with " + str(cost_radius) + " cells")
        # Inicializamos con ceros (zona libre). Mantendremos 100 para ocupados.
        cost_map = np.zeros_like(static_map, dtype=int)

        height, width = static_map.shape

        # Marca ocupados en 100 y luego propaga costos
        occupied_mask = (static_map > 50)
        cost_map[occupied_mask] = 100

        if cost_radius <= 0:
            return cost_map

        # Para cada celda ocupada inflamos y asignamos costos según la distancia en Chebyshev
        # (cost = cost_radius - max(|dx|,|dy|) + 1)
        for i in range(height):
            for j in range(width):
                if static_map[i, j] > 50:
                    # recorrer vecinos incluyendo el límite (±cost_radius)
                    for dx in range(-cost_radius, cost_radius + 1):
                        for dy in range(-cost_radius, cost_radius + 1):
                            ni = i + dx
                            nj = j + dy
                            if ni < 0 or ni >= height or nj < 0 or nj >= width:
                                continue
                            # si es ocupado lo dejamos como 100
                            if cost_map[ni, nj] == 100:
                                continue
                            # calcular costo (mayor cerca del obstáculo)
                            c = cost_radius - max(abs(dx), abs(dy)) + 1
                            # conservar el costo mayor (más conservador)
                            if c > cost_map[ni, nj]:
                                cost_map[ni, nj] = int(c)
        return cost_map

    # callback del servicio que devuelve el cost_map
    def callback_cost_map(self, request, response):
        response.map = self.cost_map
        return response

    # timer que publica el cost_map periódicamente (solo si ya está calculado)
    def callback_timer(self):
        if not hasattr(self, 'map_static') or self.map_static is None:
            return

        info = self.map_static.info
        height = info.height
        width = info.width
        res = info.resolution

        # reshape seguro del mapa estático
        try:
            static_arr = np.reshape(np.asarray(self.map_static.data, dtype='int8'), (height, width))
        except Exception as e:
            self.get_logger().error("Error reshaping static map: " + str(e))
            return

        cost_radius_m = float(self.get_parameter('cost_radius').get_parameter_value().double_value)
        # convierte metros a celdas (redondeo)
        cost_radius_cells = max(0, int(round(cost_radius_m / res)))

        cost_map_arr = self.get_cost_map(static_arr, cost_radius_cells)

        # Armar OccupancyGrid.data (lista de ints)
        flat = np.ravel(cost_map_arr).astype(int).tolist()
        self.cost_map = OccupancyGrid(info=info, data=flat)
        self.cost_map.header.frame_id = "map"
        self.cost_map.header.stamp = self.get_clock().now().to_msg()
        self.pub_cost_map.publish(self.cost_map)

    # callback asíncrono para recibir el mapa del servidor
    def _map_srv_cb(self, future):
        try:
            result = future.result()
            self.map_static = result.map
            self.get_logger().info("Received static map (async). Size: %dx%d" % (self.map_static.info.width, self.map_static.info.height))
            # Si no hemos creado el servicio/publisher, crearlos
            if not getattr(self, '_ready', False):
                self.srv_cost_map  = self.create_service(GetMap, '/get_cost_map', self.callback_cost_map)
                self.pub_cost_map = self.create_publisher(OccupancyGrid, '/cost_map', 10)
                self.timer = self.create_timer(1.0, self.callback_timer)
                self._ready = True
        except Exception as e:
            self.get_logger().error("Failed to get static map: %s" % (e,))

    def __init__(self):
        print("INITIALIZING COST MAP NODE - ", FULL_NAME)
        super().__init__("cost_map_node")

        # Client asíncrono (no bloqueante)
        self.clt_static_map = self.create_client(GetMap, '/map_server/map')
        self.declare_parameter('cost_radius', 0.05)

        # flags
        self._requested_map = False
        self._ready = False
        self.map_static = None
        self.cost_map = None

        # Timer para comprobar disponibilidad del servicio y solicitar el mapa asíncronamente
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
    cost_map_node = CostMapNode()
    rclpy.spin(cost_map_node)
    cost_map_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
