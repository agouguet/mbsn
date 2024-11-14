from collections import defaultdict
from importlib.resources import files
from math import ceil, cos, radians, sin, sqrt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from shapely import GeometryCollection, MultiPolygon, Point, Polygon as ShapelyPolygon, box
import cv2
from .map import Map
from .polygon import Polygon

POURCENTAGE_AREA_OF_POLYGON_ACCEPTABLE = 0.3


class PolygonalMap(Map, Polygon):

    def __init__(self, map, type='hexagon', polygon_size=0.5, area_minimum = POURCENTAGE_AREA_OF_POLYGON_ACCEPTABLE):
        Map.__init__(self, self.map_name_to_map_path(map))
        self._polygon = None
        self._grid = None
        self._type = type
        self._area_minimum = area_minimum
        self._polygon_size = polygon_size
        
        # # self.visibility_polygon = VisibilityPolygon(vertices, polygon)

        # self.test_grid = self.create_test_grid(polygon)
        

    def map_name_to_map_path(self, map_name):
        if "\\" in map_name or "/" in map_name:
            return map_name
        return str(files('mbsn.data').joinpath(map_name+'/'))

    @property
    def polygon(self):
        if self._polygon is not None:
            return self._polygon
        
        scale_factor = self.config["resolution"]
        origin = self.config["origin"]

        ret, mask = cv2.threshold(cv2.cvtColor(self.color.copy(), cv2.COLOR_BGR2GRAY)[:, :], 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(np.invert(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        vertices = []
        polygons = []
        for c in range(len(contours)):
            poly = []
            for p in contours[c]:
                n = tuple([p[0][0]* scale_factor + origin[0], p[0][1]* scale_factor + origin[1]])
                vertices.append(Point(n[0], n[1]))
                poly.append(n)
            if len(poly)>=4:
                polygons.append(ShapelyPolygon(poly))

        if len(polygons) == 1:
            return vertices, ShapelyPolygon(list(polygons[0].exterior.coords)), []

        outer = polygons[1]
        inners = polygons[2:]

        exterior = list(outer.exterior.coords)
        interiors = [list(inner.exterior.coords) for inner in inners]

        inners.append(polygons[0]-polygons[1])


        Polygon.__init__(self, exterior, interiors)

        return self._polygon


    @property
    def grid(self):
        if self._grid is not None:
            return self._grid

        grid = {}
        polygon = self.polygon
        minx, miny, maxx, maxy = polygon.bounds

        if self._type == "square":
            squares = []
            square_dict = defaultdict(list)
            
            x = minx
            while x < maxx:
                y = miny
                while y < maxy:
                    square_polygon = box(x, y, x + self._polygon_size, y + self._polygon_size)
                    if polygon.buffer(0).intersects(square_polygon):
                        inter = polygon.intersection(square_polygon)
                        if isinstance(inter, MultiPolygon) or isinstance(inter, GeometryCollection):
                            for poly in inter.geoms:
                                if isinstance(poly, ShapelyPolygon) and polygon.intersects(poly) and poly.area >= square_polygon.area * self._area_minimum:
                                    new_cell = Polygon(poly)
                                    grid[new_cell.id] = new_cell
                                    squares.append(new_cell)
                                    square_dict[(x, y)].append(new_cell)
                        elif isinstance(inter, ShapelyPolygon):
                            if inter.area >= square_polygon.area * self._area_minimum:
                                new_cell = Polygon(inter)
                                grid[new_cell.id] = new_cell
                                squares.append(new_cell)
                                square_dict[(x, y)].append(new_cell)
                    y += self._polygon_size
                x += self._polygon_size
            
            
            for (x, y), list_square in square_dict.items():
                neighbors_coords = [
                    (x - self._polygon_size, y), (x + self._polygon_size, y),  # left, right
                    (x, y - self._polygon_size), (x, y + self._polygon_size)   # bottom, top
                ]
                for square in list_square:
                    for nx, ny in neighbors_coords:
                        if (nx, ny) in square_dict:
                            neighbor_square = square_dict[(nx, ny)]
                            for neighbor in neighbor_square:
                                if square.polygon.buffer(0.05).intersects(neighbor.polygon):
                                    square.add_neighbor(neighbor)
            self._grid = grid

        elif self._type == "hexagon":
            hexagons = []
            hexagon_dict = defaultdict(list)

            hex_width = self._polygon_size * 2
            hex_height = self._polygon_size * 3 ** 0.5

            xo = self._polygon_size * (1 + cos(radians(60)))  # X offset
            yo = self._polygon_size * sin(radians(60)) * 2  # Y offset

            minx = ceil((minx - xo) / xo)
            maxx = ceil((maxx + xo) / xo)
            miny = ceil((miny - yo) / yo)
            maxy = ceil((maxy + yo) / yo)

            angle = 2 * 3.14159 / 6

            x = minx
            while x < maxx:
                y = miny
                while y < maxy:
                    hex_x = x * hex_width * 0.75
                    hex_y = y * hex_height + (x % 2) * hex_height / 2
                    # hexagon_polygon = NavHexagon((hex_x, hex_y), hexagon_size)
                    hexagon_polygon = ShapelyPolygon([(hex_x + self._polygon_size * cos(angle * i), hex_y + self._polygon_size * sin(angle * i)) for i in range(6)])

                    if polygon.buffer(0).intersects(hexagon_polygon):
                        inter = polygon.intersection(hexagon_polygon)
                        if isinstance(inter, MultiPolygon) or isinstance(inter, GeometryCollection):
                            for poly in inter.geoms:
                                if isinstance(poly, ShapelyPolygon) and polygon.intersects(poly) and poly.area >= hexagon_polygon.area * self._area_minimum:
                                    new_cell = Polygon(poly)
                                    grid[new_cell.id] = new_cell
                                    hexagons.append(new_cell)
                                    hexagon_dict[(x, y)].append(new_cell)
                        elif isinstance(inter, ShapelyPolygon):
                            if inter.area >= hexagon_polygon.area * self._area_minimum:
                                new_cell = Polygon(inter)
                                grid[new_cell.id] = new_cell
                                hexagons.append(new_cell)
                                hexagon_dict[(x, y)].append(new_cell)
                    y += 1
                x += 1

            directions_odd = [
                (+1, 0), 
                (+1, +1),
                (0, +1),
                (-1, +1),
                (-1, 0),
                (0, -1)
            ]

            directions_even = [
                (+1, -1), 
                (+1, 0),
                (0, +1),
                (-1, 0),
                (-1, -1),
                (0, -1)
            ]

            for (x, y), list_hexagon in hexagon_dict.items():
                for hexagon in list_hexagon:
                    directions = directions_even if x%2 == 0 else directions_odd
                    for nx, ny in directions:
                        if (x+nx, y+ny) in hexagon_dict:
                            neighbors_hexagon = hexagon_dict[(x+nx, y+ny)]
                            for neighbor in neighbors_hexagon:
                                if hexagon.polygon.buffer(0.05).intersects(neighbor.polygon):
                                    hexagon.add_neighbor(neighbor)
            self._grid = grid

        return self._grid


























    def get_rooms(self, tolerance=1.0):
        scenario = self.scenario
        rooms={}
        _, map_config = get_map_config(scenario)
        _, map, segmented_map = load_map_and_segmented_map(scenario)
        for i in np.unique(segmented_map)[1:-1]:
            mask = (segmented_map == i) * np.uint8(1)
            room = cv2.bitwise_and(map, map, mask= mask)
            room = cv2.cvtColor(room.copy(), cv2.COLOR_BGR2GRAY)
            _, poly, _ = image_to_polygon(room)
            poly = poly.simplify(tolerance, preserve_topology=True)
            poly = scale_polygon(poly, map_config)
            if poly.is_valid:
                room = NavRoom(poly)
                rooms[room.id] = room

        neighbors_pair = []
        for id1, room1 in rooms.items():
            for id2, room2 in rooms.items():
                if id1 != id2 and (id1, id2) not in neighbors_pair:
                    if room1.intersects(room2):
                        neighbors_pair.append((id1, id2))
                        neighbors_pair.append((id2, id1))
                        room1.add_neighbor(room2)
                        room2.add_neighbor(room1)
        return rooms
                
    def create_test_grid(self, polygon, buffer_size=0.05):
        grid = {}
        id_to_id = defaultdict(list)
        
        for id, cell in self.grid.items():
            if polygon.intersects(cell.polygon):
                inter = polygon.intersection(cell.polygon)
                if isinstance(inter, MultiPolygon) or isinstance(inter, GeometryCollection):
                    for poly in inter.geoms:
                        if isinstance(poly, ShapelyPolygon) and polygon.intersects(poly) and poly.area >= 0.1:
                            new_cell = Polygon(poly)
                            grid[new_cell.id] = new_cell
                            id_to_id[cell.id].append(new_cell.id)
                elif isinstance(inter, ShapelyPolygon) and inter.area >= 0.1:
                        new_cell = Polygon(inter)
                        grid[new_cell.id] = new_cell
                        id_to_id[cell.id].append(new_cell.id)

        for _ ,ids in id_to_id.items():
            if len(ids) > 1:
                for id1 in ids:
                    poly1 = grid[id1]
                    for id2 in ids:
                        if id1 != id2:
                            poly2 = grid[id2]
                            if poly1.polygon.buffer(0.01).intersects(poly2.polygon):
                                poly1.add_neighbor(poly2)



        for id, poly in self.grid.items():
            for new_poly_id in id_to_id[id]:
                new_poly = grid[new_poly_id]
                for n in poly.neighbors:
                    for new_poly_n_id in id_to_id[n.id]:
                        new_poly_n = grid[new_poly_n_id]
                        if new_poly.polygon.buffer(0.01).intersects(new_poly_n.polygon):
                            new_poly.add_neighbor(new_poly_n)

        return grid