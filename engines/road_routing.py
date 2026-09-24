"""Build display routes along the locally available OpenStreetMap road lines."""
from functools import lru_cache
import os

import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

ROAD_FILE = "data/raw/wayanad_roads.geojson"
ROUTE_CRS = "EPSG:32643"


@lru_cache(maxsize=1)
def _road_graph():
    if not os.path.exists(ROAD_FILE):
        return None
    roads = gpd.read_file(ROAD_FILE).to_crs(ROUTE_CRS)
    graph = nx.Graph()
    for geom in roads.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords, coords[1:]):
                start, end = (round(a[0], 1), round(a[1], 1)), (round(b[0], 1), round(b[1], 1))
                length = Point(start).distance(Point(end))
                if length > 0:
                    graph.add_edge(start, end, weight=length)
    return graph if graph.number_of_edges() else None


@lru_cache(maxsize=1)
def _road_node_index():
    graph = _road_graph()
    if graph is None:
        return None
    nodes = list(graph.nodes)
    return nodes, cKDTree(nodes)


@lru_cache(maxsize=4096)
def _cached_route(source_coords, destination_coords):
    """Return a WGS84 route following the road graph, with endpoint access links.

    If local road data is unavailable or disconnected, retain the direct line so
    map rendering and simulations continue to work.
    """
    source = Point(source_coords)
    destination = Point(destination_coords)
    fallback = LineString([source, destination])
    node_index = _road_node_index()
    if node_index is None:
        return fallback

    points = gpd.GeoSeries([source, destination], crs="EPSG:4326").to_crs(ROUTE_CRS)
    projected = list(points)
    nodes, node_tree = node_index
    nearest_indexes = node_tree.query([(point.x, point.y) for point in projected])[1]
    nearest = [nodes[int(index)] for index in nearest_indexes]
    try:
        path = nx.shortest_path(_road_graph(), nearest[0], nearest[1], weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return fallback
    coords = [projected[0], *(Point(node) for node in path), projected[1]]
    return gpd.GeoSeries([LineString(coords)], crs=ROUTE_CRS).to_crs("EPSG:4326").iloc[0]


def road_route(source, destination):
    return _cached_route(
        (round(source.x, 6), round(source.y, 6)),
        (round(destination.x, 6), round(destination.y, 6)),
    )
