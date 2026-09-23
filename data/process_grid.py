import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
 
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
os.makedirs(DATA_PROCESSED, exist_ok=True)

print("[1/4] Loading boundary and infrastructure...")
boundary = gpd.read_file(os.path.join(DATA_RAW, "wayanad_boundary.geojson"))
roads = gpd.read_file(os.path.join(DATA_RAW, "wayanad_roads.geojson"))
health = gpd.read_file(os.path.join(DATA_RAW, "wayanad_health.geojson"))

# Transform to projected CRS UTM 43N (EPSG:32643) for accurate meter calculations
utm_crs = "EPSG:32643"
boundary_utm = boundary.to_crs(utm_crs)
roads_utm = roads.to_crs(utm_crs)
health_utm = health.to_crs(utm_crs)

print("[2/4] Generating regular hexagonal grid over Wayanad...")
# Generate regular hexagon grid (radius ~ 1.2 km for fast processing)
bounds = boundary_utm.total_bounds
xmin, ymin, xmax, ymax = bounds
hex_radius = 1200  # meters
w = hex_radius * 2
h = np.sqrt(3) * hex_radius

cols = int((xmax - xmin) / (w * 0.75)) + 1
rows = int((ymax - ymin) / h) + 1

hexagons = []
for c in range(cols):
    x_offset = xmin + c * (w * 0.75)
    y_shift = (h / 2) if (c % 2 == 1) else 0
    for r in range(rows):
        y_offset = ymin + r * h + y_shift
        # Define 6 vertices of a hexagon
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        hx = x_offset + hex_radius * np.cos(angles)
        hy = y_offset + hex_radius * np.sin(angles)
        hexagons.append(Polygon(zip(hx, hy)))

grid = gpd.GeoDataFrame({"geometry": hexagons}, crs=utm_crs)

# Clip to Wayanad boundary
grid = gpd.clip(grid, boundary_utm).reset_index(drop=True)
grid["cell_id"] = [f"WAY_HEX_{i:04d}" for i in range(len(grid))]
print(f"Generated {len(grid)} spatial analysis cells.")

print("[3/4] Synthesizing elevation, slope, and rainfall realistic to Wayanad...")
np.random.seed(42)
centroids = grid.geometry.centroid

# Wayanad topography: higher mountains in West/South-West (Meppadi/Vellarimala), lower valleys North-East
norm_x = (centroids.x - bounds[0]) / (bounds[2] - bounds[0])
norm_y = (centroids.y - bounds[1]) / (bounds[3] - bounds[1])

# Elevation between 650m and 2100m
grid["elevation"] = 700 + (1 - norm_x) * 1100 + (1 - norm_y) * 300 + np.random.normal(0, 40, len(grid))
grid["elevation"] = grid["elevation"].clip(650, 2100).round(1)

# Slope: Steeper in high-elevation ridges (0 to 55 degrees)
grid["slope"] = ((2100 - grid["elevation"]) / 1450) * 10 + (1 - norm_x) * 35 + np.random.uniform(2, 12, len(grid))
grid["slope"] = grid["slope"].clip(2, 55).round(1)

# Monsoon Rainfall (mm/event, 100mm to 550mm)
grid["rainfall"] = 150 + (1 - norm_x) * 280 + np.random.uniform(0, 80, len(grid))
grid["rainfall"] = grid["rainfall"].clip(100, 550).round(1)

# Settlement Demographics
is_settlement = np.random.choice([1, 0], size=len(grid), p=[0.75, 0.25])
grid["population"] = np.where(is_settlement == 1, np.random.randint(120, 1500, size=len(grid)), 0)
grid["poverty_rate"] = np.random.uniform(0.10, 0.55, size=len(grid)).round(2)
grid["vulnerable_pop_ratio"] = np.random.uniform(0.15, 0.40, size=len(grid)).round(2) # Children + elderly

print("[4/4] Calculating Euclidean distance to roads & health centers...")
# Road distance (meters)
roads_union = roads_utm.union_all()
grid["dist_road_m"] = centroids.distance(roads_union).round(1)

# Healthcare distance (meters)
health_union = health_utm.union_all()
grid["dist_health_m"] = centroids.distance(health_union).round(1)

# Save processed geodataframe back to EPSG:4326 GeoPackage
grid_4326 = grid.to_crs(epsg=4326)
output_path = os.path.join(DATA_PROCESSED, "wayanad_spatial_grid.gpkg")
grid_4326.to_file(output_path, layer="cells", driver="GPKG")
print(f"Spatial Grid successfully built and saved to: {output_path}")