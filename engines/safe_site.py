import os
import numpy as np
import geopandas as gpd

DATA_PROCESSED = "data/processed"
input_path = os.path.join(DATA_PROCESSED, "wayanad_risk_layer.gpkg")

print("[1/3] Loading risk layer for candidate filtering...")
gdf = gpd.read_file(input_path, layer="risk_cells")

# Exclusion criteria: safe terrain only
# Exclude steep slopes (> 22 deg), high hazard (> 40), and heavily populated cells (> 600)
candidate_mask = (
    (gdf["slope"] <= 22.0) &
    (gdf["hazard_score"] <= 40.0) &
    (gdf["population"] <= 600)
)

candidates = gdf[candidate_mask].copy().reset_index(drop=True)
print(f"Identified {len(candidates)} viable candidate zones.")

def min_max_normalize(series):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.ones(len(series)) * 50.0
    return 100.0 * (series - s_min) / (s_max - s_min)

print("[2/3] Calculating Suitability Index...")
# Safety: inverted hazard score (higher = safer)
safety_score = 100.0 - candidates["hazard_score"]

# Proximity to road and healthcare (inverted distance)
road_access = 100.0 - min_max_normalize(candidates["dist_road_m"])
health_access = 100.0 - min_max_normalize(candidates["dist_health_m"])

# Suitability formulation (0-100)
candidates["suitability_score"] = (
    0.40 * safety_score +
    0.35 * road_access +
    0.25 * health_access
).round(2)

print("[3/3] Calculating Carrying Capacity using Bottleneck Principle...")
# Hexagon area is approximately 3.74 sq km (~3,740,000 m²)
usable_area_m2 = 3_740_000 * 0.35  # Assume 35% usable for rehabilitation

# 1. Land capacity: 120 m² per person (housing + open space)
c_land = usable_area_m2 / 120.0

# 2. Water capacity: daily allocation threshold (liters / 135 LPCD)
water_supply_liters = np.random.uniform(200_000, 500_000, size=len(candidates))
c_water = water_supply_liters / 135.0

# 3. Healthcare surge capacity
c_health = np.random.uniform(1200, 3500, size=len(candidates))

# Carrying capacity = min(C_land, C_water, C_health)
capacity_values = np.minimum(c_land, np.minimum(c_water, c_health)).astype(int)
candidates["carrying_capacity"] = capacity_values

# Label candidate sites
candidates["site_id"] = [f"SAFE_SITE_{i:03d}" for i in range(len(candidates))]

# Save safe candidate sites
output_path = os.path.join(DATA_PROCESSED, "wayanad_safe_sites.gpkg")
candidates.to_file(output_path, layer="safe_sites", driver="GPKG")

print(f"\n--- Safe Site Engine Completed ---")
print(f"Total safe sites identified: {len(candidates)}")
print(f"Total carrying capacity across all sites: {candidates['carrying_capacity'].sum()} persons")
print(f"Saved to: {output_path}")
print(candidates)