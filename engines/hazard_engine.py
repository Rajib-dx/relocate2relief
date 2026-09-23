import os
import numpy as np
import geopandas as gpd

DATA_PROCESSED = "data/processed"
input_path = os.path.join(DATA_PROCESSED, "wayanad_spatial_grid.gpkg")

print("[1/3] Loading spatial base grid...")
gdf = gpd.read_file(input_path, layer="cells")

def min_max_normalize(series):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.zeros(len(series))
    return 100.0 * (series - s_min) / (s_max - s_min)

print("[2/3] Computing individual hazard components...")
# Normalize inputs to 0-100
slope_norm = min_max_normalize(gdf["slope"])
rain_norm = min_max_normalize(gdf["rainfall"])
elev_norm = min_max_normalize(gdf["elevation"])

# Inverse elevation for flood (valleys accumulate water)
valley_norm = 100.0 - elev_norm

# 1. Landslide Susceptibility (High slope + heavy rain + high relief)
gdf["landslide_score"] = (0.50 * slope_norm + 0.35 * rain_norm + 0.15 * elev_norm).round(2)

# 2. Flash Flood Susceptibility (Valley/low elevation + heavy rain)
gdf["flood_score"] = (0.60 * valley_norm + 0.40 * rain_norm).round(2)

print("[3/3] Calculating Compound Hazard Score with Interaction Penalty...")
# Base weighted hazard
w_l, w_f = 0.55, 0.45
gdf["weighted_hazard"] = w_l * gdf["landslide_score"] + w_f * gdf["flood_score"]

# Compound interaction penalty: active when both landslide & flood > threshold T
T = 60.0
alpha = 15.0  # Max penalty points

excess_l = np.maximum(0.0, gdf["landslide_score"] - T)
excess_f = np.maximum(0.0, gdf["flood_score"] - T)
penalty = alpha * (excess_l * excess_f) / ((100.0 - T) ** 2)

gdf["hazard_score"] = np.minimum(100.0, gdf["weighted_hazard"] + penalty).round(2)

# Save updated dataset
output_path = os.path.join(DATA_PROCESSED, "wayanad_hazard_layer.gpkg")
gdf.to_file(output_path, layer="hazard_cells", driver="GPKG")

print(f"Hazard Engine complete. Summary Statistics:")
print(f"Saved layer to: {output_path}")
