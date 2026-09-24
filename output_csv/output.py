"""Export processed relocation engine outputs to CSV files."""
from pathlib import Path
import json

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each spatial export keeps its full attributes and stores geometry as WKT.
spatial_outputs = {
    "wayanad_spatial_grid.gpkg": ("cells", "spatial_grid_cells.csv"),
    "wayanad_hazard_layer.gpkg": ("hazard_cells", "hazard_cells.csv"),
    "wayanad_risk_layer.gpkg": ("risk_cells", "risk_cells.csv"),
    "wayanad_safe_sites.gpkg": ("safe_sites", "safe_sites.csv"),
    "wayanad_relocation_flows.geojson": (None, "relocation_flows.csv"),
}

for filename, (layer, output_name) in spatial_outputs.items():
    source = PROCESSED_DIR / filename
    frame = gpd.read_file(source, layer=layer) if layer else gpd.read_file(source)
    frame["geometry_wkt"] = frame.geometry.to_wkt()
    frame = frame.drop(columns="geometry")
    destination = OUTPUT_DIR / output_name
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} records to {destination.relative_to(PROJECT_ROOT)}")

plan_path = PROCESSED_DIR / "relocation_plan.json"
plan = json.loads(plan_path.read_text())
plan_frame = pd.DataFrame(plan)
plan_csv = OUTPUT_DIR / "relocation_plan.csv"
plan_frame.to_csv(plan_csv, index=False)
print(f"Wrote {len(plan_frame)} records to {plan_csv.relative_to(PROJECT_ROOT)}")
