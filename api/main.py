import os
import json
import numpy as np
import geopandas as gpd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pulp
from shapely.geometry import Point
from engines.road_routing import road_route

app = FastAPI(
    title="Multi-Hazard Relocation Intelligence Platform",
    description="Spatial API for Wayanad disaster risk evaluation and relocation modeling",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PROCESSED = "data/processed"
RISK_FILE = os.path.join(DATA_PROCESSED, "wayanad_risk_layer.gpkg")
SAFE_FILE = os.path.join(DATA_PROCESSED, "wayanad_safe_sites.gpkg")
FLOWS_FILE = os.path.join(DATA_PROCESSED, "wayanad_relocation_flows.geojson")
PLAN_FILE = os.path.join(DATA_PROCESSED, "relocation_plan.json")

def load_geojson_from_gpkg(filepath: str, layer: str):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Layer file {filepath} not found.")
    gdf = gpd.read_file(filepath, layer=layer)
    return json.loads(gdf.to_json())

class ScenarioParams(BaseModel):
    rain_multiplier: float = 1.0
    hazard_weight: float = 0.45
    capacity_buffer: float = 1.0

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "region": "Wayanad, Kerala"}

@app.get("/api/summary")
def get_system_summary():
    if not os.path.exists(RISK_FILE) or not os.path.exists(SAFE_FILE):
        raise HTTPException(status_code=404, detail="Processed layers not found.")
    
    risk_gdf = gpd.read_file(RISK_FILE, layer="risk_cells")
    safe_gdf = gpd.read_file(SAFE_FILE, layer="safe_sites")
    
    high_risk_mask = risk_gdf["relocation_priority"].isin(["Immediate Relocation", "Short-term Relocation"])
    displaced_pop = int(risk_gdf.loc[high_risk_mask, "population"].sum())
    
    return {
        "total_cells_analyzed": len(risk_gdf),
        "critical_red_zones": int((risk_gdf["risk_category"] == "Critical").sum()),
        "population_at_critical_risk": displaced_pop,
        "candidate_safe_sites": len(safe_gdf),
        "total_safe_capacity": int(safe_gdf["carrying_capacity"].sum()),
        "risk_breakdown": risk_gdf["risk_category"].value_counts().to_dict(),
        "relocation_breakdown": risk_gdf["relocation_priority"].value_counts().to_dict()
    }

@app.get("/api/layers/risk")
def get_risk_layer():
    return load_geojson_from_gpkg(RISK_FILE, "risk_cells")

@app.get("/api/layers/safe-sites")
def get_safe_sites_layer():
    return load_geojson_from_gpkg(SAFE_FILE, "safe_sites")

@app.get("/api/layers/relocation-flows")
def get_relocation_flows():
    if not os.path.exists(FLOWS_FILE):
        raise HTTPException(status_code=404, detail="Relocation flow layer not generated.")
    with open(FLOWS_FILE, "r") as f:
        return json.load(f)

@app.get("/api/plan")
def get_relocation_plan():
    if not os.path.exists(PLAN_FILE):
        raise HTTPException(status_code=404, detail="Plan summary not found.")
    with open(PLAN_FILE, "r") as f:
        return json.load(f)

@app.post("/api/simulate")
def run_simulation(params: ScenarioParams):
    if not os.path.exists(RISK_FILE) or not os.path.exists(SAFE_FILE):
        raise HTTPException(status_code=404, detail="Required geospatial data layers missing.")
    
    gdf = gpd.read_file(RISK_FILE, layer="risk_cells")
    safe_gdf = gpd.read_file(SAFE_FILE, layer="safe_sites")
    
    # 1. Scale rainfall against baseline
    base_min, base_max = 100.0, 550.0
    adjusted_rainfall = gdf["rainfall"] * params.rain_multiplier
    rain_norm = ((adjusted_rainfall - base_min) / (base_max - base_min) * 100.0).clip(0, 160)
    
    # 2. Dynamic slope & elevation
    slope_norm = ((gdf["slope"] - gdf["slope"].min()) / (gdf["slope"].max() - gdf["slope"].min()) * 100.0)
    elev_norm = ((gdf["elevation"] - 650.0) / 1450.0 * 100.0).clip(0, 100)
    valley_norm = 100.0 - elev_norm
    
    # 3. Dynamic Hazard & interaction penalty
    dyn_landslide = (0.50 * slope_norm + 0.35 * rain_norm + 0.15 * elev_norm).clip(0, 100)
    dyn_flood = (0.60 * valley_norm + 0.40 * rain_norm).clip(0, 100)
    
    T = 60.0
    excess_l = np.maximum(0.0, dyn_landslide - T)
    excess_f = np.maximum(0.0, dyn_flood - T)
    penalty = 15.0 * (excess_l * excess_f) / ((100.0 - T) ** 2)
    
    gdf["hazard_score"] = np.minimum(100.0, 0.55 * dyn_landslide + 0.45 * dyn_flood + penalty).round(2)
    
    # 4. Dynamic Risk re-scoring
    gdf["risk_score"] = (
        params.hazard_weight * gdf["hazard_score"] +
        0.30 * gdf["exposure_score"] +
        (1.0 - params.hazard_weight - 0.30) * gdf["vulnerability_score"]
    ).round(2)
    
    gdf["risk_category"] = [
        "Critical" if r >= 80 else "Very High" if r >= 60 else "High" if r >= 40 else "Moderate" if r >= 20 else "Low"
        for r in gdf["risk_score"]
    ]
    
    gdf["relocation_priority"] = [
        "Immediate Relocation" if (r >= 85 and p > 0)
        else "Short-term Relocation" if (r >= 70 and p > 0)
        else "Monitor" if (r >= 50 and p > 0)
        else "No Action"
        for r, p in zip(gdf["risk_score"], gdf["population"])
    ]
    
    # 5. Dynamic MILP Re-Optimization
    reloc_sources = gdf[
        gdf["relocation_priority"].isin(["Immediate Relocation", "Short-term Relocation"])
    ].copy().reset_index(drop=True)
    
    safe_destinations = safe_gdf.sort_values(by="suitability_score", ascending=False).head(20).copy().reset_index(drop=True)
    
    utm_crs = "EPSG:32643"
    src_utm = reloc_sources.to_crs(utm_crs).geometry.centroid
    dst_utm = safe_destinations.to_crs(utm_crs).geometry.centroid
    
    dist_matrix = {}
    for i, s_pt in enumerate(src_utm):
        for j, d_pt in enumerate(dst_utm):
            dist_matrix[(i, j)] = round(s_pt.distance(d_pt) / 1000.0, 2)
            
    prob = pulp.LpProblem("Dynamic_Relocation", pulp.LpMinimize)
    I = range(len(reloc_sources))
    J = range(len(safe_destinations))
    
    x = pulp.LpVariable.dicts("dyn_move", [(i, j) for i in I for j in J], lowBound=0, cat=pulp.LpInteger)
    
    prob += pulp.lpSum(
        x[(i, j)] * (dist_matrix[(i, j)] * 1.5 - safe_destinations.loc[j, "suitability_score"] * 0.5)
        for i in I for j in J
    )
    
    for i in I:
        prob += pulp.lpSum(x[(i, j)] for j in J) == int(reloc_sources.loc[i, "population"])
    for j in J:
        prob += pulp.lpSum(x[(i, j)] for i in I) <= int(safe_destinations.loc[j, "carrying_capacity"])
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Build updated dynamic vector lines
    dynamic_flows = []
    for i in I:
        for j in J:
            moved = int(x[(i, j)].varValue or 0)
            if moved > 0:
                line = road_route(reloc_sources.loc[i].geometry.centroid, safe_destinations.loc[j].geometry.centroid)
                dynamic_flows.append({
                    "geometry": line,
                    "from_cell": reloc_sources.loc[i, "cell_id"],
                    "to_site": safe_destinations.loc[j, "site_id"],
                    "people_moved": moved,
                    "distance_km": dist_matrix[(i, j)]
                })
                
    flows_gdf = gpd.GeoDataFrame(dynamic_flows, crs="EPSG:4326")
    
    critical_count = int((gdf["risk_score"] >= 80.0).sum())
    displaced_est = int(reloc_sources["population"].sum())
    
    return {
        "status": "simulation_complete",
        "critical_cells": critical_count,
        "newly_displaced_population": displaced_est,
        "average_risk": round(float(gdf["risk_score"].mean()), 2),
        "risk_geojson": json.loads(gdf.to_json()),
        "flows_geojson": json.loads(flows_gdf.to_json())
    }
@app.get("/", response_class=FileResponse)
def serve_dashboard():
    dashboard_path = os.path.join("frontend", "index.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(dashboard_path)