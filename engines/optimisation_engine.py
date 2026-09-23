import os
import json
import numpy as np
import geopandas as gpd
import pulp
from shapely.geometry import LineString
from engines.road_routing import road_route

DATA_PROCESSED = "data/processed"
risk_file = os.path.join(DATA_PROCESSED, "wayanad_risk_layer.gpkg")
safe_file = os.path.join(DATA_PROCESSED, "wayanad_safe_sites.gpkg")

print("[1/4] Loading high-risk settlements and candidate safe sites...")
risk_gdf = gpd.read_file(risk_file, layer="risk_cells")
safe_gdf = gpd.read_file(safe_file, layer="safe_sites")

# Filter settlements needing relocation (Immediate or Short-term)
reloc_sources = risk_gdf[
    risk_gdf["relocation_priority"].isin(["Immediate Relocation", "Short-term Relocation"]) &
    (risk_gdf["population"] > 0)
].copy().reset_index(drop=True)

# Select top safe sites ranked by suitability to keep optimization fast and realistic
safe_destinations = safe_gdf.sort_values(by="suitability_score", ascending=False).head(20).copy().reset_index(drop=True)

print(f"Relocation Sources: {len(reloc_sources)} communities (Total pop: {reloc_sources['population'].sum()})")
print(f"Candidate Destinations: {len(safe_destinations)} safe sites (Total capacity: {safe_destinations['carrying_capacity'].sum()})")

# Ensure total destination capacity can absorb the relocated population
total_pop = int(reloc_sources["population"].sum())
total_cap = int(safe_destinations["carrying_capacity"].sum())
if total_cap < total_pop:
    raise ValueError(f"Insufficient capacity! Pop: {total_pop}, Cap: {total_cap}")

print("[2/4] Constructing distance matrix (in kilometers)...")
utm_crs = "EPSG:32643"
src_utm = reloc_sources.to_crs(utm_crs).geometry.centroid
dst_utm = safe_destinations.to_crs(utm_crs).geometry.centroid

dist_matrix = {}
for i, s_pt in enumerate(src_utm):
    for j, d_pt in enumerate(dst_utm):
        dist_km = s_pt.distance(d_pt) / 1000.0
        dist_matrix[(i, j)] = round(dist_km, 2)

print("[3/4] Solving Linear Programming Model...")
prob = pulp.LpProblem("Wayanad_Relocation_Optimization", pulp.LpMinimize)

# Decision variables: x[i, j] = number of people moved from village i to site j
I = range(len(reloc_sources))
J = range(len(safe_destinations))
x = pulp.LpVariable.dicts("move", [(i, j) for i in I for j in J], lowBound=0, cat=pulp.LpInteger)

# Objective: Minimize cost = (distance * 1.5) - (suitability * 0.5)
prob += pulp.lpSum(
    x[(i, j)] * (dist_matrix[(i, j)] * 1.5 - safe_destinations.loc[j, "suitability_score"] * 0.5)
    for i in I for j in J
)

# Constraint 1: Everyone in each priority village must be relocated
for i in I:
    prob += pulp.lpSum(x[(i, j)] for j in J) == int(reloc_sources.loc[i, "population"])

# Constraint 2: Site capacity must not be exceeded
for j in J:
    prob += pulp.lpSum(x[(i, j)] for i in I) <= int(safe_destinations.loc[j, "carrying_capacity"])

# Solve
prob.solve(pulp.PULP_CBC_CMD(msg=False))
print(f"Optimization Status: {pulp.LpStatus[prob.status]}")

print("[4/4] Building relocation plan and flow vectors...")
relocation_flows = []
plan_summary = []

for i in I:
    for j in J:
        assigned_pop = int(x[(i, j)].varValue or 0)
        if assigned_pop > 0:
            src_geom = reloc_sources.loc[i].geometry.centroid
            dst_geom = safe_destinations.loc[j].geometry.centroid
            
            flow_line = road_route(src_geom, dst_geom)
            relocation_flows.append({
                "geometry": flow_line,
                "from_cell": reloc_sources.loc[i, "cell_id"],
                "to_site": safe_destinations.loc[j, "site_id"],
                "people_moved": assigned_pop,
                "distance_km": dist_matrix[(i, j)],
                "origin_risk": float(reloc_sources.loc[i, "risk_score"]),
                "destination_suitability": float(safe_destinations.loc[j, "suitability_score"])
            })
            
            plan_summary.append({
                "source_id": reloc_sources.loc[i, "cell_id"],
                "target_site": safe_destinations.loc[j, "site_id"],
                "population_moved": assigned_pop,
                "distance_km": dist_matrix[(i, j)]
            })

# Save output layers
flows_gdf = gpd.GeoDataFrame(relocation_flows, crs="EPSG:4326")
flows_output = os.path.join(DATA_PROCESSED, "wayanad_relocation_flows.geojson")
flows_gdf.to_file(flows_output, driver="GeoJSON")

plan_json = os.path.join(DATA_PROCESSED, "relocation_plan.json")
with open(plan_json, "w") as f:
    json.dump(plan_summary, f, indent=2)

print(f"\n--- Optimization Complete ---")
print(f"Total movement paths generated: {len(relocation_flows)}")
print(f"Flow vectors saved to: {flows_output}")
print(f"Plan summary saved to: {plan_json}")
print(flows_gdf.columns)