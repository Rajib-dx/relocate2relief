import os
import requests
import geopandas as gpd


DATA_RAW = "data/raw"
os.makedirs(DATA_RAW, exist_ok=True)

# 1. Download official SRTM 30m / 90m tile covering Wayanad (Tile: 11N, 76E)
# Source: AWS Open Data Terrain Tiles (Public, No API key needed)
print("[1/2] Fetching Real Elevation Data (USGS / AWS Open Data)...")
dem_out = os.path.join(DATA_RAW, "wayanad_real_dem.tif")

# Wayanad lies squarely inside N11E076. 
# We fetch the public SRTM GeoTIFF mirrored by the CGIAR/USGS open repository
srtm_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_53_11.zip"
zip_out = os.path.join(DATA_RAW, "srtm_53_11.zip")

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

if not os.path.exists(dem_out):
    print("Downloading SRTM elevation raster (~15 MB)...")
    resp = requests.get(srtm_url, headers=headers, stream=True, timeout=60)
    if resp.status_code == 200:
        with open(zip_out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("Extracting SRTM DEM...")
        import zipfile
        with zipfile.ZipFile(zip_out, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(".tif"):
                    zip_ref.extract(file, DATA_RAW)
                    extracted_path = os.path.join(DATA_RAW, file)
                    if os.path.exists(dem_out):
                        os.remove(dem_out)
                    os.rename(extracted_path, dem_out)
        if os.path.exists(zip_out):
            os.remove(zip_out)
        print("Real SRTM DEM successfully saved to:", dem_out)
    else:
        print(f"Fallback: Primary SRTM mirror returned {resp.status_code}. Using direct S3 elevation endpoint...")
        # Direct GeoTIFF from USGS/NASA mirrored S3 public endpoint
        s3_url = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/10/728/489.tif"
        r = requests.get(s3_url, headers=headers, stream=True)
        with open(dem_out, "wb") as f:
            f.write(r.content)
        print("S3 Elevation raster saved:", dem_out)
else:
    print("SRTM DEM already exists locally.")

# 2. Extract authentic settlement points from OpenStreetMap
print("\n[2/2] Fetching Real Wayanad Inhabited Settlements from OpenStreetMap...")
import osmnx as ox
query = "Wayanad, Kerala, India"
places = ox.features_from_place(query, tags={"place": ["town", "village", "hamlet", "suburb", "isolated_dwelling"]})
places = places[["place", "name", "geometry"]].to_crs(epsg=4326)
places_out = os.path.join(DATA_RAW, "wayanad_settlements.geojson")
places.to_file(places_out, driver="GeoJSON")
print(f"Saved {len(places)} real inhabited settlements to: {places_out}")

print("\n--- Real Data Fetch Complete ---")