#!/usr/bin/env python3
"""
walkshed_analysis.py
====================
Production pipeline script for Transit Station Heat Exposure Analysis.

Workflow:
1. Load transit station locations for NYC (MTA), Chicago (CTA), and Washington, D.C. (WMATA).
2. Retrieve walkable street networks via OpenStreetMap (OSMnx) in local UTM projections.
3. Compute 10-minute pedestrian network walkshed isochrones (750m network reach at 1.25 m/s).
4. Extract zonal surface temperature percentiles (median, 10th, 25th, 75th, 90th, 99th) from
   stitched Landsat 8/9 LST rasters.
5. Export standardized GeoJSON files in EPSG:4326 (consolidated and per-city).
6. Generate an interactive Folium web map with custom SVG boxplots (station_temperature_map.html).
7. Generate a 3-panel publication figure (Eos_station_walkshed_temperature_plots.png).

Usage:
    python walkshed_analysis.py [OPTIONS]

Examples:
    # Full run for all cities
    python walkshed_analysis.py

    # Process only Washington, D.C.
    python walkshed_analysis.py --cities DC

    # Skip walkshed routing and re-extract temperature statistics / maps using existing walksheds
    python walkshed_analysis.py --skip-walksheds
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure stdout and stderr handle encodings cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Optional interactive mapping and network routing libraries
try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    ox = None
    nx = None

try:
    import folium
    from folium import plugins
    from branca.colormap import LinearColormap
except ImportError:
    folium = None

# Suppress minor projection / GDAL runtime warnings
warnings.filterwarnings("ignore")

# Default study configuration
DEFAULT_CITIES = ["Chicago", "DC", "NYC"]

DEFAULT_STATION_FILES = {
    "Chicago": "metrolocs/Chicago.geojson",
    "DC": "metrolocs/DC.geojson",
    "NYC": "metrolocs/NYC.geojson"
}

DEFAULT_TEMP_FILES = {
    "Chicago": "stitched_city_temperatures/Chicago_stitched_surface_temperature.tif",
    "DC": "stitched_city_temperatures/DC_stitched_surface_temperature.tif",
    "NYC": "stitched_city_temperatures/NYC_stitched_surface_temperature.tif"
}

CITY_CENTERS = {
    "Chicago": [41.8781, -87.6298],
    "DC": [38.9072, -77.0369],
    "NYC": [40.7128, -74.0060]
}


def configure_osmnx(cache_folder="cache"):
    """Configure OSMnx caching and logging parameters."""
    if ox is None:
        raise ImportError("osmnx is required for walkshed routing. Install via `pip install osmnx`.")
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_folder
    ox.settings.log_console = False


def load_station_locations(cities=None, station_files=None):
    """
    Load station point layers from GeoJSON files and standardize them.

    Parameters:
        cities (list): List of cities to load (default: ['Chicago', 'DC', 'NYC']).
        station_files (dict): Map of city name to GeoJSON file path.

    Returns:
        dict: City names mapped to GeoDataFrames in EPSG:4326.
    """
    if cities is None:
        cities = DEFAULT_CITIES
    if station_files is None:
        station_files = DEFAULT_STATION_FILES

    loaded_stations = {}
    print("\n--- Loading Transit Station Datasets ---")
    for city in cities:
        path = station_files.get(city)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Station file for {city} not found at: {path}")
        
        gdf = gpd.read_file(path)
        # Ensure CRS is WGS84 for standardization
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)

        loaded_stations[city] = gdf
        print(f"  [OK] {city}: {len(gdf)} stations loaded from {path}")

    total = sum(len(g) for g in loaded_stations.values())
    print(f"Total stations across selected cities: {total}")
    return loaded_stations


def calculate_walksheds_for_city(stations_4326, walk_speed_mps=1.25, trip_time_seconds=600, buffer_deg=0.01):
    """
    Calculate 10-minute pedestrian walksheds for a single city.

    Parameters:
        stations_4326 (GeoDataFrame): Station points in EPSG:4326.
        walk_speed_mps (float): Walking speed in m/s (default: 1.25).
        trip_time_seconds (int): Maximum walk time in seconds (default: 600).
        buffer_deg (float): Geographic buffer around station envelope for network query.

    Returns:
        walksheds_utm (GeoDataFrame): Isochrone polygons in local UTM.
        stations_utm (GeoDataFrame): Stations in local UTM.
        utm_crs: Projected CRS object.
    """
    if ox is None or nx is None:
        raise ImportError("osmnx and networkx are required to compute network walksheds.")

    # Determine local UTM projection for distance-accurate routing
    utm_crs = stations_4326.estimate_utm_crs()

    # Create buffered polygon encompassing all stations in city
    hull_geom = stations_4326.unary_union.convex_hull
    if isinstance(hull_geom, Point):
        polygon_4326 = hull_geom.buffer(buffer_deg)
    else:
        polygon_4326 = hull_geom.buffer(buffer_deg)

    # Fetch walkable network from OpenStreetMap
    print("    Fetching walkable network from OpenStreetMap...")
    G = ox.graph_from_polygon(polygon_4326, network_type="walk")

    # Project graph and stations to local UTM
    G_proj = ox.project_graph(G, to_crs=utm_crs)
    stations_utm = stations_4326.to_crs(utm_crs)

    # Assign travel impedance in seconds to each edge
    for _, _, _, data in G_proj.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / walk_speed_mps

    # Calculate ego-graph isochrones
    walksheds = []
    for _, station in tqdm(stations_utm.iterrows(), total=len(stations_utm), desc="    Routing walksheds", leave=False):
        center_node = ox.nearest_nodes(G_proj, station.geometry.x, station.geometry.y)
        subgraph = nx.ego_graph(G_proj, center_node, radius=trip_time_seconds, distance="travel_time")
        
        node_points = [Point((data["x"], data["y"])) for _, data in subgraph.nodes(data=True)]
        if len(node_points) >= 3:
            isochrone = gpd.GeoSeries(node_points).unary_union.convex_hull
        elif len(node_points) > 0:
            # Fallback buffer around node points if fewer than 3 nodes
            isochrone = gpd.GeoSeries(node_points).unary_union.buffer(50)
        else:
            # Fallback buffer around station point
            isochrone = station.geometry.buffer(50)
            
        walksheds.append(isochrone)

    walksheds_utm = gpd.GeoDataFrame(geometry=walksheds, crs=utm_crs)
    return walksheds_utm, stations_utm, utm_crs


def generate_all_walksheds(loaded_stations, walk_speed_mps=1.25, trip_time_seconds=600):
    """
    Run walkshed routing across all loaded cities and structure the consolidated dataset.

    Returns:
        all_walksheds_gdf (GeoDataFrame): Unified walkshed polygons in EPSG:4326.
        city_walksheds_utm (dict): Per-city walksheds in UTM.
        city_stations_utm (dict): Per-city stations in UTM.
        city_utm_crs (dict): Per-city UTM CRS objects.
    """
    city_walksheds_utm = {}
    city_stations_utm = {}
    city_utm_crs = {}
    all_records = []

    print("\n--- Delineating 10-Minute Network Walksheds ---")
    for city, stations_gdf in loaded_stations.items():
        print(f"\nProcessing {city} ({len(stations_gdf)} stations)...")
        walksheds_utm, stations_utm, utm_crs = calculate_walksheds_for_city(
            stations_gdf, walk_speed_mps=walk_speed_mps, trip_time_seconds=trip_time_seconds
        )
        city_walksheds_utm[city] = walksheds_utm
        city_stations_utm[city] = stations_utm
        city_utm_crs[city] = utm_crs

        # Standardize records to EPSG:4326 for output export
        walksheds_4326 = walksheds_utm.to_crs(epsg=4326)
        stations_4326 = stations_utm.to_crs(epsg=4326)

        for i in range(len(walksheds_4326)):
            w_geom = walksheds_4326.geometry.iloc[i]
            s_geom = stations_4326.geometry.iloc[i]
            orig_row = stations_gdf.iloc[i]

            station_name = f"{city}_Station_{i+1}"
            for col in orig_row.index:
                if col != "geometry" and any(k in col.lower() for k in ["name", "station"]):
                    val = str(orig_row[col]).strip()
                    if val and val.lower() != "nan":
                        station_name = val
                        break

            rec = {
                "city": city,
                "station_id": i + 1,
                "station_name": station_name,
                "geometry": w_geom,
                "station_lon": s_geom.x,
                "station_lat": s_geom.y,
                "original_crs": str(utm_crs),
                "output_crs": "EPSG:4326",
                "walk_time_minutes": int(trip_time_seconds / 60),
                "walk_speed_mps": walk_speed_mps,
                "area_km2": walksheds_utm.geometry.iloc[i].area / 1e6
            }

            # Retain original attribute columns
            for col in orig_row.index:
                if col != "geometry" and col not in rec:
                    rec[col] = orig_row[col]

            all_records.append(rec)

    all_walksheds_gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
    return all_walksheds_gdf, city_walksheds_utm, city_stations_utm, city_utm_crs


def export_walksheds(all_walksheds_gdf, output_path="all_walksheds_10min.geojson"):
    """Export consolidated walksheds and per-city GeoJSON layers."""
    print(f"\n--- Exporting Walkshed Datasets ---")
    all_walksheds_gdf.to_file(output_path, driver="GeoJSON")
    print(f"  [OK] Saved consolidated walksheds: {output_path} ({len(all_walksheds_gdf)} features)")

    for city in all_walksheds_gdf["city"].unique():
        city_sub = all_walksheds_gdf[all_walksheds_gdf["city"] == city]
        city_file = f"walksheds_10min_{city.lower()}.geojson"
        city_sub.to_file(city_file, driver="GeoJSON")
        print(f"  [OK] Saved {city} walksheds: {city_file} ({len(city_sub)} features)")


def extract_zonal_temperature_stats(walkshed_geom, walkshed_crs, raster_path):
    """
    Clip thermal GeoTIFF to walkshed polygon and calculate percentile statistics.

    Returns:
        dict: median, 10th, 25th, 75th, 90th, 99th percentile surface temperatures in °C.
    """
    empty_stats = {
        "median_temp": np.nan,
        "temp_10th": np.nan,
        "temp_25th": np.nan,
        "temp_75th": np.nan,
        "temp_90th": np.nan,
        "temp_99th": np.nan
    }

    if not os.path.exists(raster_path):
        return empty_stats

    try:
        with rasterio.open(raster_path) as src:
            walkshed_gdf = gpd.GeoDataFrame([1], geometry=[walkshed_geom], crs=walkshed_crs)
            walkshed_raster_crs = walkshed_gdf.to_crs(src.crs)

            clipped_data, _ = mask(src, walkshed_raster_crs.geometry, crop=True, nodata=np.nan)
            values = clipped_data[0].flatten()
            valid_values = values[~np.isnan(values)]

            if len(valid_values) == 0:
                return empty_stats

            return {
                "median_temp": float(np.median(valid_values)),
                "temp_10th": float(np.percentile(valid_values, 10)),
                "temp_25th": float(np.percentile(valid_values, 25)),
                "temp_75th": float(np.percentile(valid_values, 75)),
                "temp_90th": float(np.percentile(valid_values, 90)),
                "temp_99th": float(np.percentile(valid_values, 99))
            }
    except Exception as e:
        return empty_stats


def compute_station_temperature_statistics(loaded_stations, all_walksheds_gdf, temp_files=None):
    """
    Compute zonal surface temperature statistics for all stations.

    Returns:
        station_stats_gdf (GeoDataFrame): Station point layer in EPSG:4326 with temperature statistics.
    """
    if temp_files is None:
        temp_files = DEFAULT_TEMP_FILES

    print("\n--- Extracting Thermal Zonal Statistics ---")
    station_records = []

    for city in loaded_stations.keys():
        print(f"\nExtracting LST statistics for {city}...")
        raster_path = temp_files.get(city)
        if not raster_path or not os.path.exists(raster_path):
            print(f"  [WARN] LST raster not found at {raster_path}. Stats will be NaN.")

        city_walksheds = all_walksheds_gdf[all_walksheds_gdf["city"] == city]
        city_stations = loaded_stations[city]

        for i in tqdm(range(len(city_walksheds)), desc=f"  Zonal stats ({city})", leave=False):
            w_row = city_walksheds.iloc[i]
            s_row = city_stations.iloc[i]

            walkshed_geom = w_row.geometry
            # Extract statistics using EPSG:4326 walkshed polygon (reprojected inside mask function)
            stats = extract_zonal_temperature_stats(walkshed_geom, "EPSG:4326", raster_path)

            station_point = s_row.geometry
            record = {
                "city": city,
                "station_name": w_row.get("station_name", f"{city}_Station_{i+1}"),
                "geometry": station_point,
                **stats
            }

            # Retain original station attributes
            for col in s_row.index:
                if col != "geometry" and col not in record:
                    record[col] = s_row[col]

            station_records.append(record)

    station_stats_gdf = gpd.GeoDataFrame(station_records, crs="EPSG:4326")
    return station_stats_gdf


def export_temperature_statistics(station_stats_gdf, output_path="station_temperature_statistics.geojson"):
    """Export consolidated and per-city station temperature statistics GeoJSON files."""
    print(f"\n--- Exporting Station Temperature Statistics ---")
    station_stats_gdf.to_file(output_path, driver="GeoJSON")
    print(f"  [OK] Saved consolidated statistics: {output_path} ({len(station_stats_gdf)} stations)")

    for city in station_stats_gdf["city"].unique():
        city_sub = station_stats_gdf[station_stats_gdf["city"] == city]
        city_file = f"station_temperature_statistics_{city.lower()}.geojson"
        city_sub.to_file(city_file, driver="GeoJSON")
        print(f"  [OK] Saved {city} statistics: {city_file} ({len(city_sub)} stations)")


def create_boxplot_svg(station):
    """Generate inline SVG boxplot string for Folium popups."""
    temp_10th = station.get("temp_10th", np.nan)
    temp_25th = station.get("temp_25th", np.nan)
    median = station.get("median_temp", np.nan)
    temp_75th = station.get("temp_75th", np.nan)
    temp_90th = station.get("temp_90th", np.nan)

    if any(np.isnan([temp_10th, temp_25th, median, temp_75th, temp_90th])):
        return "<p style='color:#777;font-style:italic;'>Temperature data unavailable</p>"

    svg_width = 280
    svg_height = 80
    plot_left = 40
    plot_right = 240
    plot_width = plot_right - plot_left
    center_y = svg_height // 2

    temp_range = temp_90th - temp_10th
    if temp_range > 0:
        def temp_to_x(val):
            return plot_left + ((val - temp_10th) / temp_range) * plot_width
    else:
        def temp_to_x(val):
            return (plot_left + plot_right) // 2

    x_10th = temp_to_x(temp_10th)
    x_25th = temp_to_x(temp_25th)
    x_median = temp_to_x(median)
    x_75th = temp_to_x(temp_75th)
    x_90th = temp_to_x(temp_90th)

    return f"""
    <svg width="{svg_width}" height="{svg_height}" style="background: white;">
        <!-- Left whisker -->
        <line x1="{x_10th}" y1="{center_y}" x2="{x_25th}" y2="{center_y}" stroke="#E74C3C" stroke-width="2"/>
        <line x1="{x_10th}" y1="{center_y-8}" x2="{x_10th}" y2="{center_y+8}" stroke="#E74C3C" stroke-width="2"/>

        <!-- IQR Box -->
        <rect x="{x_25th}" y="{center_y-12}" width="{max(1, x_75th-x_25th)}" height="24"
              fill="#E74C3C" stroke="#E74C3C" stroke-width="1" opacity="0.8"/>

        <!-- Right whisker -->
        <line x1="{x_75th}" y1="{center_y}" x2="{x_90th}" y2="{center_y}" stroke="#E74C3C" stroke-width="2"/>
        <line x1="{x_90th}" y1="{center_y-8}" x2="{x_90th}" y2="{center_y+8}" stroke="#E74C3C" stroke-width="2"/>

        <!-- Median marker line -->
        <line x1="{x_median}" y1="{center_y-12}" x2="{x_median}" y2="{center_y+12}" stroke="white" stroke-width="3"/>

        <!-- Numeric value labels -->
        <text x="{x_10th}" y="{center_y+25}" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">{temp_10th:.1f}°C</text>
        <text x="{x_25th}" y="{center_y-18}" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">{temp_25th:.1f}°C</text>
        <text x="{x_median}" y="{center_y+25}" text-anchor="middle" font-family="Arial" font-size="11" fill="#333" font-weight="bold">{median:.1f}°C</text>
        <text x="{x_75th}" y="{center_y-18}" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">{temp_75th:.1f}°C</text>
        <text x="{x_90th}" y="{center_y+25}" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">{temp_90th:.1f}°C</text>
    </svg>
    """


def generate_interactive_map(station_stats_gdf, output_html="station_temperature_map.html"):
    """
    Generate an interactive Folium map with temperature distributions, layer controls, and city jump menus.
    """
    if folium is None:
        print("  [WARN] Folium not installed. Skipping interactive map generation.")
        return

    print("\n--- Generating Interactive Folium Map ---")
    gdf_4326 = station_stats_gdf.to_crs(epsg=4326)

    # Initialize map centered on Washington, D.C.
    m = folium.Map(
        location=[38.9072, -77.0369],
        zoom_start=10,
        tiles="cartodb positron"
    )

    # Define color ramp
    valid_medians = gdf_4326["median_temp"].dropna()
    vmin = valid_medians.min() if len(valid_medians) > 0 else 20
    vmax = valid_medians.max() if len(valid_medians) > 0 else 50

    colormap = LinearColormap(
        colors=["#000080", "#0066CC", "#00CCFF", "#00FF99", "#66FF66", "#CCFF00", "#FFCC00", "#FF6600", "#FF0000", "#CC0000", "#990000"],
        vmin=vmin,
        vmax=vmax,
        caption="Median Walkshed Surface Temperature (deg C)"
    )

    # Base layers
    folium.TileLayer("CartoDB Positron", name="Light Positron").add_to(m)
    folium.TileLayer("CartoDB Dark_Matter", name="Dark Matter").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Add markers per city feature group
    for city in gdf_4326["city"].unique():
        group = folium.FeatureGroup(name=f"{city} Transit Stations")
        city_sub = gdf_4326[gdf_4326["city"] == city]

        for _, station in city_sub.iterrows():
            boxplot_html = create_boxplot_svg(station)
            med_str = f"{station['median_temp']:.1f}°C" if not np.isnan(station['median_temp']) else "N/A"
            r10_str = f"{station['temp_10th']:.1f}°C" if not np.isnan(station['temp_10th']) else "N/A"
            r90_str = f"{station['temp_90th']:.1f}°C" if not np.isnan(station['temp_90th']) else "N/A"
            r99_str = f"{station['temp_99th']:.1f}°C" if not np.isnan(station['temp_99th']) else "N/A"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 300px;">
                <h4 style="margin: 5px 0; color: #2C3E50;">{station['station_name']}</h4>
                <p style="margin: 3px 0; color: #555;"><strong>City:</strong> {station['city']}</p>
                <h5 style="margin: 12px 0 8px 0; color: #2C3E50;">Walkshed Thermal Exposure</h5>
                {boxplot_html}
                <p style="margin: 10px 0 5px 0; font-size: 11px; color: #555; text-align: center;">
                    <strong>10-90% Range:</strong> {r10_str} - {r90_str} | <strong>99th %:</strong> {r99_str}
                </p>
            </div>
            """

            fill_color = colormap(station["median_temp"]) if not np.isnan(station["median_temp"]) else "#888888"

            folium.CircleMarker(
                location=[station.geometry.y, station.geometry.x],
                radius=4,
                popup=folium.Popup(popup_html, max_width=320),
                color="black",
                weight=0.8,
                fillColor=fill_color,
                fillOpacity=0.85,
                tooltip=f"{station['station_name']} ({city}) -- Median: {med_str}"
            ).add_to(group)

        group.add_to(m)

    colormap.add_to(m)

    # City zoom dropdown control
    dropdown_html = """
    <div style="position: fixed; top: 12px; left: 75px; width: 170px; z-index:9999;
                background-color: white; border: 1px solid #ccc; border-radius: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px; padding: 6px 8px;">
        <label for="city-select" style="font-weight: bold; color: #333; display:block; margin-bottom:4px;">Zoom to Metro:</label>
        <select id="city-select" onchange="zoomToCity()" style="width: 100%; padding: 3px; font-size: 12px;">
            <option value="">-- Select City --</option>
            <option value="chicago">Chicago (CTA)</option>
            <option value="dc">Washington, D.C. (WMATA)</option>
            <option value="nyc">New York City (MTA)</option>
        </select>
    </div>
    <script>
    function zoomToCity() {
        var sel = document.getElementById('city-select');
        var val = sel.value;
        var coords = {
            'chicago': [41.8781, -87.6298],
            'dc': [38.9072, -77.0369],
            'nyc': [40.7128, -74.0060]
        };
        if (val && coords[val]) {
            var mapObj = window[Object.keys(window).find(k => k.startsWith('map_'))];
            if (mapObj) {
                mapObj.setView(coords[val], 12);
            }
        }
    }
    </script>
    """
    m.get_root().html.add_child(folium.Element(dropdown_html))
    folium.LayerControl().add_to(m)

    m.save(output_html)
    print(f"  [OK] Saved interactive Folium map: {output_html}")


def generate_sample_figure(loaded_stations, all_walksheds_gdf, temp_files=None, output_png="Eos_station_walkshed_temperature_plots.png"):
    """
    Generate the 1x3 sample station figure comparing street network, clipped temperature,
    walkshed boundary, and station point across NYC, DC, and Chicago.
    """
    if ox is None:
        print("  [WARN] osmnx not installed. Skipping sample comparison figure.")
        return

    if temp_files is None:
        temp_files = DEFAULT_TEMP_FILES

    print("\n--- Generating Publication Comparison Figure ---")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    cities_to_plot = ["NYC", "DC", "Chicago"]
    im = None

    for i, city in enumerate(cities_to_plot):
        ax = axes[i]
        if city not in loaded_stations:
            ax.set_visible(False)
            continue

        city_w = all_walksheds_gdf[all_walksheds_gdf["city"] == city]
        city_s = loaded_stations[city]

        # Select representative sample station (index 21 or first)
        sample_idx = min(21, len(city_w) - 1)
        w_row = city_w.iloc[sample_idx]
        s_row = city_s.iloc[sample_idx]

        station_geom = s_row.geometry
        walkshed_geom = w_row.geometry
        station_name = w_row.get("station_name", f"{city} Station")

        # Project to local UTM for accurate network and raster rendering
        stations_gdf_tmp = gpd.GeoDataFrame(geometry=[station_geom], crs="EPSG:4326")
        utm_crs = stations_gdf_tmp.estimate_utm_crs()
        station_utm = stations_gdf_tmp.to_crs(utm_crs).geometry.iloc[0]
        walkshed_utm = gpd.GeoDataFrame(geometry=[walkshed_geom], crs="EPSG:4326").to_crs(utm_crs).geometry.iloc[0]

        # 1. Fetch street network around station (1.2 km radius)
        try:
            G = ox.graph_from_point((station_geom.y, station_geom.x), dist=1200, network_type="walk")
            G_proj = ox.project_graph(G, to_crs=utm_crs)
            edges = ox.graph_to_gdfs(G_proj, nodes=False)
            edges.plot(ax=ax, color="#888888", linewidth=0.5, alpha=0.5, zorder=1)
        except Exception as e:
            print(f"    Notice: Could not fetch graph for {city} sample station: {e}")

        # 2. Clip and plot temperature raster within walkshed
        raster_path = temp_files.get(city)
        if raster_path and os.path.exists(raster_path):
            try:
                with rasterio.open(raster_path) as src:
                    walkshed_gdf_tmp = gpd.GeoDataFrame(geometry=[walkshed_utm], crs=utm_crs)
                    walkshed_raster_crs = walkshed_gdf_tmp.to_crs(src.crs)

                    out_image, out_transform = mask(src, walkshed_raster_crs.geometry, crop=True, nodata=np.nan)

                    # Reproject clipped raster to UTM
                    dst_crs = utm_crs
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, out_image.shape[2], out_image.shape[1],
                        left=out_transform.c,
                        bottom=out_transform.f + out_transform.e * out_image.shape[1],
                        right=out_transform.c + out_transform.a * out_image.shape[2],
                        top=out_transform.f
                    )

                    destination = np.zeros((1, height, width), dtype=np.float32)
                    reproject(
                        source=out_image,
                        destination=destination,
                        src_transform=out_transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )

                    destination[0] = np.where(destination[0] == 0, np.nan, destination[0])
                    im = ax.imshow(
                        destination[0],
                        extent=[transform.c, transform.c + transform.a * width, transform.f + transform.e * height, transform.f],
                        cmap="YlOrRd",
                        alpha=0.75,
                        zorder=2,
                        vmin=30,
                        vmax=45
                    )
            except Exception as e:
                print(f"    Notice: Could not clip raster for {city}: {e}")
                gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="#3498DB", alpha=0.35, zorder=2)
        else:
            gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="#3498DB", alpha=0.35, zorder=2)

        # 3. Walkshed boundary outline
        gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5, zorder=3)

        # 4. Station point marker
        ax.scatter(station_utm.x, station_utm.y, color="cyan", edgecolor="black", s=90, linewidth=1.0, zorder=4, label="Station")

        ax.set_title(f"{city}\n{station_name}", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    # Add shared colorbar if raster image was plotted
    if im is not None:
        cbar_ax = fig.add_axes([0.33, -0.04, 0.34, 0.05])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Surface Temperature (°C)", fontsize=10)
        cbar.ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved publication comparison figure: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 10-minute pedestrian walksheds and extract zonal surface temperature statistics across transit stations."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=DEFAULT_CITIES,
        choices=["Chicago", "DC", "NYC"],
        help="Cities to analyze (default: Chicago DC NYC)"
    )
    parser.add_argument(
        "--walk-speed",
        type=float,
        default=1.25,
        help="Pedestrian walking speed in meters per second (default: 1.25 m/s)"
    )
    parser.add_argument(
        "--walk-time",
        type=int,
        default=10,
        help="Pedestrian walking time threshold in minutes (default: 10 min)"
    )
    parser.add_argument(
        "--output-walksheds",
        type=str,
        default="all_walksheds_10min.geojson",
        help="Output path for consolidated walksheds GeoJSON (default: all_walksheds_10min.geojson)"
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        default="station_temperature_statistics.geojson",
        help="Output path for station temperature statistics GeoJSON (default: station_temperature_statistics.geojson)"
    )
    parser.add_argument(
        "--output-map",
        type=str,
        default="station_temperature_map.html",
        help="Output path for interactive Folium map (default: station_temperature_map.html)"
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="Eos_station_walkshed_temperature_plots.png",
        help="Output path for 3-panel comparison figure (default: Eos_station_walkshed_temperature_plots.png)"
    )
    parser.add_argument(
        "--skip-walksheds",
        action="store_true",
        help="Skip OSM walkshed computation and load existing walksheds GeoJSON directly"
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Skip Folium interactive HTML map generation"
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Skip 3-panel publication figure generation"
    )

    args = parser.parse_args()

    print("===================================================================")
    print("      Transit Stations Heat Exposure Analysis Pipeline             ")
    print("===================================================================")
    print(f"Target Cities: {', '.join(args.cities)}")
    print(f"Walkshed Parameters: {args.walk_time} min at {args.walk_speed} m/s ({int(args.walk_time * 60 * args.walk_speed)} m max network reach)")

    # 1. Load station locations
    loaded_stations = load_station_locations(cities=args.cities)

    # 2. Delineate or load walksheds
    if args.skip_walksheds:
        if not os.path.exists(args.output_walksheds):
            raise FileNotFoundError(f"Existing walksheds file not found: {args.output_walksheds}. Cannot skip calculation.")
        print(f"\n[INFO] Skipping walkshed calculation; loading existing: {args.output_walksheds}")
        all_walksheds_gdf = gpd.read_file(args.output_walksheds)
        # Filter for requested cities if subset specified
        all_walksheds_gdf = all_walksheds_gdf[all_walksheds_gdf["city"].isin(args.cities)]
    else:
        configure_osmnx()
        all_walksheds_gdf, _, _, _ = generate_all_walksheds(
            loaded_stations,
            walk_speed_mps=args.walk_speed,
            trip_time_seconds=args.walk_time * 60
        )
        export_walksheds(all_walksheds_gdf, output_path=args.output_walksheds)

    # 3. Compute thermal zonal statistics
    station_stats_gdf = compute_station_temperature_statistics(
        loaded_stations,
        all_walksheds_gdf,
        temp_files=DEFAULT_TEMP_FILES
    )
    export_temperature_statistics(station_stats_gdf, output_path=args.output_stats)

    # 4. Generate visual outputs
    if not args.skip_map:
        generate_interactive_map(station_stats_gdf, output_html=args.output_map)

    if not args.skip_figure:
        generate_sample_figure(
            loaded_stations,
            all_walksheds_gdf,
            temp_files=DEFAULT_TEMP_FILES,
            output_png=args.output_figure
        )

    print("\n===================================================================")
    print("Pipeline completed successfully!")
    print(f"  * Walkshed polygons: {args.output_walksheds}")
    print(f"  * Station temperature stats: {args.output_stats}")
    if not args.skip_map:
        print(f"  * Interactive map: {args.output_map}")
    if not args.skip_figure:
        print(f"  * Publication figure: {args.output_figure}")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
