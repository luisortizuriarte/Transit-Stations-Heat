#!/usr/bin/env python3
"""
street_buffered_walkshed_analysis.py
====================================
Calculates street-segment buffered walksheds (Street Corridor Walksheds)
and extracts surface temperature statistics across transit networks in
NYC (MTA), Washington, D.C. (WMATA), and Chicago (CTA).

Concept:
Instead of constructing a filled convex hull encompassing all reachable nodes,
this analysis extracts all reachable street segments within the 10-minute
pedestrian network (750m at 1.25 m/s) and applies a metric buffer (e.g. 25m)
around each navigable edge.

This restricts Landsat Surface Temperature (ST) extraction exclusively to the
immediate pedestrian right-of-way / street corridors, excluding interior
building footprints and non-navigable block interiors.

Usage:
    python street_buffered_walkshed_analysis.py [OPTIONS]

Example:
    python street_buffered_walkshed_analysis.py --buffer-dist 25.0
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

# Safe encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Geospatial routing libraries
try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    ox = None
    nx = None

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

CITY_DISPLAY_NAMES = {
    "NYC": "New York City (MTA)",
    "DC": "Washington, D.C. (WMATA)",
    "Chicago": "Chicago (CTA)"
}


def configure_osmnx(cache_folder="cache"):
    """Configure OSMnx caching and logging parameters."""
    if ox is None:
        raise ImportError("osmnx is required for walkshed routing. Install via `pip install osmnx`.")
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_folder
    ox.settings.log_console = False


def load_station_locations(cities=None, station_files=None):
    """Load station point layers from GeoJSON files."""
    if cities is None:
        cities = DEFAULT_CITIES
    if station_files is None:
        station_files = DEFAULT_STATION_FILES

    loaded_stations = {}
    print("\n--- Loading Transit Station Locations ---")
    for city in cities:
        path = station_files.get(city)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Station file for {city} not found at: {path}")

        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)

        loaded_stations[city] = gdf
        print(f"  [OK] {city}: {len(gdf)} stations loaded from {path}")

    return loaded_stations


def calculate_street_buffered_walksheds_for_city(
    stations_4326,
    walk_speed_mps=1.25,
    trip_time_seconds=600,
    segment_buffer_m=25.0,
    buffer_deg=0.01
):
    """
    Calculate street-segment buffered walksheds for a single city.

    Parameters:
        stations_4326 (GeoDataFrame): Station point layer in EPSG:4326.
        walk_speed_mps (float): Walking velocity in m/s (default: 1.25).
        trip_time_seconds (int): Maximum travel time threshold (default: 600s).
        segment_buffer_m (float): Planar metric buffer applied to each reachable street edge.
        buffer_deg (float): Geographic buffer around station envelope for network query.

    Returns:
        street_walksheds_utm (GeoDataFrame): Street corridor polygons in local UTM.
        stations_utm (GeoDataFrame): Station points in local UTM.
        utm_crs: Projected CRS object.
    """
    if ox is None or nx is None:
        raise ImportError("osmnx and networkx are required to compute street-buffered walksheds.")

    utm_crs = stations_4326.estimate_utm_crs()

    # Delineate regional query envelope
    hull_geom = stations_4326.unary_union.convex_hull
    polygon_4326 = hull_geom.buffer(buffer_deg)

    print("    Fetching walkable street network from OpenStreetMap...")
    G = ox.graph_from_polygon(polygon_4326, network_type="walk")

    # Project graph and stations to local UTM (metric)
    G_proj = ox.project_graph(G, to_crs=utm_crs)
    stations_utm = stations_4326.to_crs(utm_crs)

    # Assign travel times to edges
    for _, _, _, data in G_proj.edges(keys=True, data=True):
        data["travel_time"] = data["length"] / walk_speed_mps

    street_walksheds = []
    print(f"    Delineating street corridor walksheds ({segment_buffer_m}m buffer)...")
    for _, station in tqdm(stations_utm.iterrows(), total=len(stations_utm), desc="    Routing & buffering edges", leave=False):
        center_node = ox.nearest_nodes(G_proj, station.geometry.x, station.geometry.y)
        subgraph = nx.ego_graph(G_proj, center_node, radius=trip_time_seconds, distance="travel_time")

        edge_geometries = []
        if len(subgraph.edges) > 0:
            for u, v, k, data in subgraph.edges(keys=True, data=True):
                if "geometry" in data:
                    edge_geometries.append(data["geometry"])
                else:
                    # Construct straight line from node coordinates if geometry attribute not present
                    p1 = Point(subgraph.nodes[u]["x"], subgraph.nodes[u]["y"])
                    p2 = Point(subgraph.nodes[v]["x"], subgraph.nodes[v]["y"])
                    edge_geometries.append(LineString([p1, p2]))

        if len(edge_geometries) > 0:
            # Buffer all reachable edges and compute unary union
            edge_series = gpd.GeoSeries(edge_geometries, crs=utm_crs)
            buffered_corridor = edge_series.buffer(segment_buffer_m).unary_union
        elif len(subgraph.nodes) > 0:
            # Fallback buffer around reachable nodes
            node_points = [Point(data["x"], data["y"]) for _, data in subgraph.nodes(data=True)]
            buffered_corridor = gpd.GeoSeries(node_points, crs=utm_crs).buffer(segment_buffer_m).unary_union
        else:
            # Fallback buffer around station point
            buffered_corridor = station.geometry.buffer(segment_buffer_m)

        street_walksheds.append(buffered_corridor)

    street_walksheds_utm = gpd.GeoDataFrame(geometry=street_walksheds, crs=utm_crs)
    return street_walksheds_utm, stations_utm, utm_crs


def generate_all_street_buffered_walksheds(
    loaded_stations,
    walk_speed_mps=1.25,
    trip_time_seconds=600,
    segment_buffer_m=25.0
):
    """
    Run street-segment buffered walkshed generation across all study cities.

    Returns:
        all_street_walksheds_gdf (GeoDataFrame): Consolidated dataset in EPSG:4326.
    """
    all_records = []
    print(f"\n--- Generating Street-Segment Buffered Walksheds ({segment_buffer_m}m Buffer) ---")

    for city, stations_gdf in loaded_stations.items():
        print(f"\nProcessing {city} ({len(stations_gdf)} stations)...")
        walksheds_utm, stations_utm, utm_crs = calculate_street_buffered_walksheds_for_city(
            stations_gdf,
            walk_speed_mps=walk_speed_mps,
            trip_time_seconds=trip_time_seconds,
            segment_buffer_m=segment_buffer_m
        )

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
                "walkshed_type": "street_corridor",
                "segment_buffer_m": segment_buffer_m,
                "original_crs": str(utm_crs),
                "output_crs": "EPSG:4326",
                "walk_time_minutes": int(trip_time_seconds / 60),
                "walk_speed_mps": walk_speed_mps,
                "area_km2": walksheds_utm.geometry.iloc[i].area / 1e6
            }

            for col in orig_row.index:
                if col != "geometry" and col not in rec:
                    rec[col] = orig_row[col]

            all_records.append(rec)

    all_street_walksheds_gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
    return all_street_walksheds_gdf


def extract_zonal_temperature_stats(walkshed_geom, walkshed_crs, raster_path):
    """Clip thermal GeoTIFF to walkshed polygon and calculate percentile statistics."""
    empty_stats = {
        "median_temp": np.nan,
        "temp_10th": np.nan,
        "temp_25th": np.nan,
        "temp_75th": np.nan,
        "temp_90th": np.nan,
        "temp_99th": np.nan,
        "pixel_count": 0
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
                "temp_99th": float(np.percentile(valid_values, 99)),
                "pixel_count": int(len(valid_values))
            }
    except Exception:
        return empty_stats


def compute_street_temperature_statistics(
    loaded_stations,
    street_walksheds_gdf,
    hull_stats_path="station_temperature_statistics.geojson",
    temp_files=None
):
    """
    Extract zonal LST statistics from street corridor walksheds and compute delta vs hull.

    Returns:
        station_street_stats_gdf (GeoDataFrame): Station points with street thermal metrics and delta T.
    """
    if temp_files is None:
        temp_files = DEFAULT_TEMP_FILES

    print("\n--- Extracting Thermal Zonal Statistics for Street Corridors ---")
    station_records = []

    hull_gdf = None
    if hull_stats_path and os.path.exists(hull_stats_path):
        hull_gdf = gpd.read_file(hull_stats_path)

    for city in loaded_stations.keys():
        print(f"\nExtracting street-corridor LST statistics for {city}...")
        raster_path = temp_files.get(city)
        city_walksheds = street_walksheds_gdf[street_walksheds_gdf["city"] == city]
        city_stations = loaded_stations[city]
        city_hull = hull_gdf[hull_gdf["city"] == city] if hull_gdf is not None else None

        for i in tqdm(range(len(city_walksheds)), desc=f"  Zonal stats ({city})", leave=False):
            w_row = city_walksheds.iloc[i]
            s_row = city_stations.iloc[i]

            walkshed_geom = w_row.geometry
            stats = extract_zonal_temperature_stats(walkshed_geom, "EPSG:4326", raster_path)

            # Compare with convex hull median temp if available
            hull_med = np.nan
            if city_hull is not None and i < len(city_hull):
                hull_med = city_hull.iloc[i].get("median_temp", np.nan)

            delta_temp = stats["median_temp"] - hull_med if not np.isnan(stats["median_temp"]) and not np.isnan(hull_med) else np.nan

            station_point = s_row.geometry
            record = {
                "city": city,
                "station_name": w_row.get("station_name", f"{city}_Station_{i+1}"),
                "geometry": station_point,
                "walkshed_type": "street_corridor",
                "segment_buffer_m": w_row.get("segment_buffer_m", 25.0),
                "corridor_area_km2": w_row.get("area_km2", np.nan),
                "street_median_temp": stats["median_temp"],
                "street_temp_10th": stats["temp_10th"],
                "street_temp_25th": stats["temp_25th"],
                "street_temp_75th": stats["temp_75th"],
                "street_temp_90th": stats["temp_90th"],
                "street_temp_99th": stats["temp_99th"],
                "street_pixel_count": stats["pixel_count"],
                "hull_median_temp": hull_med,
                "delta_street_vs_hull": delta_temp
            }

            for col in s_row.index:
                if col != "geometry" and col not in record:
                    record[col] = s_row[col]

            station_records.append(record)

    station_street_stats_gdf = gpd.GeoDataFrame(station_records, crs="EPSG:4326")
    return station_street_stats_gdf


def export_datasets(
    street_walksheds_gdf,
    street_stats_gdf,
    output_walksheds_path="all_street_buffered_walksheds_10min.geojson",
    output_stats_path="station_street_temperature_statistics.geojson"
):
    """Export standardized GeoJSON layers in EPSG:4326."""
    print(f"\n--- Exporting Street-Buffered Walkshed Datasets ---")
    street_walksheds_gdf.to_file(output_walksheds_path, driver="GeoJSON")
    print(f"  [OK] Saved consolidated street walksheds: {output_walksheds_path} ({len(street_walksheds_gdf)} polygons)")

    street_stats_gdf.to_file(output_stats_path, driver="GeoJSON")
    print(f"  [OK] Saved consolidated street statistics: {output_stats_path} ({len(street_stats_gdf)} stations)")

    for city in street_walksheds_gdf["city"].unique():
        sub_w = street_walksheds_gdf[street_walksheds_gdf["city"] == city]
        w_file = f"street_walksheds_10min_{city.lower()}.geojson"
        sub_w.to_file(w_file, driver="GeoJSON")

        sub_s = street_stats_gdf[street_stats_gdf["city"] == city]
        s_file = f"station_street_temperature_statistics_{city.lower()}.geojson"
        sub_s.to_file(s_file, driver="GeoJSON")
        print(f"  [OK] Saved {city} layers: {w_file} and {s_file}")


def generate_comparison_figure(
    loaded_stations,
    street_walksheds_gdf,
    hull_walksheds_path="all_walksheds_10min.geojson",
    output_png="convex_vs_street_buffered_walksheds.png",
    temp_files=None,
    vmin=30.0,
    vmax=45.0,
    dpi=300
):
    """
    Generate a 3-city x 2-column comparative figure directly contrasting
    Standard Convex Hull Walksheds vs. Street-Segment Buffered Walksheds.
    """
    if ox is None:
        print("  [WARN] osmnx not available for comparative figure.")
        return

    if not os.path.exists(hull_walksheds_path):
        print(f"  [WARN] {hull_walksheds_path} not found. Skipping comparative figure.")
        return

    if temp_files is None:
        temp_files = DEFAULT_TEMP_FILES

    print("\n--- Generating Convex Hull vs. Street-Buffered Comparison Figure ---")
    hull_walksheds_gdf = gpd.read_file(hull_walksheds_path)

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 13.5), dpi=dpi)
    sample_cities = ["NYC", "DC", "Chicago"]
    im_ref = None

    for row_idx, city in enumerate(sample_cities):
        city_stations = loaded_stations[city]
        city_hull = hull_walksheds_gdf[hull_walksheds_gdf["city"] == city]
        city_street = street_walksheds_gdf[street_walksheds_gdf["city"] == city]
        raster_path = temp_files.get(city)

        # Select representative sample station
        sample_idx = min(21, len(city_stations) - 1)
        s_row = city_stations.iloc[sample_idx]
        h_row = city_hull.iloc[sample_idx]
        w_row = city_street.iloc[sample_idx]

        station_point = s_row.geometry
        station_name = w_row.get("station_name", f"{city} Station")

        # UTM conversion
        gdf_pt = gpd.GeoDataFrame(geometry=[station_point], crs="EPSG:4326")
        utm_crs = gdf_pt.estimate_utm_crs()

        stn_utm = gdf_pt.to_crs(utm_crs).geometry.iloc[0]
        hull_utm = gpd.GeoDataFrame(geometry=[h_row.geometry], crs="EPSG:4326").to_crs(utm_crs).geometry.iloc[0]
        street_utm = gpd.GeoDataFrame(geometry=[w_row.geometry], crs="EPSG:4326").to_crs(utm_crs).geometry.iloc[0]

        # Fetch background street network
        G = ox.graph_from_point((station_point.y, station_point.x), dist=1200, network_type="walk")
        G_proj = ox.project_graph(G, to_crs=utm_crs)
        edges = ox.graph_to_gdfs(G_proj, nodes=False)

        # Column 0: Standard Convex Hull
        ax_hull = axes[row_idx, 0]
        edges.plot(ax=ax_hull, color="#888888", linewidth=0.5, alpha=0.45, zorder=1)

        # Clip raster with convex hull
        if raster_path and os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                h_gdf = gpd.GeoDataFrame(geometry=[hull_utm], crs=utm_crs).to_crs(src.crs)
                out_img, out_tr = mask(src, h_gdf.geometry, crop=True, nodata=np.nan)
                tr, w, h = calculate_default_transform(src.crs, utm_crs, out_img.shape[2], out_img.shape[1], left=out_tr.c, bottom=out_tr.f + out_tr.e * out_img.shape[1], right=out_tr.c + out_tr.a * out_img.shape[2], top=out_tr.f)
                dest = np.zeros((1, h, w), dtype=np.float32)
                reproject(out_img, dest, src_transform=out_tr, src_crs=src.crs, dst_transform=tr, dst_crs=utm_crs, resampling=Resampling.nearest)
                dest[0] = np.where(dest[0] == 0, np.nan, dest[0])
                im_ref = ax_hull.imshow(dest[0], extent=[tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f], cmap="YlOrRd", alpha=0.80, vmin=vmin, vmax=vmax, zorder=2)

        gpd.GeoSeries([hull_utm], crs=utm_crs).plot(ax=ax_hull, facecolor="none", edgecolor="black", linewidth=1.5, zorder=3)
        ax_hull.scatter(stn_utm.x, stn_utm.y, color="cyan", edgecolor="black", s=85, linewidth=1.0, zorder=4)
        ax_hull.set_title(f"{city}: {station_name}\n(Convex Hull Polygon)", fontsize=10.5, fontweight="bold", pad=6)
        ax_hull.axis("off")
        ax_hull.set_aspect("equal")

        # Column 1: Street-Segment Buffered Walkshed
        ax_street = axes[row_idx, 1]
        edges.plot(ax=ax_street, color="#888888", linewidth=0.5, alpha=0.45, zorder=1)

        # Clip raster with street corridor buffer
        if raster_path and os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                s_gdf = gpd.GeoDataFrame(geometry=[street_utm], crs=utm_crs).to_crs(src.crs)
                out_img, out_tr = mask(src, s_gdf.geometry, crop=True, nodata=np.nan)
                tr, w, h = calculate_default_transform(src.crs, utm_crs, out_img.shape[2], out_img.shape[1], left=out_tr.c, bottom=out_tr.f + out_tr.e * out_img.shape[1], right=out_tr.c + out_tr.a * out_img.shape[2], top=out_tr.f)
                dest = np.zeros((1, h, w), dtype=np.float32)
                reproject(out_img, dest, src_transform=out_tr, src_crs=src.crs, dst_transform=tr, dst_crs=utm_crs, resampling=Resampling.nearest)
                dest[0] = np.where(dest[0] == 0, np.nan, dest[0])
                ax_street.imshow(dest[0], extent=[tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f], cmap="YlOrRd", alpha=0.85, vmin=vmin, vmax=vmax, zorder=2)

        gpd.GeoSeries([street_utm], crs=utm_crs).plot(ax=ax_street, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)
        ax_street.scatter(stn_utm.x, stn_utm.y, color="cyan", edgecolor="black", s=85, linewidth=1.0, zorder=4)
        ax_street.set_title(f"{city}: {station_name}\n(Street Corridor Buffer)", fontsize=10.5, fontweight="bold", pad=6)
        ax_street.axis("off")
        ax_street.set_aspect("equal")

    # Titles for columns
    axes[0, 0].text(0.5, 1.28, "Standard Convex Hull Walkshed\n(Enclosed Catchment Area)", transform=axes[0, 0].transAxes, fontsize=11.5, fontweight="bold", ha="center", va="bottom")
    axes[0, 1].text(0.5, 1.28, "Street-Segment Buffered Walkshed\n(Pedestrian Right-of-Way Corridor)", transform=axes[0, 1].transAxes, fontsize=11.5, fontweight="bold", ha="center", va="bottom")

    # Shared colorbar
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.020])
        cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Surface Temperature (°C)", fontsize=10.5, labelpad=5)
        cbar.ax.tick_params(labelsize=9)

    plt.subplots_adjust(top=0.91, bottom=0.09, left=0.05, right=0.95, hspace=0.28, wspace=0.15)
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved comparative figure: {output_png}")


def generate_street_grid_3x3(
    street_walksheds_gdf,
    output_png="Eos_street_buffered_walkshed_plots_3x3.png",
    temp_files=None,
    vmin=30.0,
    vmax=45.0,
    dpi=300
):
    """
    Generate a 3x3 multi-station grid figure using the street-buffered walksheds.
    """
    if ox is None:
        return

    if temp_files is None:
        temp_files = DEFAULT_TEMP_FILES

    print("\n--- Generating 3x3 Street-Buffered Walkshed Figure ---")
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.5), dpi=dpi)
    im_ref = None

    default_selected = {
        "NYC": ["36 Av", "Times Sq-42 St", "City Hall"],
        "DC": ["Rhode Island Ave-Brentwood", "Columbia Heights", "Cleveland Park"],
        "Chicago": ["Cumberland", "Clinton-Lake", "Loyola"]
    }

    for col_idx, city in enumerate(["NYC", "DC", "Chicago"]):
        city_w = street_walksheds_gdf[street_walksheds_gdf["city"] == city]
        raster_path = temp_files.get(city)
        selected_names = default_selected.get(city, [0, 1, 2])

        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            station_id = selected_names[row_idx]

            # Find matching station row
            if isinstance(station_id, str):
                match = city_w[city_w["station_name"].str.contains(station_id, case=False, na=False)]
                station_row = match.iloc[0] if len(match) > 0 else city_w.iloc[row_idx]
            else:
                station_row = city_w.iloc[station_id]

            station_name = station_row["station_name"]
            station_lon = station_row["station_lon"]
            station_lat = station_row["station_lat"]
            walkshed_geom = station_row.geometry

            # UTM conversion
            gdf_pt = gpd.GeoDataFrame(geometry=[Point(station_lon, station_lat)], crs="EPSG:4326")
            utm_crs = gdf_pt.estimate_utm_crs()

            stn_utm = gdf_pt.to_crs(utm_crs).geometry.iloc[0]
            walkshed_utm = gpd.GeoDataFrame(geometry=[walkshed_geom], crs="EPSG:4326").to_crs(utm_crs).geometry.iloc[0]

            # Fetch background street network
            try:
                G = ox.graph_from_point((station_lat, station_lon), dist=1200, network_type="walk")
                G_proj = ox.project_graph(G, to_crs=utm_crs)
                edges = ox.graph_to_gdfs(G_proj, nodes=False)
                edges.plot(ax=ax, color="#888888", linewidth=0.5, alpha=0.45, zorder=1)
            except Exception:
                pass

            # Clip raster
            if raster_path and os.path.exists(raster_path):
                try:
                    with rasterio.open(raster_path) as src:
                        s_gdf = gpd.GeoDataFrame(geometry=[walkshed_utm], crs=utm_crs).to_crs(src.crs)
                        out_img, out_tr = mask(src, s_gdf.geometry, crop=True, nodata=np.nan)
                        tr, w, h = calculate_default_transform(src.crs, utm_crs, out_img.shape[2], out_img.shape[1], left=out_tr.c, bottom=out_tr.f + out_tr.e * out_img.shape[1], right=out_tr.c + out_tr.a * out_img.shape[2], top=out_tr.f)
                        dest = np.zeros((1, h, w), dtype=np.float32)
                        reproject(out_img, dest, src_transform=out_tr, src_crs=src.crs, dst_transform=tr, dst_crs=utm_crs, resampling=Resampling.nearest)
                        dest[0] = np.where(dest[0] == 0, np.nan, dest[0])
                        im_ref = ax.imshow(dest[0], extent=[tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f], cmap="YlOrRd", alpha=0.85, vmin=vmin, vmax=vmax, zorder=2)
                except Exception:
                    gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="#3498DB", alpha=0.35, zorder=2)

            gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)
            ax.scatter(stn_utm.x, stn_utm.y, color="cyan", edgecolor="black", s=80, linewidth=1.0, zorder=4)

            med_temp = station_row.get("median_temp", station_row.get("street_median_temp", None))
            if med_temp is not None and not np.isnan(med_temp):
                title_str = f"{station_name}\n({med_temp:.1f}°C)"
            else:
                title_str = f"{station_name}"

            ax.set_title(title_str, fontsize=10.0, fontweight="bold", pad=5)
            ax.axis("off")
            ax.set_aspect("equal")

    # Column titles
    for col_idx, city in enumerate(["NYC", "DC", "Chicago"]):
        axes[0, col_idx].text(0.5, 1.25, CITY_DISPLAY_NAMES[city], transform=axes[0, col_idx].transAxes, fontsize=12, fontweight="bold", ha="center", va="bottom")

    if im_ref is not None:
        cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.022])
        cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Surface Temperature (°C)", fontsize=11, labelpad=5)
        cbar.ax.tick_params(labelsize=9.5)

    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.04, right=0.96, hspace=0.28, wspace=0.18)
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved 3x3 street corridor figure: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute street-segment buffered walksheds and extract corridor surface temperatures across transit networks."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=DEFAULT_CITIES,
        choices=["Chicago", "DC", "NYC"],
        help="Cities to analyze (default: Chicago DC NYC)"
    )
    parser.add_argument(
        "--buffer-dist",
        type=float,
        default=25.0,
        help="Planar buffer distance in meters applied to each reachable street segment (default: 25.0m)"
    )
    parser.add_argument(
        "--walk-speed",
        type=float,
        default=1.25,
        help="Pedestrian walking speed in m/s (default: 1.25 m/s)"
    )
    parser.add_argument(
        "--walk-time",
        type=int,
        default=10,
        help="Walking duration threshold in minutes (default: 10 min)"
    )
    parser.add_argument(
        "--output-walksheds",
        type=str,
        default="all_street_buffered_walksheds_10min.geojson",
        help="Output GeoJSON path for street-buffered walkshed polygons"
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        default="station_street_temperature_statistics.geojson",
        help="Output GeoJSON path for station street temperature statistics"
    )
    parser.add_argument(
        "--output-comparison-fig",
        type=str,
        default="convex_vs_street_buffered_walksheds.png",
        help="Output PNG path for comparison figure (Convex Hull vs. Street Buffer)"
    )
    parser.add_argument(
        "--output-grid-fig",
        type=str,
        default="Eos_street_buffered_walkshed_plots_3x3.png",
        help="Output PNG path for 3x3 street-buffered walkshed grid"
    )
    parser.add_argument(
        "--skip-walksheds",
        action="store_true",
        help="Skip street network routing and reuse existing street walkshed polygons"
    )
    parser.add_argument(
        "--skip-figs",
        action="store_true",
        help="Skip generating PNG visual figures"
    )

    args = parser.parse_args()

    print("===================================================================")
    print("      Street-Segment Buffered Walkshed Analysis Pipeline           ")
    print("===================================================================")
    print(f"Target Cities: {', '.join(args.cities)}")
    print(f"Segment Buffer Distance: {args.buffer_dist} meters")
    print(f"Walkshed Parameters: {args.walk_time} min at {args.walk_speed} m/s ({int(args.walk_time * 60 * args.walk_speed)} m max reach)")

    loaded_stations = load_station_locations(cities=args.cities)

    # 1. Delineate or load street-buffered walksheds
    if args.skip_walksheds:
        if not os.path.exists(args.output_walksheds):
            raise FileNotFoundError(f"Existing street walksheds file not found: {args.output_walksheds}")
        print(f"\n[INFO] Skipping routing; loading existing street walksheds from {args.output_walksheds}")
        street_walksheds_gdf = gpd.read_file(args.output_walksheds)
        street_walksheds_gdf = street_walksheds_gdf[street_walksheds_gdf["city"].isin(args.cities)]
    else:
        configure_osmnx()
        street_walksheds_gdf = generate_all_street_buffered_walksheds(
            loaded_stations,
            walk_speed_mps=args.walk_speed,
            trip_time_seconds=args.walk_time * 60,
            segment_buffer_m=args.buffer_dist
        )

    # 2. Extract zonal statistics & compute delta vs convex hull
    street_stats_gdf = compute_street_temperature_statistics(
        loaded_stations,
        street_walksheds_gdf,
        hull_stats_path="station_temperature_statistics.geojson",
        temp_files=DEFAULT_TEMP_FILES
    )

    # Attach stats to walksheds for visualization
    if "median_temp" not in street_walksheds_gdf.columns and "street_median_temp" in street_stats_gdf.columns:
        street_walksheds_gdf["median_temp"] = street_stats_gdf["street_median_temp"].values

    # 3. Export datasets
    export_datasets(
        street_walksheds_gdf,
        street_stats_gdf,
        output_walksheds_path=args.output_walksheds,
        output_stats_path=args.output_stats
    )

    # 4. Generate visual outputs
    if not args.skip_figs:
        generate_comparison_figure(
            loaded_stations,
            street_walksheds_gdf,
            hull_walksheds_path="all_walksheds_10min.geojson",
            output_png=args.output_comparison_fig,
            temp_files=DEFAULT_TEMP_FILES
        )

        generate_street_grid_3x3(
            street_walksheds_gdf,
            output_png=args.output_grid_fig,
            temp_files=DEFAULT_TEMP_FILES
        )

    # Summary of delta analysis
    valid_deltas = street_stats_gdf["delta_street_vs_hull"].dropna()
    if len(valid_deltas) > 0:
        print("\n--- Summary of Street Corridor vs. Convex Hull Differences ---")
        print(f"  • Total stations evaluated: {len(valid_deltas)}")
        print(f"  • Mean Difference (Street - Hull): {valid_deltas.mean():+.2f}°C")
        print(f"  • Median Difference: {valid_deltas.median():+.2f}°C")
        print(f"  • Min / Max Difference: {valid_deltas.min():+.2f}°C to {valid_deltas.max():+.2f}°C")
        print(f"  • Average Corridor Land Area: {street_walksheds_gdf['area_km2'].mean():.2f} km²")

    print("\n===================================================================")
    print("Street-Buffered Walkshed Analysis completed successfully!")
    print(f"  * Street corridor walksheds: {args.output_walksheds}")
    print(f"  * Street temperature stats: {args.output_stats}")
    if not args.skip_figs:
        print(f"  * Comparison figure: {args.output_comparison_fig}")
        print(f"  * 3x3 Street grid figure: {args.output_grid_fig}")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
