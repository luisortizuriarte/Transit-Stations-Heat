#!/usr/bin/env python3
"""
generate_walkshed_grid_3x3.py
=============================
Generates a 3-column x 3-row publication figure comparing 9 distinct transit
station walksheds across New York City (MTA), Washington, D.C. (WMATA), and
Chicago (CTA).

Structure:
- Columns (3 cities): NYC, Washington D.C., Chicago
- Rows (3 stations per city): 3 distinct stations displaying diverse microclimates
  and urban morphology (e.g. High heat corridor, Urban Core, and Moderated/outer corridor)
- Consistent colormap across all 9 panels (default: YlOrRd from 30°C to 45°C)
- Pedestrian network graph overlay, clipped Landsat LST raster, walkshed boundary,
  and transit station point marker.
- Unified shared horizontal colorbar.

Usage:
    python generate_walkshed_grid_3x3.py [OPTIONS]

Example:
    python generate_walkshed_grid_3x3.py --output Eos_station_walkshed_temperature_plots_3x3.png
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

# Safe encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Required geospatial libraries
try:
    import osmnx as ox
except ImportError:
    ox = None

warnings.filterwarnings("ignore")

# Configuration constants
CITIES = ["NYC", "DC", "Chicago"]

CITY_DISPLAY_NAMES = {
    "NYC": "New York City (MTA)",
    "DC": "Washington, D.C. (WMATA)",
    "Chicago": "Chicago (CTA)"
}

TEMP_FILES = {
    "NYC": "stitched_city_temperatures/NYC_stitched_surface_temperature.tif",
    "DC": "stitched_city_temperatures/DC_stitched_surface_temperature.tif",
    "Chicago": "stitched_city_temperatures/Chicago_stitched_surface_temperature.tif"
}

# Curated default representative stations across 3 tiers (High heat, Core hub, Moderated/canopy)
DEFAULT_SELECTED_STATIONS = {
    "NYC": [
        "36 Av",                  # Row 1: High heat / Queens elevated corridor (~42.0°C)
        "Times Sq-42 St",         # Row 2: Dense Manhattan commercial core (~40.6°C)
        "City Hall"               # Row 3: Lower Manhattan / Park vicinity (~39.3°C)
    ],
    "DC": [
        "Rhode Island Ave-Brentwood", # Row 1: High heat / industrial rail corridor (~49.3°C)
        "Columbia Heights",           # Row 2: Dense mixed-use urban corridor (~46.9°C)
        "Cleveland Park"              # Row 3: Canopy cover / residential northwest (~41.0°C)
    ],
    "Chicago": [
        "Cumberland",             # Row 1: High heat / expressway transit center (~38.7°C)
        "Clinton-Lake",           # Row 2: West Loop / River edge core (~36.4°C)
        "Loyola"                  # Row 3: Lakefront / north residential (~33.4°C)
    ]
}


def setup_osmnx(cache_folder="cache"):
    """Enable OSMnx caching for fast repeatable network fetches."""
    if ox is None:
        raise ImportError("osmnx is required for street network rendering. Install via `pip install osmnx`.")
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_folder
    ox.settings.log_console = False


def find_station_row(gdf_city, identifier):
    """
    Find matching station row by exact or partial name match, or by integer index.
    """
    if isinstance(identifier, int):
        if 0 <= identifier < len(gdf_city):
            return gdf_city.iloc[identifier]
        raise IndexError(f"Station index {identifier} out of range for city dataset.")

    # Try exact name match
    name_str = str(identifier).strip()
    match = gdf_city[gdf_city["station_name"].str.strip().str.lower() == name_str.lower()]
    if len(match) > 0:
        return match.iloc[0]

    # Try substring match
    match = gdf_city[gdf_city["station_name"].str.contains(name_str, case=False, na=False)]
    if len(match) > 0:
        return match.iloc[0]

    # Fallback to first row
    print(f"  [WARN] Station '{identifier}' not found in city dataset. Falling back to first station.")
    return gdf_city.iloc[0]


def render_station_panel(ax, station_row, temp_raster_path, vmin=30.0, vmax=45.0, cmap="YlOrRd"):
    """
    Render a single panel with street network, clipped surface temperature raster,
    walkshed boundary polygon, and station point marker.
    """
    station_name = station_row.get("station_name", "Transit Station")
    city = station_row.get("city", "Metro")
    walkshed_geom = station_row.geometry

    # Station coordinates
    station_lon = station_row.get("station_lon", None)
    station_lat = station_row.get("station_lat", None)
    if station_lon is None or station_lat is None or np.isnan(station_lon):
        if walkshed_geom.geom_type == "Point":
            station_point_4326 = walkshed_geom
        else:
            station_point_4326 = walkshed_geom.centroid
    else:
        station_point_4326 = Point(station_lon, station_lat)

    # Local UTM projection
    gdf_pt = gpd.GeoDataFrame(geometry=[station_point_4326], crs="EPSG:4326")
    utm_crs = gdf_pt.estimate_utm_crs()

    station_utm = gdf_pt.to_crs(utm_crs).geometry.iloc[0]
    walkshed_utm = gpd.GeoDataFrame(geometry=[walkshed_geom], crs="EPSG:4326").to_crs(utm_crs).geometry.iloc[0]

    # 1. Fetch walkable street network in 1.2 km radius around station
    try:
        G = ox.graph_from_point((station_point_4326.y, station_point_4326.x), dist=1200, network_type="walk")
        G_proj = ox.project_graph(G, to_crs=utm_crs)
        edges = ox.graph_to_gdfs(G_proj, nodes=False)
        edges.plot(ax=ax, color="#888888", linewidth=0.5, alpha=0.45, zorder=1)
    except Exception as e:
        print(f"    Notice: Could not fetch street network for {station_name}: {e}")

    im = None
    # 2. Clip and plot temperature raster within walkshed polygon
    if temp_raster_path and os.path.exists(temp_raster_path):
        try:
            with rasterio.open(temp_raster_path) as src:
                walkshed_gdf_tmp = gpd.GeoDataFrame(geometry=[walkshed_utm], crs=utm_crs)
                walkshed_raster_crs = walkshed_gdf_tmp.to_crs(src.crs)

                out_image, out_transform = mask(src, walkshed_raster_crs.geometry, crop=True, nodata=np.nan)

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
                    cmap=cmap,
                    alpha=0.80,
                    zorder=2,
                    vmin=vmin,
                    vmax=vmax
                )
        except Exception as e:
            print(f"    Notice: Could not clip temperature raster for {station_name}: {e}")
            gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="#3498DB", alpha=0.35, zorder=2)
    else:
        gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(ax=ax, facecolor="#3498DB", alpha=0.35, zorder=2)

    # 3. Plot walkshed boundary polygon
    gpd.GeoSeries([walkshed_utm], crs=utm_crs).plot(
        ax=ax, facecolor="none", edgecolor="black", linewidth=1.4, zorder=3
    )

    # 4. Plot transit station portal point marker
    ax.scatter(
        station_utm.x, station_utm.y,
        color="cyan", edgecolor="black", s=85, linewidth=1.0, zorder=4
    )

    # Clean formatting
    med_temp = station_row.get("median_temp", None)
    if med_temp is not None and not np.isnan(med_temp):
        title_text = f"{station_name}\n({med_temp:.1f}°C)"
    else:
        title_text = f"{station_name}"

    ax.set_title(title_text, fontsize=10.5, fontweight="bold", pad=5)
    ax.axis("off")
    ax.set_aspect("equal")

    return im


def generate_grid_3x3(
    walksheds_path="all_walksheds_10min.geojson",
    stats_path="station_temperature_statistics.geojson",
    output_path="Eos_station_walkshed_temperature_plots_3x3.png",
    vmin=30.0,
    vmax=45.0,
    cmap="YlOrRd",
    selected_stations_map=None,
    dpi=300
):
    """
    Build and save the 3x3 multi-station publication grid figure.
    """
    setup_osmnx()

    if not os.path.exists(walksheds_path):
        raise FileNotFoundError(f"Walksheds dataset not found: {walksheds_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Station statistics dataset not found: {stats_path}")

    print(f"\n--- Loading geospatial datasets ---")
    walksheds_gdf = gpd.read_file(walksheds_path)
    stats_gdf = gpd.read_file(stats_path)

    # Merge median_temp from stats into walksheds if needed
    if "median_temp" not in walksheds_gdf.columns and "median_temp" in stats_gdf.columns:
        walksheds_gdf["median_temp"] = stats_gdf["median_temp"].values

    if selected_stations_map is None:
        selected_stations_map = DEFAULT_SELECTED_STATIONS

    print(f"\n--- Constructing 3x3 Multi-Station Figure ---")
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.5), dpi=dpi)

    im_ref = None

    for col_idx, city in enumerate(CITIES):
        city_walksheds = walksheds_gdf[walksheds_gdf["city"] == city]
        city_raster = TEMP_FILES.get(city)
        selected_stations = selected_stations_map.get(city, [0, 1, 2])

        print(f"\nRendering Column {col_idx+1}: {CITY_DISPLAY_NAMES[city]}...")
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            station_id = selected_stations[row_idx]
            station_row = find_station_row(city_walksheds, station_id)
            print(f"  Row {row_idx+1}: {station_row['station_name']}")

            im = render_station_panel(
                ax,
                station_row,
                city_raster,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap
            )
            if im is not None:
                im_ref = im

    # Column Titles across top row
    for col_idx, city in enumerate(CITIES):
        axes[0, col_idx].text(
            0.5, 1.25, CITY_DISPLAY_NAMES[city],
            transform=axes[0, col_idx].transAxes,
            fontsize=12, fontweight="bold",
            ha="center", va="bottom"
        )

    # Add shared unified colorbar at bottom
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.022])
        cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Surface Temperature (°C)", fontsize=11, labelpad=5)
        cbar.ax.tick_params(labelsize=9.5)

    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.04, right=0.96, hspace=0.28, wspace=0.18)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Successfully saved 3x3 figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3-column x 3-row publication figure comparing 9 transit station walksheds across NYC, DC, and Chicago."
    )
    parser.add_argument(
        "--walksheds",
        type=str,
        default="all_walksheds_10min.geojson",
        help="Path to all_walksheds_10min.geojson"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="station_temperature_statistics.geojson",
        help="Path to station_temperature_statistics.geojson"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Eos_station_walkshed_temperature_plots_3x3.png",
        help="Output PNG path (default: Eos_station_walkshed_temperature_plots_3x3.png)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=30.0,
        help="Minimum surface temperature for colormap in °C (default: 30.0)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=45.0,
        help="Maximum surface temperature for colormap in °C (default: 45.0)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlOrRd",
        help="Matplotlib colormap name (default: YlOrRd)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image resolution DPI (default: 300)"
    )

    args = parser.parse_args()

    print("===================================================================")
    print("      3x3 Transit Station Walkshed Temperature Figure Generator     ")
    print("===================================================================")

    generate_grid_3x3(
        walksheds_path=args.walksheds,
        stats_path=args.stats,
        output_path=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
