"""
generate_walkshed_maps.py
=========================
Production script to generate publication-ready multi-city figures
displaying pedestrian walkshed isochrones across New York City (MTA),
Chicago (CTA), and Washington, D.C. (WMATA).

Options:
- Option 1: Full metropolitan overview (1 column x 3 rows) displaying all 738 station walksheds.
- Option 2: High-density urban core zoom-in (1 column x 3 rows) showing detailed overlapping walkshed morphology.
- Option 3: Walksheds choropleth-colored by zonal median surface temperature (°C) (1 column x 3 rows).
- Option 4: Walksheds choropleth-colored by zonal median surface temperature (°C) in a 1 row x 3 columns horizontal design.

Constraints Satisfied:
- CartoDB Positron basemap from contextily on all panels.
- Only the city name (e.g., 'NYC', 'Chicago', 'DC') as ax title.
- No figure suptitle.
"""

import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import contextily as cx
import warnings

# Suppress minor projection / GDAL runtime warnings
warnings.filterwarnings("ignore")

# Configuration constants
CITIES = ["NYC", "Chicago", "DC"]
CITY_TITLES = {
    "NYC": "NYC",
    "Chicago": "Chicago",
    "DC": "DC"
}

# Coordinate bounding boxes (WGS84) for Option 2 core transit zoom-in
ZOOM_BOUNDS_WGS84 = {
    "NYC": box(-74.020, 40.700, -73.960, 40.775),       # Manhattan Core (Downtown to Central Park South)
    "Chicago": box(-87.665, 41.865, -87.615, 41.915),   # The Loop & River North / West Loop
    "DC": box(-77.055, 38.875, -76.995, 38.925)        # Downtown DC, Metro Center & National Mall
}


def load_datasets():
    """Load walksheds and station statistics GeoJSON datasets in Web Mercator (EPSG:3857)."""
    walksheds_path = "all_walksheds_10min.geojson"
    stats_path = "station_temperature_statistics.geojson"

    if not os.path.exists(walksheds_path):
        raise FileNotFoundError(f"Walkshed dataset not found: {walksheds_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Station statistics dataset not found: {stats_path}")

    # Load and reproject to Web Mercator (EPSG:3857) for seamless Contextily basemap alignment
    walksheds_gdf = gpd.read_file(walksheds_path).to_crs(epsg=3857)
    stats_gdf = gpd.read_file(stats_path).to_crs(epsg=3857)

    # Attach temperature attributes to walkshed polygons
    if "median_temp" in stats_gdf.columns and "median_temp" not in walksheds_gdf.columns:
        walksheds_gdf["median_temp"] = stats_gdf["median_temp"].values

    return walksheds_gdf, stats_gdf


def create_option1_full_overview(walksheds_gdf, stats_gdf, output_path="walksheds_option1_overview.png", dpi=300):
    """
    Option 1: Full metropolitan overview displaying all station walksheds across NYC, Chicago, and DC.
    1-column, 3-panel figure with Contextily basemaps and city names as ax titles.
    """
    print("Generating Option 1: Full-City Walkshed Overview (1 column x 3 rows)...")
    fig, axes = plt.subplots(3, 1, figsize=(7, 18), dpi=dpi)

    for ax, city in zip(axes, CITIES):
        w_city = walksheds_gdf[walksheds_gdf["city"] == city]
        s_city = stats_gdf[stats_gdf["city"] == city]

        # Plot walksheds with semi-transparent fill and clear boundary outline
        w_city.plot(
            ax=ax,
            facecolor="#1F4E79",
            edgecolor="#0D233A",
            alpha=0.35,
            linewidth=0.7,
            zorder=2
        )

        # Plot station point markers
        s_city.plot(
            ax=ax,
            color="#D9381E",
            markersize=14,
            edgecolor="white",
            linewidth=0.5,
            zorder=3
        )

        # Adjust axis limits with 5% spatial margin
        minx, miny, maxx, maxy = w_city.total_bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        # Add Contextily basemap
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron, alpha=0.85, zorder=1)

        # Set ax title (strictly city name only) and clean formatting
        ax.set_title(CITY_TITLES[city], fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Clean layout without figure suptitle
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved Option 1 to: {output_path}")


def create_option2_zoomed_core(walksheds_gdf, stats_gdf, output_path="walksheds_option2_zoomed.png", dpi=300):
    """
    Option 2: High-density urban core zoom-in showing detailed overlapping walkshed morphology.
    1-column, 3-panel figure with street-level Contextily basemaps and city names as ax titles.
    """
    print("Generating Option 2: High-Density Core Zoom-in (1 column x 3 rows)...")
    fig, axes = plt.subplots(3, 1, figsize=(7, 18), dpi=dpi)

    for ax, city in zip(axes, CITIES):
        w_city = walksheds_gdf[walksheds_gdf["city"] == city]
        s_city = stats_gdf[stats_gdf["city"] == city]

        # Convert WGS84 zoom box to Web Mercator
        zoom_box_3857 = gpd.GeoSeries([ZOOM_BOUNDS_WGS84[city]], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
        minx, miny, maxx, maxy = zoom_box_3857.bounds

        # Plot walksheds with enhanced boundary contrast
        w_city.plot(
            ax=ax,
            facecolor="#2B5C8F",
            edgecolor="#102A45",
            alpha=0.35,
            linewidth=1.2,
            zorder=2
        )

        # Plot station point markers with distinct borders
        s_city.plot(
            ax=ax,
            color="#D9381E",
            markersize=35,
            edgecolor="black",
            linewidth=0.8,
            zorder=4
        )

        # Set zoom extents
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Add street-level Contextily basemap
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron, alpha=0.90, zorder=1)

        # Set ax title (strictly city name only) and clean formatting
        ax.set_title(CITY_TITLES[city], fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Clean layout without figure suptitle
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved Option 2 to: {output_path}")


def create_option3_temperature_choropleth(walksheds_gdf, stats_gdf, output_path="walksheds_option3_temperature.png", dpi=300):
    """
    Option 3: Walksheds choropleth-colored by zonal median surface temperature (°C).
    1-column, 3-panel vertical figure with shared temperature colorbar and city names as ax titles.
    """
    print("Generating Option 3: Walkshed Temperature Exposure Choropleth (1 column x 3 rows)...")
    vmin, vmax = 30.0, 48.0
    cmap = "YlOrRd"

    fig, axes = plt.subplots(3, 1, figsize=(7, 18), dpi=dpi)

    for ax, city in zip(axes, CITIES):
        w_city = walksheds_gdf[walksheds_gdf["city"] == city]
        s_city = stats_gdf[stats_gdf["city"] == city]

        # Plot walksheds choropleth by median temperature
        w_city.plot(
            column="median_temp",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.65,
            zorder=2
        )

        # Plot station locations
        s_city.plot(
            ax=ax,
            color="black",
            markersize=12,
            zorder=3
        )

        # Adjust spatial extents
        minx, miny, maxx, maxy = w_city.total_bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        # Add Contextily basemap
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron, alpha=0.85, zorder=1)

        # Set ax title (strictly city name only) and clean formatting
        ax.set_title(CITY_TITLES[city], fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Add shared horizontal colorbar at the bottom
    cax = fig.add_axes([0.20, 0.04, 0.60, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Walkshed Median Surface Temperature (°C)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Adjust layout without suptitle
    plt.subplots_adjust(top=0.96, bottom=0.08, hspace=0.15)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved Option 3 to: {output_path}")


def create_option4_temperature_row(walksheds_gdf, stats_gdf, output_path="walksheds_option4_temperature_row.png", dpi=300):
    """
    Option 4: Walksheds choropleth-colored by zonal median surface temperature (°C).
    1-row, 3-column horizontal design with shared bottom colorbar and city names as ax titles.
    """
    print("Generating Option 4: Walkshed Temperature Exposure Choropleth (1 row x 3 columns)...")
    vmin, vmax = 30.0, 48.0
    cmap = "YlOrRd"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=dpi)

    for ax, city in zip(axes, CITIES):
        w_city = walksheds_gdf[walksheds_gdf["city"] == city]
        s_city = stats_gdf[stats_gdf["city"] == city]

        # Plot walksheds choropleth by median temperature
        w_city.plot(
            column="median_temp",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.65,
            zorder=2
        )

        # Plot station locations
        s_city.plot(
            ax=ax,
            color="black",
            markersize=12,
            zorder=3
        )

        # Adjust spatial extents with 5% margin
        minx, miny, maxx, maxy = w_city.total_bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        # Add Contextily basemap
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron, alpha=0.85, zorder=1)

        # Set ax title (strictly city name only) and clean formatting
        ax.set_title(CITY_TITLES[city], fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Add shared horizontal colorbar across the bottom
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Walkshed Median Surface Temperature (°C)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Adjust layout without suptitle
    plt.subplots_adjust(top=0.92, bottom=0.18, wspace=0.12)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved Option 4 to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-city walkshed publication maps.")
    parser.add_argument("--option", type=str, default="all", choices=["1", "2", "3", "4", "all"],
                        help="Select option to generate (1=Overview 1x3, 2=Zoomed 1x3, 3=Temp 1x3, 4=Temp 1x3 Row, all=All)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution in dots per inch (default: 300)")
    args = parser.parse_args()

    # Load geospatial datasets
    walksheds_gdf, stats_gdf = load_datasets()

    # Generate selected options
    if args.option in ["1", "all"]:
        create_option1_full_overview(walksheds_gdf, stats_gdf, dpi=args.dpi)

    if args.option in ["2", "all"]:
        create_option2_zoomed_core(walksheds_gdf, stats_gdf, dpi=args.dpi)

    if args.option in ["3", "all"]:
        create_option3_temperature_choropleth(walksheds_gdf, stats_gdf, dpi=args.dpi)

    if args.option in ["4", "all"]:
        create_option4_temperature_row(walksheds_gdf, stats_gdf, dpi=args.dpi)

    print("\nAll requested walkshed map options generated successfully!")


if __name__ == "__main__":
    main()
