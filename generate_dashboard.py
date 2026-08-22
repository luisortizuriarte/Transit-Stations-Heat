import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# Configuration
CONFIG = {
    "cities": ["NYC", "Chicago", "DC"],
    "paths": {
        "input_template": "stitched_city_temperatures/{city}_subset_surface_temperature.tif",
        "station_stats_template": "station_temperature_statistics_{city_lower}.geojson",
        "output_filename": "combined_city_temperature_analysis.png",
        "output_filename_subset": "combined_city_temperature_analysis_subset.png"
    },
    "vmin": 15,
    "vmax": 55,
    "top_n": 20
}

# Create figure 1 (3x3 grid) - scaled to fit perfectly on a standard page
fig, axes = plt.subplots(3, 3, figsize=(11, 11), dpi=300)
for ax in axes.flat:
    ax.set_box_aspect(1.0)
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
im_map = None
maps_marginals = []

# Create figure 2 (3x2 grid, Map and Hist columns only)
fig2, axes2 = plt.subplots(3, 2, figsize=(8, 11), dpi=300)
for ax in axes2.flat:
    ax.set_box_aspect(1.0)
labels2 = ['a', 'b', 'c', 'd', 'e', 'f']
im_map2 = None
maps_marginals2 = []

for i, city in enumerate(CONFIG["cities"]):
    city_lower = city.lower()
    
    # ------------------ Load Data ------------------
    raster_path = CONFIG["paths"]["input_template"].format(city=city)
    geojson_path = CONFIG["paths"]["station_stats_template"].format(city_lower=city_lower)
    
    # Load Raster
    with rio.open(raster_path) as src:
        raster_data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            raster_data[raster_data == src.nodata] = np.nan
        extent = plotting_extent(src)
        raster_crs = src.crs
        
    # Load GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    # Sort and get top 20 hottest stations
    # Use median_temp column to sort descending
    gdf_sorted = gdf.sort_values(by="median_temp", ascending=False)
    top20 = gdf_sorted.head(CONFIG["top_n"]).copy()
    
    # Reproject geojson geometries to match raster CRS for accurate spatial overlay
    gdf_proj = gdf.to_crs(raster_crs)
    top20_proj = top20.to_crs(raster_crs)
    
    # ------------------ Column 0: Horizontal Bar Chart (Fig 1 only) ------------------
    ax_bar = axes[i, 0]
    
    # Sort top 20 ascending so hottest stations appear at the top of the horizontal bar chart
    top20_bar = top20.sort_values(by="median_temp", ascending=True)
    
    # Calculate error boundaries around the median temp using 25th and 75th percentiles
    xerr_lower = (top20_bar["median_temp"] - top20_bar["temp_25th"]).clip(lower=0).values
    xerr_upper = (top20_bar["temp_75th"] - top20_bar["median_temp"]).clip(lower=0).values
    xerr = np.array([xerr_lower, xerr_upper])
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(top20_bar))
    bars = ax_bar.barh(
        y=y_pos,
        width=top20_bar["median_temp"],
        xerr=xerr,
        color='#78909C',
        edgecolor='#78909C',
        linewidth=0.5,
        ecolor='black',
        capsize=2.0,
        error_kw=dict(elinewidth=0.8, capthick=0.8),
        height=0.6
    )
    
    # Style and format bar charts
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top20_bar["station_name"], fontsize=7, color='black')
    ax_bar.set_xlim(35.0, 52.5)
    ax_bar.set_xticks([35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0])
    ax_bar.tick_params(axis='both', labelsize=7.5)
    ax_bar.set_xlabel('Temperature (°C)', fontsize=8.0, labelpad=3)
    
    # Frame/Spines and Grids
    for spine in ax_bar.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        spine.set_visible(True)
    ax_bar.grid(axis='x', linestyle='-', color='#E5E8E8', linewidth=0.5)
    ax_bar.set_axisbelow(True)
    
    # Custom Legend under Column 0 (bottom panel - DC) for Figure 1
    if i == 2:
        legend_element = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#7EA3B2', markeredgecolor='none', markersize=8, label='Warmest metro stations')
        ]
        ax_bar.legend(
            handles=legend_element,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            frameon=False,
            fontsize=8.0
        )
        
    # ------------------ Column 1: Map Plot ------------------
    ax_map = axes[i, 1]
    ax_map2 = axes2[i, 0]
    
    # Load plan area fraction (uMORPH) raster and align with LST raster extent
    umorph_path = "uMORPH/merged_mean_building_height_100m_wgs.tif"
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    
    with rio.open(umorph_path) as morph_src:
        morph_crs = morph_src.crs
        # Project LST bounds to morph CRS
        left_m, bottom_m, right_m, top_m = transform_bounds(
            raster_crs, morph_crs,
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        )
        window = from_bounds(left_m, bottom_m, right_m, top_m, transform=morph_src.transform)
        # Read and resample to match LST raster dimensions (height, width)
        morph_data = morph_src.read(1, window=window, out_shape=(raster_data.shape[0], raster_data.shape[1])).astype(np.float32)
        if morph_src.nodata is not None:
            morph_data[morph_data == morph_src.nodata] = np.nan
        morph_data[morph_data < 0] = np.nan

    # Compute left-to-right (column) and top-to-bottom (row) distributions of plan area fraction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        column_means = np.nanmean(morph_data, axis=0)
        row_means = np.nanmean(morph_data, axis=1)
        
    column_means = np.nan_to_num(column_means, nan=0.0)
    row_means = np.nan_to_num(row_means, nan=0.0)
    
    # Plot surface temperature raster on both figures
    im = ax_map.imshow(raster_data, extent=extent, cmap='YlOrRd', vmin=CONFIG["vmin"], vmax=CONFIG["vmax"], aspect='auto', origin='upper')
    im2 = ax_map2.imshow(raster_data, extent=extent, cmap='YlOrRd', vmin=CONFIG["vmin"], vmax=CONFIG["vmax"], aspect='auto', origin='upper')
    
    if i == 2:
        im_map = im  # Save bottom-middle image for the common colorbar of Fig 1
        im_map2 = im2  # Save bottom-middle image for the common colorbar of Fig 2
        
    # Plot all stations in black (small dots) on both figures
    ax_map.scatter(gdf_proj.geometry.x, gdf_proj.geometry.y, color='black', marker='o', s=3.0, zorder=3)
    ax_map2.scatter(gdf_proj.geometry.x, gdf_proj.geometry.y, color='black', marker='o', s=3.0, zorder=3)
    
    # Plot top 20 hottest stations in #7EA3B2 (with black edge) on both figures
    ax_map.scatter(top20_proj.geometry.x, top20_proj.geometry.y, color='#7EA3B2', marker='o', s=18.0, edgecolor='black', linewidths=0.5, zorder=4)
    ax_map2.scatter(top20_proj.geometry.x, top20_proj.geometry.y, color='#7EA3B2', marker='o', s=18.0, edgecolor='black', linewidths=0.5, zorder=4)
    
    # Overlay City Name label in upper-left corner of both map panels
    ax_map.text(0.05, 0.95, city, transform=ax_map.transAxes, fontsize=10.0, fontweight='bold', va='top', ha='left', bbox=dict(boxstyle='square,pad=0.3', facecolor='#FADBD8', alpha=0.9, edgecolor='none'))
    ax_map2.text(0.05, 0.95, city, transform=ax_map2.transAxes, fontsize=10.0, fontweight='bold', va='top', ha='left', bbox=dict(boxstyle='square,pad=0.3', facecolor='#FADBD8', alpha=0.9, edgecolor='none'))
    
    # Hide axis ticks but KEEP spines/frame visible on both maps
    for am in [ax_map, ax_map2]:
        am.set_xticks([])
        am.set_yticks([])
        for spine in am.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)
            spine.set_visible(True)
        
    # Store data for appending marginal plots manually after tight_layout finalizes positions
    maps_marginals.append((ax_map, extent, column_means, row_means))
    maps_marginals2.append((ax_map2, extent, column_means, row_means))
    
    # ------------------ Column 2: Histogram & KDE ------------------
    ax_hist = axes[i, 2]
    ax_hist2 = axes2[i, 1]
    
    all_temps = gdf["median_temp"].dropna().values
    hot_temps = top20["median_temp"].dropna().values
    
    # 1. Plot all stations distribution
    # Histogram in #78909C and KDE in black
    ax_hist.hist(all_temps, bins=15, density=True, facecolor=(120/255, 144/255, 156/255, 0.8), edgecolor='white', linewidth=0.8, label='All Stations')
    ax_hist2.hist(all_temps, bins=15, density=True, facecolor=(120/255, 144/255, 156/255, 0.8), edgecolor='white', linewidth=0.8, label='All Stations')
    
    kde_all = stats.gaussian_kde(all_temps)
    x_all = np.linspace(all_temps.min() - 3, all_temps.max() + 3, 200)
    ax_hist.plot(x_all, kde_all(x_all), color='black', linewidth=1.5)
    ax_hist2.plot(x_all, kde_all(x_all), color='black', linewidth=1.5)
    
    # 2. Plot hottest stations distribution
    # Histogram in coral/red fill with solid red outline and red KDE line
    ax_hist.hist(hot_temps, bins=8, density=True, facecolor=(231/255, 76/255, 60/255, 0.4), edgecolor='#E74C3C', linewidth=0.8, label='Top 20 Warmest')
    ax_hist2.hist(hot_temps, bins=8, density=True, facecolor=(231/255, 76/255, 60/255, 0.4), edgecolor='#E74C3C', linewidth=0.8, label='Top 20 Warmest')
    
    if len(hot_temps) > 1:
        kde_hot = stats.gaussian_kde(hot_temps)
        x_hot = np.linspace(hot_temps.min() - 3, hot_temps.max() + 3, 200)
        ax_hist.plot(x_hot, kde_hot(x_hot), color='#C0392B', linewidth=1.5)
        ax_hist2.plot(x_hot, kde_hot(x_hot), color='#C0392B', linewidth=1.5)
        
    # Style and format histograms on both figures
    for ah in [ax_hist, ax_hist2]:
        ah.set_xlim(15.0, 55.0)
        ah.set_xticks([15, 20, 25, 30, 35, 40, 45, 50, 55])
        ah.set_xlabel('Temperature (°C)', fontsize=8.0)
        ah.set_ylabel('Density', fontsize=8.0)
        ah.tick_params(labelsize=7.5)
        
        # Frame/Spines and Grids
        for spine in ah.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)
            spine.set_visible(True)
        ah.grid(True, linestyle='-', color='#E5E8E8', linewidth=0.5)
        ah.set_axisbelow(True)
    
    # Legend only in top-right panel (c) for Figure 1 and panel (b) for Figure 2
    if i == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#78909C', edgecolor='white', label='All Stations'),
            Patch(facecolor=(231/255, 76/255, 60/255, 0.4), edgecolor='#E74C3C', label='Top 20 Warmest'),
            Line2D([0], [0], color='black', linewidth=1.5, label='KDE')
        ]
        ax_hist.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', fontsize=7.5)
        ax_hist2.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', fontsize=7.5)
    
    # ------------------ Annotate Panel Letters ------------------
    # Add panel letters (a to i) to the lower-right corner of each plot in Fig 1
    for col_idx, ax in enumerate([ax_bar, ax_map, ax_hist]):
        label_idx = i * 3 + col_idx
        label = labels[label_idx]
        ax.text(0.95, 0.05, f"({label})", transform=ax.transAxes, fontsize=10.0, fontweight='bold', va='bottom', ha='right', color='black', zorder=10)

    # Add panel letters (a to f) to the lower-right corner of each plot in Fig 2
    for col_idx, ax in enumerate([ax_map2, ax_hist2]):
        label_idx = i * 2 + col_idx
        label = labels2[label_idx]
        ax.text(0.95, 0.05, f"({label})", transform=ax.transAxes, fontsize=10.0, fontweight='bold', va='bottom', ha='right', color='black', zorder=10)

# ================== Figure 1 Layout Finalization ==================
fig.tight_layout()
fig.subplots_adjust(hspace=0.18, wspace=0.35, bottom=0.15, left=0.18, right=0.96, top=0.96)

# Appending marginal plots manually using final positions to prevent any alignment shifts
for ax_map, extent, column_means, row_means in maps_marginals:
    pos = ax_map.get_position()
    pad = 0.008
    mrg_size = 0.035
    
    ax_top = fig.add_axes([pos.x0, pos.y1 + pad, pos.width, mrg_size], sharex=ax_map)
    ax_right = fig.add_axes([pos.x1 + pad, pos.y0, mrg_size, pos.height], sharey=ax_map)
    
    # Style marginal top/right axes with borders hidden
    for ax_mrg in [ax_top, ax_right]:
        for spine in ax_mrg.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)
            spine.set_visible(False)
            
    # Hide shared tick labels but keep non-shared ticks and tick labels
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_top.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=6.0, pad=2)
    ax_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_right.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=6.0, pad=2)
    
    # Add labels specifying this is building height
    ax_top.set_title('Bldg Height (m)', fontsize=6.5, pad=2)
    
    # Plot left-to-right building height distribution on the top axis as a bar chart
    x_coords = np.linspace(extent[0], extent[1], len(column_means))
    dx = (extent[1] - extent[0]) / len(column_means)
    ax_top.bar(x_coords, column_means, width=dx, color='#78909C', align='center', edgecolor='none')
    ax_top.set_ylim(0, 20)
    
    # Plot top-to-bottom building height distribution on the right axis as a horizontal bar chart
    y_coords = np.linspace(extent[3], extent[2], len(row_means))
    dy = abs(extent[3] - extent[2]) / len(row_means)
    ax_right.barh(y_coords, row_means, height=dy, color='#78909C', align='center', edgecolor='none')
    ax_right.set_xlim(0, 20)

# Common Colorbar (centered under bottom-middle map)
pos = axes[2, 1].get_position()
cax = fig.add_axes([pos.x0, pos.y0 - 0.04, pos.width, 0.015])
cbar = fig.colorbar(im_map, cax=cax, orientation='horizontal')
cbar.set_ticks([15, 20, 25, 30, 35, 40, 45, 50, 55])
cbar.ax.tick_params(labelsize=7.5)
cbar.set_label('Temperature (°C)', fontsize=8.0, fontweight='bold', labelpad=3)

# Save first figure
output_path = CONFIG["paths"]["output_filename"]
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved dashboard to: {output_path}")


# ================== Figure 2 Layout Finalization ==================
fig2.tight_layout()
fig2.subplots_adjust(hspace=0.18, wspace=0.35, bottom=0.15, left=0.15, right=0.95, top=0.96)

# Appending marginal plots manually using final positions for Figure 2
for ax_map2, extent, column_means, row_means in maps_marginals2:
    pos = ax_map2.get_position()
    pad = 0.008
    mrg_size = 0.035
    
    ax_top2 = fig2.add_axes([pos.x0, pos.y1 + pad, pos.width, mrg_size], sharex=ax_map2)
    ax_right2 = fig2.add_axes([pos.x1 + pad, pos.y0, mrg_size, pos.height], sharey=ax_map2)
    
    # Style marginal top/right axes with borders hidden
    for ax_mrg in [ax_top2, ax_right2]:
        for spine in ax_mrg.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)
            spine.set_visible(False)
            
    # Hide shared tick labels but keep non-shared ticks and tick labels
    ax_top2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_top2.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=6.0, pad=2)
    ax_right2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_right2.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=6.0, pad=2)
    
    # Add labels specifying this is building height
    ax_top2.set_title('Bldg Height (m)', fontsize=6.5, pad=2)
    
    # Plot left-to-right building height distribution on the top axis as a bar chart
    x_coords = np.linspace(extent[0], extent[1], len(column_means))
    dx = (extent[1] - extent[0]) / len(column_means)
    ax_top2.bar(x_coords, column_means, width=dx, color='#78909C', align='center', edgecolor='none')
    ax_top2.set_ylim(0, 20)
    
    # Plot top-to-bottom building height distribution on the right axis as a horizontal bar chart
    y_coords = np.linspace(extent[3], extent[2], len(row_means))
    dy = abs(extent[3] - extent[2]) / len(row_means)
    ax_right2.barh(y_coords, row_means, height=dy, color='#78909C', align='center', edgecolor='none')
    ax_right2.set_xlim(0, 10)

# Common Colorbar for Figure 2 (centered under bottom map)
pos2 = axes2[2, 0].get_position()
cax2 = fig2.add_axes([pos2.x0, pos2.y0 - 0.04, pos2.width, 0.015])
cbar2 = fig2.colorbar(im_map2, cax=cax2, orientation='horizontal')
cbar2.set_ticks([15, 20, 25, 30, 35, 40, 45, 50, 55])
cbar2.ax.tick_params(labelsize=7.5)
cbar2.set_label('Temperature (°C)', fontsize=8.0, fontweight='bold', labelpad=3)

# Custom Legend under Column 1 (bottom-right panel - DC histogram) for Figure 2
legend_element2 = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#7EA3B2', markeredgecolor='none', markersize=8, label='Warmest metro stations')
]
axes2[2, 1].legend(
    handles=legend_element2,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.22),
    frameon=False,
    fontsize=8.0
)

# Save second figure
output_path2 = CONFIG["paths"]["output_filename_subset"]
fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved subset dashboard to: {output_path2}")
