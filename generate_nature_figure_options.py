#!/usr/bin/env python3
"""
generate_nature_figure_options.py
=================================
Generates 5 publication-quality figure options designed for Nature journals
(e.g., Nature Cities, Nature Sustainability, Nature Communications) summarizing
the transit station walkshed heat exposure and urban morphology analysis across
New York City (MTA), Chicago (CTA), and Washington, D.C. (WMATA).

Adheres strictly to the `scientific-figure-design` skill standards:
- 300 DPI, Nature standard double-column width (~180 mm / 7.1 in)
- Typology-differentiated palette (High-Rise Core: #2C3E50, Mid-Rise: #16A085, Low-Rise: #E74C3C)
- City glyphs (NYC: Circle, Chicago: Square, DC: Diamond)
- Softened grids (#E5E8E8), clean spines, sans-serif typography, self-sufficient captions
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy import stats
from scipy.interpolate import griddata
import rasterio
from rasterstats import zonal_stats

warnings.filterwarnings('ignore')

# ------------------ Universal Publication Styling ------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.edgecolor': '#212121',
    'axes.linewidth': 0.8,
    'axes.labelsize': 8.0,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04
})

# Color & Glyph Constants
PALETTE = {
    'High-Rise Core': '#2C3E50',     # Slate Blue / Charcoal
    'Mid-Rise Dense': '#16A085',     # Teal / Sea Green
    'Open Low-Rise': '#E74C3C',      # Warm Coral / Red
    'All Stations': '#78909C'        # Slate Gray
}

CITY_MARKERS = {
    'NYC': 'o',
    'Chicago': 's',
    'DC': 'D'
}

CITY_LABELS = {
    'NYC': 'New York City (MTA)',
    'Chicago': 'Chicago (CTA)',
    'DC': 'Washington, D.C. (WMATA)'
}

# ------------------ Data Preparation Pipeline ------------------
def load_and_prepare_data():
    """Load stations, walksheds, network density, and extract NLCD imperviousness."""
    print("Loading and preparing spatial datasets...")
    
    # 1. Load station temperature stats
    stations = gpd.read_file('station_temperature_statistics.geojson')
    
    # 2. Load Network Density & match to stations via nearest spatial join
    nd = gpd.read_file('Network Density/Walksheds_with_NetworkDensity.shp')[['D3A', 'D3APO', 'D3B', 'geometry']]
    stations_proj = stations.to_crs(epsg=3857)
    nd_proj = nd.to_crs(epsg=3857)
    
    joined_nd = gpd.sjoin_nearest(stations_proj, nd_proj, how='left')
    joined_nd = joined_nd[~joined_nd.index.duplicated(keep='first')]
    
    stations['D3A'] = joined_nd['D3A'].values
    stations['D3APO'] = joined_nd['D3APO'].values
    stations['D3B'] = joined_nd['D3B'].values
    
    # 3. Extract NLCD Imperviousness from rasters
    nlcd_files = {
        'NYC': 'NLCD_imperviousFraction/NYC/Annual_NLCD_FctImp_2024_CU_C1V1_ffd3767f-a804-4fd8-a912-5d2ef49a1bd7.tiff',
        'Chicago': 'NLCD_imperviousFraction/Chicago/Annual_NLCD_FctImp_2024_CU_C1V1_fcd09b81-feca-461d-aeed-c756d4ca7fba.tiff',
        'DC': 'NLCD_imperviousFraction/DC/Annual_NLCD_FctImp_2024_CU_C1V1_ebeafd14-d002-4b47-b967-84c63d6f7b1e.tiff'
    }
    
    walksheds = gpd.read_file('all_walksheds_10min.geojson')
    imp_means = []
    
    for idx, row in walksheds.iterrows():
        city = row['city']
        rpath = nlcd_files[city]
        with rasterio.open(rpath) as src:
            poly_proj = gpd.GeoSeries([row.geometry], crs=walksheds.crs).to_crs(src.crs)
            st = zonal_stats(poly_proj, rpath, stats=['mean'])
            imp_means.append(st[0]['mean'] if st[0]['mean'] is not None else np.nan)
            
    stations['impervious_fraction'] = imp_means
    
    # Fill any sparse NLCD NaNs with city median
    stations['impervious_fraction'] = stations.groupby('city')['impervious_fraction'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # 4. Morphological Archetype Classification (Tercile Composite Score)
    stations['norm_d3a'] = stations.groupby('city')['D3A'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
    stations['norm_d3apo'] = stations.groupby('city')['D3APO'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
    stations['norm_imp'] = stations.groupby('city')['impervious_fraction'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
    
    stations['built_score'] = 0.35 * stations['norm_d3a'] + 0.35 * stations['norm_d3apo'] + 0.30 * stations['norm_imp']
    
    # Assign archetypes by terciles per city
    def assign_city_terciles(df_city):
        q33 = df_city['built_score'].quantile(0.333)
        q67 = df_city['built_score'].quantile(0.667)
        def label(val):
            if val >= q67: return 'High-Rise Core'
            elif val >= q33: return 'Mid-Rise Dense'
            else: return 'Open Low-Rise'
        return df_city['built_score'].apply(label)
        
    stations['archetype'] = stations.groupby('city', group_keys=False).apply(assign_city_terciles)
    
    # Derived metric: Intra-walkshed thermal range (IQR and 90th-10th)
    stations['intra_range_90_10'] = stations['temp_90th'] - stations['temp_10th']
    stations['intra_iqr'] = stations['temp_75th'] - stations['temp_25th']
    stations['compactness_index'] = (stations['impervious_fraction'] * stations['D3APO']) / 100.0
    
    print(f"Dataset prepared: {len(stations)} stations.")
    print("Archetype distribution:\n", pd.crosstab(stations['city'], stations['archetype']))
    return stations, walksheds


# Helper to style cartesian panels cleanly
def style_panel(ax, panel_tag, xlim=None, ylim=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=0.5)
    ax.set_axisbelow(True)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.text(0.95, 0.05, f"({panel_tag})", transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', ha='right', va='bottom', zorder=10)


def create_common_legend(fig, handles=None, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=6):
    if handles is None:
        handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#2C3E50', markeredgecolor='black', markersize=6, label='High-Rise Core'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#16A085', markeredgecolor='black', markersize=6, label='Mid-Rise Dense'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#E74C3C', markeredgecolor='black', markersize=6, label='Open Low-Rise'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#78909C', markeredgecolor='black', markersize=5, label='NYC (MTA)'),
            Line2D([0], [0], marker='s', color='none', markerfacecolor='#78909C', markeredgecolor='black', markersize=5, label='Chicago (CTA)'),
            Line2D([0], [0], marker='D', color='none', markerfacecolor='#78909C', markeredgecolor='black', markersize=5, label='DC (WMATA)')
        ]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor,
               ncol=ncol, frameon=False, fontsize=7.5)


# =====================================================================
# Option 1: The 3D Morphological Triad (2x3 Grid, 6 Panels)
# =====================================================================
def generate_option1(df):
    print("Generating Option 1: The 3D Morphological Triad...")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), dpi=300)
    
    # (a) LST vs Impervious Fraction
    ax = axes[0, 0]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['impervious_fraction'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.75, edgecolors='black', linewidths=0.3)
    
    sns.regplot(data=df, x='impervious_fraction', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5, 'linestyle': '-'})
    ax.set_xlabel('Impervious Surface Fraction (%)')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'a', xlim=(20, 100), ylim=(22, 54))
    
    # (b) LST vs Road Network Density (D3A)
    ax = axes[0, 1]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['D3A'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.75, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='D3A', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5, 'linestyle': '-'})
    ax.set_xlabel('Road Network Density (link mi/mi²)')
    ax.set_ylabel('')
    style_panel(ax, 'b', xlim=(0, 65), ylim=(22, 54))
    
    # (c) LST vs Pedestrian Intersection Density (D3APO)
    ax = axes[0, 2]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['D3APO'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.75, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='D3APO', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5, 'linestyle': '-'})
    ax.set_xlabel('Pedestrian Intersections (D3APO, /mi²)')
    ax.set_ylabel('')
    style_panel(ax, 'c', xlim=(0, 40), ylim=(22, 54))
    
    # (d) Cross-City Typology Boxplots
    ax = axes[1, 0]
    archetype_order = ['High-Rise Core', 'Mid-Rise Dense', 'Open Low-Rise']
    sns.boxplot(data=df, x='city', y='median_temp', hue='archetype', hue_order=archetype_order,
                palette=[PALETTE[k] for k in archetype_order], ax=ax,
                fliersize=1.5, linewidth=0.8, width=0.65)
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'd', ylim=(22, 54))
    
    # (e) 2D Bivariate Morphological Surface (Imperviousness vs D3APO)
    ax = axes[1, 1]
    hb = ax.hexbin(df['impervious_fraction'], df['D3APO'], C=df['median_temp'],
                   gridsize=18, cmap='YlOrRd', mincnt=1, vmin=28, vmax=48, edgecolors='#E5E8E8', linewidths=0.3)
    ax.set_xlabel('Impervious Surface Fraction (%)')
    ax.set_ylabel('Pedestrian Intersections (D3APO)')
    style_panel(ax, 'e', xlim=(20, 100), ylim=(0, 40))
    
    # (f) Top Warmest & Coolest Station Rankings
    ax = axes[1, 2]
    top_hot = df.sort_values(by='median_temp', ascending=False).head(5)
    top_cool = df.sort_values(by='median_temp', ascending=True).head(5)
    extremes = pd.concat([top_cool, top_hot]).sort_values(by='median_temp', ascending=True)
    
    y_pos = np.arange(len(extremes))
    err_low = (extremes['median_temp'] - extremes['temp_25th']).clip(lower=0).values
    err_high = (extremes['temp_75th'] - extremes['median_temp']).clip(lower=0).values
    colors = [PALETTE[a] for a in extremes['archetype']]
    
    ax.barh(y_pos, extremes['median_temp'], xerr=[err_low, err_high],
            color=colors, height=0.55, edgecolor='black', linewidth=0.4,
            error_kw=dict(elinewidth=0.7, capthick=0.7, capsize=1.8, ecolor='#212121'))
    
    clean_names = [f"{n[:10]} ({c})" for n, c in zip(extremes['station_name'], extremes['city'])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=6.2)
    ax.set_xlabel('Median LST (°C) [IQR Error]')
    style_panel(ax, 'f', xlim=(20, 55))
    
    plt.tight_layout()
    create_common_legend(fig, bbox_to_anchor=(0.5, -0.01))
    
    out_path = 'option1_morphological_triad.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Option 2: The Microclimate Archetype Continuum (2x2 Grid, 4 Panels)
# =====================================================================
def generate_option2(df):
    print("Generating Option 2: The Microclimate Archetype Continuum...")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=300)
    
    # (a) Morphological Phase Space & Heat Contours
    ax = axes[0, 0]
    xi = np.linspace(20, 100, 100)
    yi = np.linspace(0, 40, 100)
    XI, YI = np.meshgrid(xi, yi)
    
    valid = df.dropna(subset=['impervious_fraction', 'D3APO', 'median_temp'])
    ZI = griddata((valid['impervious_fraction'], valid['D3APO']), valid['median_temp'], (XI, YI), method='linear')
    
    ax.contourf(XI, YI, ZI, levels=12, cmap='YlOrRd', alpha=0.35, vmin=25, vmax=50)
    
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['impervious_fraction'], csub['D3APO'],
                       color=c, marker=m, s=16, alpha=0.85, edgecolors='black', linewidths=0.3)
            
    ax.set_xlabel('Impervious Surface Fraction (%)')
    ax.set_ylabel('Pedestrian Intersections (D3APO, /mi²)')
    style_panel(ax, 'a', xlim=(20, 100), ylim=(0, 40))
    
    # (b) Cross-City Typology Box & Strip Plots
    ax = axes[0, 1]
    archetype_order = ['High-Rise Core', 'Mid-Rise Dense', 'Open Low-Rise']
    sns.boxplot(data=df, x='archetype', y='median_temp', order=archetype_order,
                palette=[PALETTE[k] for k in archetype_order], ax=ax,
                linewidth=0.8, width=0.55, fliersize=0)
    
    # Overlay strip plot
    city_palette = {'NYC': '#2C3E50', 'Chicago': '#D35400', 'DC': '#16A085'}
    sns.stripplot(data=df, x='archetype', y='median_temp', order=archetype_order,
                  hue='city', palette=city_palette,
                  dodge=True, size=2.8, alpha=0.5, jitter=0.2, ax=ax)
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'b', ylim=(22, 54))
    
    # (c) Network Density Modulation stratified by High-Rise vs Low-Rise
    ax = axes[1, 0]
    for arch in ['High-Rise Core', 'Open Low-Rise']:
        sub = df[df['archetype'] == arch]
        c = PALETTE[arch]
        ax.scatter(sub['D3APO'], sub['median_temp'], color=c, alpha=0.4, s=14, edgecolors='none')
        sns.regplot(data=sub, x='D3APO', y='median_temp', ax=ax, scatter=False,
                    color=c, line_kws={'linewidth': 1.8, 'label': arch})
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', loc='lower right', fontsize=6.8)
    ax.set_xlabel('Pedestrian Intersections (D3APO, /mi²)')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'c', xlim=(0, 40), ylim=(22, 54))
    
    # (d) Intra-Walkshed Microclimate Variance (IQR) vs Imperviousness
    ax = axes[1, 1]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['impervious_fraction'], csub['intra_iqr'],
                       color=c, marker=m, s=14, alpha=0.75, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='impervious_fraction', y='intra_iqr', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Impervious Surface Fraction (%)')
    ax.set_ylabel('Walkshed Thermal Variance (IQR, °C)')
    style_panel(ax, 'd', xlim=(20, 100), ylim=(0, 14))
    
    plt.tight_layout()
    create_common_legend(fig, bbox_to_anchor=(0.5, -0.01))
    
    out_path = 'option2_archetype_continuum.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Option 3: The Dual-Axis Morphology & Connectivity Quadrant (2x2 Grid)
# =====================================================================
def generate_option3(df):
    print("Generating Option 3: The Dual-Axis Morphology & Connectivity Quadrant...")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=300)
    
    # (a) Surface Imperviousness vs Median LST with Quantile Regressions
    ax = axes[0, 0]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['impervious_fraction'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    
    x_vals = np.linspace(25, 98, 100)
    slope, intercept, r, p, se = stats.linregress(df['impervious_fraction'], df['median_temp'])
    ax.plot(x_vals, intercept + slope * x_vals, color='black', linewidth=1.5, label='Median Fit (50th)')
    
    q10 = df['temp_10th'].dropna()
    s10, i10, _, _, _ = stats.linregress(df.loc[q10.index, 'impervious_fraction'], q10)
    ax.plot(x_vals, i10 + s10 * x_vals, color='#78909C', linestyle='--', linewidth=1.2, label='10th %ile')
    
    q90 = df['temp_90th'].dropna()
    s90, i90, _, _, _ = stats.linregress(df.loc[q90.index, 'impervious_fraction'], q90)
    ax.plot(x_vals, i90 + s90 * x_vals, color='#C0392B', linestyle='--', linewidth=1.2, label='90th %ile')
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', fontsize=6.5, loc='upper left')
    ax.set_xlabel('Impervious Surface Fraction (%)')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'a', xlim=(20, 100), ylim=(20, 56))
    
    # (b) Road Density (D3A) vs Median LST
    ax = axes[0, 1]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['D3A'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='D3A', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Road Network Density (D3A, link mi/mi²)')
    ax.set_ylabel('')
    style_panel(ax, 'b', xlim=(0, 65), ylim=(20, 56))
    
    # (c) Pedestrian Intersection Density (D3APO) vs Median LST
    ax = axes[1, 0]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['D3APO'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='D3APO', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Pedestrian Intersections (D3APO, /mi²)')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'c', xlim=(0, 40), ylim=(20, 56))
    
    # (d) Cross-City Typology Synthesis (Grouped Mean & 95% CI)
    ax = axes[1, 1]
    summary = df.groupby(['city', 'archetype'])['median_temp'].agg(
        mean='mean',
        std='std',
        count='count',
        sem=lambda x: stats.sem(x)
    ).reset_index()
    summary['ci95'] = summary['sem'] * 1.96
    
    y_indices = []
    y_labels = []
    current_y = 0
    
    for city in ['NYC', 'Chicago', 'DC']:
        c_df = summary[summary['city'] == city]
        for arch in ['High-Rise Core', 'Mid-Rise Dense', 'Open Low-Rise']:
            r = c_df[c_df['archetype'] == arch]
            if not r.empty:
                val = r['mean'].values[0]
                ci = r['ci95'].values[0]
                ax.errorbar(val, current_y, xerr=ci, fmt=CITY_MARKERS[city],
                            color=PALETTE[arch], ecolor='black', elinewidth=1.0, capsize=3,
                            markersize=6, markeredgecolor='black', markeredgewidth=0.5)
                y_labels.append(f"{city} - {arch.split()[0]}")
                y_indices.append(current_y)
                current_y += 1
        current_y += 0.5
        
    ax.set_yticks(y_indices)
    ax.set_yticklabels(y_labels, fontsize=6.8)
    ax.set_xlabel('Mean Median LST (95% CI, °C)')
    ax.set_ylabel('')
    style_panel(ax, 'd', xlim=(28, 48), ylim=(-0.5, current_y - 0.5))
    
    plt.tight_layout()
    create_common_legend(fig, bbox_to_anchor=(0.5, -0.01))
    
    out_path = 'option3_morphology_connectivity_quadrant.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Option 4: The Spatial-to-Morphology Synthesis Dashboard (3x2 Grid)
# =====================================================================
def generate_option4(df, walksheds):
    print("Generating Option 4: The Spatial-to-Morphology Synthesis Dashboard...")
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8), dpi=300)
    
    # (a, b, c) Spatial Walkshed Maps for NYC, Chicago, DC
    for i, city in enumerate(['NYC', 'Chicago', 'DC']):
        ax_map = axes[i, 0]
        c_stations = df[df['city'] == city].to_crs(epsg=3857)
        c_walksheds = walksheds[walksheds['city'] == city].to_crs(epsg=3857)
        
        c_walksheds.plot(ax=ax_map, facecolor='#F5F5F5', edgecolor='#B0BEC5', linewidth=0.4, alpha=0.6)
        
        for arch, c in PALETTE.items():
            if arch == 'All Stations': continue
            sub = c_stations[c_stations['archetype'] == arch]
            sizes = ((sub['median_temp'] - 20) / 1.8).clip(lower=4, upper=28)
            ax_map.scatter(sub.geometry.x, sub.geometry.y, color=c, s=sizes,
                           alpha=0.85, edgecolors='black', linewidths=0.3, zorder=4)
            
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for sp in ax_map.spines.values(): sp.set_color('black'); sp.set_linewidth(0.8)
        
        ax_map.text(0.04, 0.94, CITY_LABELS[city], transform=ax_map.transAxes,
                    fontsize=7.8, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='#E5E8E8', alpha=0.9))
        
        panel_tag = chr(ord('a') + i)
        ax_map.text(0.95, 0.05, f"({panel_tag})", transform=ax_map.transAxes,
                    fontsize=9.0, fontweight='bold', ha='right', va='bottom', zorder=10)
        
    # (d) Built Density Phase Space (Impervious vs D3APO) colored by LST
    ax_d = axes[0, 1]
    sc = ax_d.scatter(df['impervious_fraction'], df['D3APO'], c=df['median_temp'],
                      cmap='YlOrRd', s=16, alpha=0.85, edgecolors='black', linewidths=0.3, vmin=25, vmax=50)
    ax_d.set_xlabel('Impervious Surface Fraction (%)')
    ax_d.set_ylabel('Pedestrian Intersections (D3APO)')
    style_panel(ax_d, 'd', xlim=(20, 100), ylim=(0, 40))
    
    # (e) Pedestrian Connectivity vs Median LST by Archetype
    ax_e = axes[1, 1]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        ax_e.scatter(sub['D3APO'], sub['median_temp'], color=c, alpha=0.4, s=14, edgecolors='black', linewidths=0.2)
        sns.regplot(data=sub, x='D3APO', y='median_temp', ax=ax_e, scatter=False,
                    color=c, line_kws={'linewidth': 1.6, 'label': arch})
    ax_e.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', fontsize=6.5, loc='upper left')
    ax_e.set_xlabel('Pedestrian Intersections (D3APO, /mi²)')
    ax_e.set_ylabel('Median LST (°C)')
    style_panel(ax_e, 'e', xlim=(0, 40), ylim=(22, 54))
    
    # (f) Probability Density Profiles (KDE) by Archetype
    ax_f = axes[2, 1]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]['median_temp'].dropna()
        if len(sub) > 2:
            sns.kdeplot(sub, ax=ax_f, color=c, fill=True, alpha=0.3, linewidth=1.5, label=arch)
    ax_f.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', fontsize=6.5, loc='upper left')
    ax_f.set_xlabel('Median LST (°C)')
    ax_f.set_ylabel('Density')
    style_panel(ax_f, 'f', xlim=(20, 54))
    
    plt.tight_layout()
    create_common_legend(fig, bbox_to_anchor=(0.5, -0.01))
    
    out_path = 'option4_spatial_morphology_dashboard.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Option 5: The Urban Canyon & Thermal Mass Mechanism Matrix (2x3 Grid)
# =====================================================================
def generate_option5(df):
    print("Generating Option 5: The Urban Canyon & Thermal Mass Mechanism Matrix...")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), dpi=300)
    
    # (a) Impervious Thermal Mass
    ax = axes[0, 0]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['impervious_fraction'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='impervious_fraction', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Impervious Surface Fraction (%)')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'a', xlim=(20, 100), ylim=(22, 54))
    
    # (b) Network Density Impact (D3A)
    ax = axes[0, 1]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['D3A'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='D3A', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Road Network Density (D3A, link mi/mi²)')
    ax.set_ylabel('')
    style_panel(ax, 'b', xlim=(0, 65), ylim=(22, 54))
    
    # (c) Composite Built Form Index (Imperviousness * D3APO / 100)
    ax = axes[0, 2]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['compactness_index'], csub['median_temp'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='compactness_index', y='median_temp', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Built Compactness Index (Imp × D3APO / 100)')
    ax.set_ylabel('')
    style_panel(ax, 'c', xlim=(0, 40), ylim=(22, 54))
    
    # (d) Pedestrian Connectivity by City
    ax = axes[1, 0]
    city_colors = {'NYC': '#2C3E50', 'Chicago': '#D35400', 'DC': '#16A085'}
    for city in ['NYC', 'Chicago', 'DC']:
        sub = df[df['city'] == city]
        ax.scatter(sub['D3APO'], sub['median_temp'], color=city_colors[city],
                   marker=CITY_MARKERS[city], s=12, alpha=0.5, edgecolors='none')
        sns.regplot(data=sub, x='D3APO', y='median_temp', ax=ax, scatter=False,
                    color=city_colors[city], line_kws={'linewidth': 1.5, 'label': city})
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E5E8E8', fontsize=6.5, loc='upper left')
    ax.set_xlabel('Pedestrian Intersections (D3APO, /mi²)')
    ax.set_ylabel('Median LST (°C)')
    style_panel(ax, 'd', xlim=(0, 40), ylim=(22, 54))
    
    # (e) Cross-City Ridge Distributions
    ax = axes[1, 1]
    y_pos = 0
    y_ticks = []
    y_labels = []
    for city in ['NYC', 'Chicago', 'DC']:
        c_df = df[df['city'] == city]
        for arch in ['High-Rise Core', 'Mid-Rise Dense', 'Open Low-Rise']:
            sub = c_df[c_df['archetype'] == arch]['median_temp'].dropna()
            if len(sub) > 3:
                kde = stats.gaussian_kde(sub)
                x = np.linspace(22, 52, 200)
                y_dens = kde(x) * 1.5
                ax.fill_between(x, y_pos, y_pos + y_dens, color=PALETTE[arch], alpha=0.6, edgecolor='black', linewidth=0.5)
                y_ticks.append(y_pos + 0.2)
                y_labels.append(f"{city}-{arch[:4]}")
                y_pos += 0.8
        y_pos += 0.4
        
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6.2)
    ax.set_xlabel('Median LST (°C)')
    style_panel(ax, 'e', xlim=(20, 54), ylim=(-0.2, y_pos))
    
    # (f) Walkshed Thermal Range (90th - 10th) vs Compactness Index
    ax = axes[1, 2]
    for arch, c in PALETTE.items():
        if arch == 'All Stations': continue
        sub = df[df['archetype'] == arch]
        for city, m in CITY_MARKERS.items():
            csub = sub[sub['city'] == city]
            ax.scatter(csub['compactness_index'], csub['intra_range_90_10'],
                       color=c, marker=m, s=14, alpha=0.7, edgecolors='black', linewidths=0.3)
    sns.regplot(data=df, x='compactness_index', y='intra_range_90_10', ax=ax,
                scatter=False, color='#212121', line_kws={'linewidth': 1.5})
    ax.set_xlabel('Built Compactness Index')
    ax.set_ylabel('Intra-Walkshed Range (T90 - T10, °C)')
    style_panel(ax, 'f', xlim=(0, 40), ylim=(0, 20))
    
    plt.tight_layout()
    create_common_legend(fig, bbox_to_anchor=(0.5, -0.01))
    
    out_path = 'option5_urban_canyon_mechanism_matrix.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
# Main Execution
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Starting Generation of 5 Nature Publication Figure Options")
    print("=" * 60)
    
    df_stations, gdf_walksheds = load_and_prepare_data()
    
    generate_option1(df_stations)
    generate_option2(df_stations)
    generate_option3(df_stations)
    generate_option4(df_stations, gdf_walksheds)
    generate_option5(df_stations)
    
    print("=" * 60)
    print("All 5 Nature Figure Options Successfully Generated!")
    print("=" * 60)
