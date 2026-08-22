# Transit Stations Heat Exposure Analysis: Agent Guidelines (`AGENTS.md`)

Welcome! This workspace contains the code, data, and presentation assets for an urban environmental equity study analyzing **pedestrian-scale heat exposure at rapid transit stations** across **New York City (MTA)**, **Chicago (CTA)**, and **Washington, D.C. (WMATA)**.

---

## 1. Quick Project Context

- **Objective**: Measure and compare surface temperature exposure in 10-minute pedestrian walksheds surrounding transit stations, assessing correlation with urban morphology (building height, plan area fraction), impervious surfaces, and neighborhood demographics.
- **Key Metro Areas**:
  - **NYC**: MTA Subway stations across 5 boroughs.
  - **Chicago**: CTA 'L' rapid transit stations.
  - **DC**: WMATA Metrorail stations across DC, MD, and VA.
- **Primary Data Sources**:
  - **Landsat 8 & 9 ARD (Band 10 & QA_PIXEL)**: Summer surface temperature and clear-sky masking.
  - **OpenStreetMap via OSMnx**: 10-minute network walksheds ($750\text{ m}$ at $1.25\text{ m/s}$).
  - **uMORPH**: 3D urban canopy morphology parameters (building height and plan area fraction).
  - **USGS NLCD (2024)**: Fractional impervious surface rasters.
  - **EPA Smart Location Database (SLD)**: Road and intersection network density (`D3A`, `D3APO`).
  - **US Census ACS (5-year)**: Socioeconomic and commute mode indicators.

---

## 2. Directory Layout & Key Files

| Directory / File | Description |
| :--- | :--- |
| [`generate_dashboard.py`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/generate_dashboard.py) | **Main Figure Generator**: Produces 3×3 figure ([`combined_city_temperature_analysis.png`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/combined_city_temperature_analysis.png)) & 3×2 figure ([`combined_city_temperature_analysis_subset.png`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/combined_city_temperature_analysis_subset.png)). |
| [`walkshed_analysis_st.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/walkshed_analysis_st.ipynb) | Computes 10-min walksheds, extracts zonal LST stats, and outputs Folium map ([`station_temperature_map.html`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/station_temperature_map.html)). |
| [`combine_lst.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/combine_lst.ipynb) / [`test_tile_combine.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/test_tile_combine.ipynb) | QA filtering, temporal averaging, and scene stitching into `stitched_city_temperatures/`. |
| [`subset_lst.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/subset_lst.ipynb) | Crops stitched LST rasters to station bounding boxes (`*_subset_surface_temperature.tif`). |
| [`lst_urbanparameters.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/lst_urbanparameters.ipynb) | Zonal analysis joining LST with uMORPH building heights, plan area fractions, and EPA SLD metrics. |
| [`download_acs.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/download_acs.ipynb) & [`combine_bg_indicators.ipynb`](file:///c:/Users/lortizur/Documents/projects/Transit-Stations-Heat/combine_bg_indicators.ipynb) | Census ACS API queries and block group GIS merges (`blockgroups_indicators.geojson`). |
| `metrolocs/` | Station location GeoJSON point layers: `NYC.geojson`, `Chicago.geojson`, `DC.geojson`. |
| `stitched_city_temperatures/` | Multi-temporal stitched and cropped GeoTIFFs per city. |
| `station_temperature_statistics*.geojson` | Primary processed datasets with temperature distributions per station walkshed. |
| `AGU2025/` | Conference poster, presentation slide decks, and abstract. |
| `.agents/skills/transit-heat-pipeline/` | Custom workspace skill containing runbooks and data dictionaries. |
| `.agents/skills/scientific-figure-design/` | Mandatory skill for scientific figure design, decluttering, and best practices. |

---

## 3. Strict Rules & Conventions

### 1. Temperature Calculation Formula
$$T(^\circ\text{C}) = (\text{DN} \times 0.00341802 + 149.0) - 273.15$$
Never use raw DN values or Kelvin in user-facing plots or statistics. All figures and outputs are in **Degrees Celsius (°C)**.

### 2. Quality Flag Bitmasking
When processing Landsat `QA_PIXEL` rasters, cast to `np.uint16` and filter out:
- Dilated cloud (Bit 1), Cirrus (Bit 2), Cloud (Bit 3), Cloud Shadow (Bit 4), Snow (Bit 5), Water (Bit 7).
- High confidence flags for cloud/shadow/cirrus (Bits 8-15).

### 3. Coordinate Reference Systems (CRS)
- **Calculations/Buffers**: Local UTM (`EPSG:32618` for NYC & DC, `EPSG:32616` for Chicago). Always auto-detect using `gdf.estimate_utm_crs()`.
- **Basemaps (Contextily)**: `EPSG:3857` (Web Mercator).
- **GeoJSON Output Files**: `EPSG:4326` (WGS84).

### 4. OSMnx Performance & API Etiquette
- Always enable caching: `ox.settings.use_cache = True` and `ox.settings.cache_folder = 'cache'`.
- Process cities sequentially to avoid network throttling and excessive memory usage.

### 5. Multi-Panel Figure Layouts
- Use fixed square aspect ratio `ax.set_box_aspect(1.0)` for maps and bar charts.
- Append marginal histograms/distributions using `fig.add_axes()` with coordinates derived from `ax.get_position()` **after** calling `fig.tight_layout()` to preserve strict panel alignments.

### 6. Mandatory Scientific Figure Design Standard
- Whenever creating, modifying, or reviewing any scientific figure, plot, map, or dashboard, you **MUST** automatically activate and apply the `scientific-figure-design` skill (`.agents/skills/scientific-figure-design/SKILL.md`).
- Strictly adhere to the synthesis of **Rougier et al. (PLOS Comput Biol 2014)** and **Simplified Science Publishing**:
  - Message trumps beauty; eliminate chartjunk (no heavy borders, soften gridlines to `#E5E8E8`).
  - Use 1–2 intentional accent colors on a neutral/slate background base.
  - Never use rainbow/jet colormaps; verify colorblind accessibility.
  - Pass the "Hidden Text" test and provide self-sufficient figure captions defining all visual encodings.

### 7. Dedicated Local Environment (`./env`)
- The workspace uses a dedicated local Conda/Mamba prefix environment located at `./env` built using Miniforge (`C:\Users\lortizur\AppData\Local\miniforge-pypy3`).
- To run analysis scripts using the environment:
  ```powershell
  & "C:\Users\lortizur\AppData\Local\miniforge-pypy3\condabin\conda.bat" run -p ./env python <script_name.py>
  ```
- Or activate via:
  ```powershell
  & "C:\Users\lortizur\AppData\Local\miniforge-pypy3\condabin\conda.bat" activate C:\Users\lortizur\Documents\projects\Transit-Stations-Heat\env
  ```


