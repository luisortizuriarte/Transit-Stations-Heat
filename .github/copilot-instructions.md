# Transit Station Heat Exposure Analysis

## Project Architecture
- **Goal**: Analyze urban heat island effects near transit stations in NYC, DC, and Chicago using Landsat thermal data and OSMnx walksheds.
- **Core Pipeline**:
  1. **Temperature**: `combine_lst.ipynb` processes Landsat 8/9 ARD (QA masking -> Celsius conversion -> Temporal Mean).
  2. **Walksheds**: `walkshed_analysis_st.ipynb` generates 10-min pedestrian isochrones via OSMnx.
  3. **Integration**: `metro_locations_analysis.ipynb` combines temp, walksheds, and census data.
- **Data Flow**: Raw Landsat/GeoJSON -> Processed GeoTIFFs/GeoJSONs -> Final Analysis/Maps.

## Critical Conventions
- **CRS Strategy**:
  - **Analysis**: Local UTM (auto-detected via `estimate_utm_crs()`) for accurate distance/area.
  - **Visualization**: EPSG:3857 (Web Mercator) for contextily basemaps.
  - **Storage**: EPSG:4326 (WGS84) for GeoJSON.
  - *Always* explicitly reproject before spatial operations (clipping, joining).
- **Temperature**:
  - Units: **Celsius**.
  - Formula: `T(°C) = (DN * 0.00341802 + 149.0) - 273.15`.
  - QA Masking: Exclude Water (Bit 7), High Cloud/Shadow/Snow (Bits 8-13).
- **File Structure**:
  - `landsat/{city}/`: Raw ARD (git-ignored).
  - `stitched_city_temperatures/`: Processed thermal rasters.
  - `metrolocs/`: Station GeoJSONs.
  - `cache/`: OSMnx cache.

## Developer Workflow
- **Notebooks**: Designed for sequential execution.
- **Key Libraries**: `geopandas`, `rasterio`, `osmnx`, `networkx`, `contextily`.
- **Common Pitfalls**:
  - **Memory**: Process cities sequentially (Landsat scenes are large).
  - **OSMnx**: Use caching (`ox.settings.use_cache = True`) to avoid API limits.
  - **QA Bitmask**: Use `np.uint16` to prevent overflow when processing QA bands.
