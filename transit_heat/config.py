"""
config.py
=========
Central configuration parameters, spatial coordinate reference systems (CRS),
radiometric calibration constants, and quality assessment bit flags for the
Transit Stations Heat Exposure Analysis pipeline.
"""

from typing import Dict, List, Tuple

# Supported study metropolitan regions
DEFAULT_CITIES: List[str] = ["NYC", "Chicago", "DC"]

# Coordinate Reference Systems (CRS) Mapping
# Local UTM conformal metric projections to eliminate distance distortions
CITY_UTM_CRS: Dict[str, str] = {
    "NYC": "EPSG:32618",      # UTM Zone 18N
    "Chicago": "EPSG:32616",  # UTM Zone 16N
    "DC": "EPSG:32618"        # UTM Zone 18N
}

# Standard output projection for geospatial vector publishing
OUTPUT_CRS: str = "EPSG:4326"

# Basemap projection
BASEMAP_CRS: str = "EPSG:3857"

# Landsat Collection 2 Level-2 Surface Temperature (ST Band 10) Calibration
# Formula: T(°C) = (DN * ST_SCALE_FACTOR + ST_ADD_OFFSET) - KELVIN_TO_CELSIUS_OFFSET
ST_SCALE_FACTOR: float = 0.00341802
ST_ADD_OFFSET: float = 149.0
KELVIN_TO_CELSIUS: float = 273.15

# Pedestrian Network Routing Parameters
# Standard adult urban walking velocity: 1.25 m/s (4.5 km/h)
# Catchment walking duration: 10 minutes (600 seconds)
# Maximum network reachable distance: 1.25 m/s * 600 s = 750 meters
DEFAULT_WALK_SPEED_MPS: float = 1.25
DEFAULT_TRIP_TIME_SECONDS: int = 600
DEFAULT_MAX_WALK_DISTANCE_METERS: float = DEFAULT_WALK_SPEED_MPS * DEFAULT_TRIP_TIME_SECONDS  # 750.0 m

# OSMnx Graph Fetching Parameters
NETWORK_TYPE: str = "walk"
GRAPH_BUFFER_METERS: float = 1200.0  # Buffer around stations to avoid boundary truncation
OSMNX_CACHE_FOLDER: str = "cache"

# Quality Assessment (QA_PIXEL) Bit Flags for Landsat Collection 2
# Bit 1: Dilated Cloud, Bit 2: Cirrus, Bit 3: Cloud, Bit 4: Cloud Shadow, Bit 5: Snow, Bit 7: Water
QA_EXCLUDE_BITS = {
    "dilated_cloud": 1,
    "cirrus": 2,
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
    "water": 7
}

# Default Directory Structure Conventions
DEFAULT_PATHS = {
    "stations_template": "metrolocs/{city}.geojson",
    "landsat_dir": "landsat/{city}",
    "stitched_raster_template": "stitched_city_temperatures/{city}_stitched_surface_temperature.tif",
    "subset_raster_template": "stitched_city_temperatures/{city}_subset_surface_temperature.tif",
    "walksheds_output_template": "walksheds_10min_{city_lower}.geojson",
    "walksheds_all": "all_walksheds_10min.geojson",
    "stats_output_template": "station_temperature_statistics_{city_lower}.geojson",
    "stats_all": "station_temperature_statistics.geojson"
}
