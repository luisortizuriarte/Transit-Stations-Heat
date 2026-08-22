"""
pipeline.py
===========
End-to-end pipeline orchestrator for multi-city transit heat exposure analysis.
"""

import os
import glob
from typing import Dict, List, Optional
import geopandas as gpd
import pandas as pd

from .config import DEFAULT_CITIES, DEFAULT_PATHS
from .thermal import ThermalProcessor
from .walksheds import WalkshedGenerator
from .zonal import ZonalExtractor


class TransitHeatPipeline:
    """
    Orchestrates the three core stages of the Transit Station Heat Exposure Analysis:
    1. Thermal Radiometry (QA masking, temporal composites, tile stitching, cropping)
    2. Pedestrian Walkshed Routing (OSMnx 10-minute network isochrones)
    3. Zonal Thermal Extraction (percentile distributions per station catchment)
    """

    def __init__(self, cities: Optional[List[str]] = None,
                 walk_speed_mps: float = 1.25,
                 trip_time_seconds: int = 600):
        self.cities = cities or DEFAULT_CITIES
        self.thermal_processor = ThermalProcessor()
        self.walkshed_generator = WalkshedGenerator(
            walk_speed_mps=walk_speed_mps,
            trip_time_seconds=trip_time_seconds
        )
        self.zonal_extractor = ZonalExtractor()

    def process_walksheds(self, city: str, station_file: str,
                          output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """Execute Stage 2: Pedestrian Walkshed Generation."""
        print(f"\n--- [Stage 2] Generating Walksheds for {city} ---")
        stations_gdf = self.walkshed_generator.load_stations(station_file, city_name=city)
        walksheds_gdf = self.walkshed_generator.generate_city_walksheds(stations_gdf, city_name=city)

        if output_path:
            self.zonal_extractor.export_statistics(walksheds_gdf, output_path)
            print(f"Saved walkshed polygons to: {output_path}")

        return walksheds_gdf

    def process_zonal_stats(self, city: str, walksheds_gdf: gpd.GeoDataFrame,
                            raster_path: str,
                            output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """Execute Stage 3: Zonal Thermal Extraction."""
        print(f"\n--- [Stage 3] Extracting LST Statistics for {city} ---")
        stats_gdf = self.zonal_extractor.process_city_zonal_stats(walksheds_gdf, raster_path)

        if output_path:
            self.zonal_extractor.export_statistics(stats_gdf, output_path)
            print(f"Saved station statistics to: {output_path}")

        return stats_gdf

    def run(self, stages: Optional[List[str]] = None,
            custom_paths: Optional[Dict[str, str]] = None):
        """
        Run the complete pipeline across all configured metropolitan areas.

        Parameters:
        -----------
        stages : Optional[List[str]]
            List of stages to execute: ['thermal', 'walksheds', 'zonal'] or None for all.
        custom_paths : Optional[Dict[str, str]]
            Optional path overrides.
        """
        if stages is None:
            stages = ['walksheds', 'zonal']

        all_walksheds = []
        all_stats = []

        for city in self.cities:
            city_lower = city.lower()
            station_file = (custom_paths or {}).get(
                f"{city}_stations", DEFAULT_PATHS["stations_template"].format(city=city)
            )
            raster_file = (custom_paths or {}).get(
                f"{city}_raster", DEFAULT_PATHS["subset_raster_template"].format(city=city)
            )
            walkshed_out = DEFAULT_PATHS["walksheds_output_template"].format(city_lower=city_lower)
            stats_out = DEFAULT_PATHS["stats_output_template"].format(city_lower=city_lower)

            walksheds_gdf = None

            # Stage 2: Walkshed Generation
            if 'walksheds' in stages:
                if os.path.exists(station_file):
                    walksheds_gdf = self.process_walksheds(city, station_file, walkshed_out)
                    all_walksheds.append(walksheds_gdf)
                else:
                    print(f"Warning: Station file not found for {city} at {station_file}. Skipping.")

            # Load existing walksheds if walkshed stage was skipped
            if walksheds_gdf is None and os.path.exists(walkshed_out):
                walksheds_gdf = gpd.read_file(walkshed_out)

            # Stage 3: Zonal Thermal Extraction
            if 'zonal' in stages and walksheds_gdf is not None:
                if os.path.exists(raster_file):
                    stats_gdf = self.process_zonal_stats(city, walksheds_gdf, raster_file, stats_out)
                    all_stats.append(stats_gdf)
                else:
                    print(f"Warning: LST raster not found for {city} at {raster_file}. Skipping zonal extraction.")

        # Consolidate cross-city master outputs if multiple cities processed
        if len(all_walksheds) > 1:
            consolidated_ws = gpd.GeoDataFrame(pd.concat(all_walksheds, ignore_index=True), crs=all_walksheds[0].crs)
            self.zonal_extractor.export_statistics(consolidated_ws, DEFAULT_PATHS["walksheds_all"])
            print(f"\nSaved consolidated walksheds to: {DEFAULT_PATHS['walksheds_all']}")

        if len(all_stats) > 1:
            consolidated_stats = gpd.GeoDataFrame(pd.concat(all_stats, ignore_index=True), crs=all_stats[0].crs)
            self.zonal_extractor.export_statistics(consolidated_stats, DEFAULT_PATHS["stats_all"])
            print(f"Saved consolidated station statistics to: {DEFAULT_PATHS['stats_all']}")
