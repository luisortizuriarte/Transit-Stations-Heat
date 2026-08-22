"""
zonal.py
========
Zonal statistical extraction linking calibrated Land Surface Temperature (LST)
GeoTIFF rasters with pedestrian walkshed catchment polygons.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from tqdm import tqdm

from .config import OUTPUT_CRS


class ZonalExtractor:
    """
    Extracts microclimatic surface temperature statistical distributions
    (median, mean, min, max, std, IQR, 10th, 25th, 75th, 90th, 99th percentiles)
    across station walkshed boundaries.
    """

    def __init__(self, percentiles: Optional[List[float]] = None):
        if percentiles is None:
            self.percentiles = [10.0, 25.0, 75.0, 90.0, 99.0]
        else:
            self.percentiles = percentiles

    def extract_single_walkshed_stats(self, walkshed_geom,
                                      src_raster: rio.DatasetReader) -> Dict[str, float]:
        """
        Extract temperature statistics for a single walkshed polygon from an open rasterio dataset.

        Parameters:
        -----------
        walkshed_geom : shapely.geometry.Polygon
            Walkshed boundary polygon in raster CRS.
        src_raster : rio.DatasetReader
            Open rasterio dataset.

        Returns:
        --------
        Dict[str, float]
            Dictionary of computed temperature statistical metrics in °C.
        """
        try:
            out_image, _ = mask(src_raster, [walkshed_geom], crop=True, nodata=np.nan)
            values = out_image[0]
            
            # Mask out nodata and non-finite values
            valid_pixels = values[np.isfinite(values)]
            if src_raster.nodata is not None:
                valid_pixels = valid_pixels[valid_pixels != src_raster.nodata]
                
            # Filter out extreme physical outliers (< -10°C or > 80°C)
            valid_pixels = valid_pixels[(valid_pixels >= -10.0) & (valid_pixels <= 80.0)]

            if len(valid_pixels) == 0:
                return {
                    "median_temp": np.nan,
                    "mean_temp": np.nan,
                    "min_temp": np.nan,
                    "max_temp": np.nan,
                    "std_temp": np.nan,
                    "temp_10th": np.nan,
                    "temp_25th": np.nan,
                    "temp_75th": np.nan,
                    "temp_90th": np.nan,
                    "temp_99th": np.nan,
                    "pixel_count": 0
                }

            return {
                "median_temp": float(np.median(valid_pixels)),
                "mean_temp": float(np.mean(valid_pixels)),
                "min_temp": float(np.min(valid_pixels)),
                "max_temp": float(np.max(valid_pixels)),
                "std_temp": float(np.std(valid_pixels)),
                "temp_10th": float(np.percentile(valid_pixels, 10)),
                "temp_25th": float(np.percentile(valid_pixels, 25)),
                "temp_75th": float(np.percentile(valid_pixels, 75)),
                "temp_90th": float(np.percentile(valid_pixels, 90)),
                "temp_99th": float(np.percentile(valid_pixels, 99)),
                "pixel_count": int(len(valid_pixels))
            }
        except Exception:
            return {
                "median_temp": np.nan,
                "mean_temp": np.nan,
                "min_temp": np.nan,
                "max_temp": np.nan,
                "std_temp": np.nan,
                "temp_10th": np.nan,
                "temp_25th": np.nan,
                "temp_75th": np.nan,
                "temp_90th": np.nan,
                "temp_99th": np.nan,
                "pixel_count": 0
            }

    def process_city_zonal_stats(self, walksheds_gdf: gpd.GeoDataFrame,
                                 temperature_raster_path: str) -> gpd.GeoDataFrame:
        """
        Extract zonal temperature statistics for all station walksheds in a city.

        Parameters:
        -----------
        walksheds_gdf : gpd.GeoDataFrame
            GeoDataFrame of walkshed catchment polygons.
        temperature_raster_path : str
            Path to calibrated LST GeoTIFF.

        Returns:
        --------
        gpd.GeoDataFrame
            Standardized GeoDataFrame containing station metadata, walkshed geometries,
            and extracted thermal percentiles in EPSG:4326.
        """
        print(f"Opening LST raster: {temperature_raster_path}")
        with rio.open(temperature_raster_path) as src:
            walksheds_proj = walksheds_gdf.to_crs(src.crs)

            results = []
            print(f"Extracting zonal temperature stats for {len(walksheds_proj)} walksheds...")
            for _, row in tqdm(walksheds_proj.iterrows(), total=len(walksheds_proj)):
                stats_dict = self.extract_single_walkshed_stats(row.geometry, src)
                results.append(stats_dict)

        stats_df = gpd.GeoDataFrame(results)
        
        # Merge statistics back into the original walksheds layer
        output_gdf = walksheds_gdf.copy()
        for col in stats_df.columns:
            output_gdf[col] = stats_df[col].values

        if output_gdf.crs.to_string() != OUTPUT_CRS:
            output_gdf = output_gdf.to_crs(OUTPUT_CRS)

        return output_gdf

    @staticmethod
    def export_statistics(gdf: gpd.GeoDataFrame, output_path: str) -> str:
        """
        Export processed station temperature statistics to GeoJSON or CSV format.

        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Feature collection with thermal statistics.
        output_path : str
            Destination filepath.

        Returns:
        --------
        str
            Saved file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith('.geojson') or output_path.endswith('.json'):
            gdf.to_crs(OUTPUT_CRS).to_file(output_path, driver="GeoJSON")
        elif output_path.endswith('.csv'):
            gdf.drop(columns=['geometry'], errors='ignore').to_csv(output_path, index=False)
        return output_path
