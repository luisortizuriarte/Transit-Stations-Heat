"""
thermal.py
==========
Thermal radiometry, Landsat Collection 2 Level-2 QA bitmasking, temporal mean
compositing, multi-tile spatial mosaicing, and corridor subset cropping.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
import geopandas as gpd

from .config import (
    ST_SCALE_FACTOR,
    ST_ADD_OFFSET,
    KELVIN_TO_CELSIUS,
    QA_EXCLUDE_BITS
)


class ThermalProcessor:
    """
    Processes Landsat 8 and 9 Collection 2 Level-2 Surface Temperature (ST Band 10)
    and Pixel Quality Assessment (QA_PIXEL) rasters into calibrated, cloud-free,
    temporally composited, and spatially stitched city temperature GeoTIFFs.
    """

    def __init__(self, scale_factor: float = ST_SCALE_FACTOR,
                 offset: float = ST_ADD_OFFSET,
                 kelvin_offset: float = KELVIN_TO_CELSIUS):
        self.scale_factor = scale_factor
        self.offset = offset
        self.kelvin_offset = kelvin_offset

    @staticmethod
    def create_clear_mask(qa_array: np.ndarray) -> np.ndarray:
        """
        Generate a boolean mask for valid, clear-sky, non-water land pixels
        from a Landsat Collection 2 QA_PIXEL array.

        Parameters:
        -----------
        qa_array : np.ndarray
            16-bit unsigned integer array of QA_PIXEL values.

        Returns:
        --------
        np.ndarray
            Boolean array where True indicates clear, valid land surface pixels.
        """
        qa = qa_array.astype(np.uint16)

        # Exclude Dilated Cloud (Bit 1), Cirrus (Bit 2), Cloud (Bit 3),
        # Cloud Shadow (Bit 4), Snow (Bit 5), and Water (Bit 7)
        dilated_cloud = (qa & (1 << QA_EXCLUDE_BITS["dilated_cloud"])) != 0
        cirrus        = (qa & (1 << QA_EXCLUDE_BITS["cirrus"])) != 0
        cloud         = (qa & (1 << QA_EXCLUDE_BITS["cloud"])) != 0
        cloud_shadow  = (qa & (1 << QA_EXCLUDE_BITS["cloud_shadow"])) != 0
        snow          = (qa & (1 << QA_EXCLUDE_BITS["snow"])) != 0
        water         = (qa & (1 << QA_EXCLUDE_BITS["water"])) != 0

        # Exclude high-confidence cloud, shadow, and cirrus flags (Bits 8-15)
        cloud_conf_high  = ((qa >> 8) & 3) == 3
        shadow_conf_high = ((qa >> 10) & 3) == 3
        cirrus_conf_high = ((qa >> 14) & 3) == 3

        invalid_mask = (
            dilated_cloud | cirrus | cloud | cloud_shadow |
            snow | water | cloud_conf_high | shadow_conf_high | cirrus_conf_high
        )
        return ~invalid_mask

    def calibrate_temperature(self, dn_array: np.ndarray,
                              mask_array: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calibrate raw Landsat Digital Numbers (DN) to degrees Celsius (°C).

        Formula: T(°C) = (DN * 0.00341802 + 149.0) - 273.15
        """
        calibrated = (dn_array.astype(np.float32) * self.scale_factor + self.offset) - self.kelvin_offset
        if mask_array is not None:
            calibrated[~mask_array] = np.nan
        return calibrated

    def process_tile_temporal_mean(self, st_files: List[str],
                                   qa_files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Process multiple scene acquisitions for a single geographic tile footprint,
        apply pixel-level QA masking, and compute the pixel-wise temporal mean.

        Parameters:
        -----------
        st_files : List[str]
            Paths to ST_B10.TIF surface temperature rasters.
        qa_files : List[str]
            Corresponding paths to QA_PIXEL.TIF rasters.

        Returns:
        --------
        Tuple[np.ndarray, dict]
            Temporally averaged 2D temperature array in °C, and rasterio profile.
        """
        if not st_files or len(st_files) != len(qa_files):
            raise ValueError("Mismatched or empty ST and QA file lists.")

        masked_layers = []
        profile = None

        for st_path, qa_path in zip(st_files, qa_files):
            with rio.open(st_path) as st_src, rio.open(qa_path) as qa_src:
                if profile is None:
                    profile = st_src.profile.copy()

                st_data = st_src.read(1)
                qa_data = qa_src.read(1)

                clear_mask = self.create_clear_mask(qa_data)
                temp_celsius = self.calibrate_temperature(st_data, clear_mask)
                masked_layers.append(temp_celsius)

        # Compute temporal pixel-wise mean ignoring NaNs
        stack = np.dstack(masked_layers)
        with np.errstate(all='ignore'):
            temporal_mean = np.nanmean(stack, axis=2)

        profile.update(dtype=rio.float32, count=1, nodata=np.nan)
        return temporal_mean.astype(np.float32), profile

    def stitch_city_tiles(self, tile_raster_paths: List[str],
                          output_path: str) -> str:
        """
        Merge adjacent temporal mean tile GeoTIFFs into a unified regional mosaic.

        Parameters:
        -----------
        tile_raster_paths : List[str]
            List of paths to GeoTIFF tiles.
        output_path : str
            Destination path for the stitched GeoTIFF mosaic.

        Returns:
        --------
        str
            Output GeoTIFF path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        src_files = [rio.open(p) for p in tile_raster_paths]
        try:
            mosaic, out_trans = merge(src_files, nodata=np.nan)
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "lzw"
            })

            with rio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic.astype(np.float32))
        finally:
            for src in src_files:
                src.close()

        return output_path

    def crop_to_station_bounds(self, input_raster_path: str,
                              stations_gdf: gpd.GeoDataFrame,
                              output_path: str,
                              buffer_degrees: float = 0.05) -> str:
        """
        Crop a regional surface temperature GeoTIFF mosaic to the bounding box
        encompassing all station points/walksheds, expanded with a spatial buffer.

        Parameters:
        -----------
        input_raster_path : str
            Path to full regional stitched GeoTIFF.
        stations_gdf : gpd.GeoDataFrame
            GeoDataFrame containing station point locations or walksheds.
        output_path : str
            Destination path for cropped GeoTIFF subset.
        buffer_degrees : float
            Bounding margin expansion in geographic degrees.

        Returns:
        --------
        str
            Output cropped GeoTIFF path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with rio.open(input_raster_path) as src:
            stations_proj = stations_gdf.to_crs(src.crs)
            minx, miny, maxx, maxy = stations_proj.total_bounds

            # Apply buffer margin in projected CRS units
            bbox_geom = box(minx - buffer_degrees, miny - buffer_degrees,
                            maxx + buffer_degrees, maxy + buffer_degrees)

            out_image, out_transform = mask(src, [bbox_geom], crop=True, nodata=np.nan)
            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "lzw"
            })

            with rio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image.astype(np.float32))

        return output_path
