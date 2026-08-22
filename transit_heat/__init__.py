"""
transit_heat
============
Transit Station Heat Exposure Analysis package.

Modules:
- config: Central configuration, spatial coordinate systems, calibration constants.
- thermal: Landsat Collection 2 Level-2 Surface Temperature calibration, QA masking, and mosaicing.
- walksheds: OSMnx pedestrian network routing and 10-minute isochrone walkshed construction.
- zonal: Zonal temperature extraction linking satellite rasters with pedestrian catchments.
- pipeline: End-to-end pipeline orchestrator.
"""

from .thermal import ThermalProcessor
from .walksheds import WalkshedGenerator
from .zonal import ZonalExtractor
from .pipeline import TransitHeatPipeline

__version__ = "1.0.0"
__all__ = [
    "ThermalProcessor",
    "WalkshedGenerator",
    "ZonalExtractor",
    "TransitHeatPipeline",
    "__version__"
]
