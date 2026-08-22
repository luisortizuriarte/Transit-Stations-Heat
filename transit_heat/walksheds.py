"""
walksheds.py
============
Pedestrian network extraction, metric UTM projection, Dijkstra shortest-path
isochrone routing, and polygonal walkshed catchment boundary generation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import osmnx as ox
import networkx as nx
from tqdm import tqdm

from .config import (
    CITY_UTM_CRS,
    OUTPUT_CRS,
    DEFAULT_WALK_SPEED_MPS,
    DEFAULT_TRIP_TIME_SECONDS,
    DEFAULT_MAX_WALK_DISTANCE_METERS,
    NETWORK_TYPE,
    GRAPH_BUFFER_METERS,
    OSMNX_CACHE_FOLDER
)


class WalkshedGenerator:
    """
    Computes realistic 10-minute pedestrian walksheds (750m at 1.25 m/s) around
    rapid transit stations using graph-theoretic routing along OpenStreetMap networks.
    """

    def __init__(self, walk_speed_mps: float = DEFAULT_WALK_SPEED_MPS,
                 trip_time_seconds: int = DEFAULT_TRIP_TIME_SECONDS,
                 cache_folder: str = OSMNX_CACHE_FOLDER):
        self.walk_speed_mps = walk_speed_mps
        self.trip_time_seconds = trip_time_seconds
        self.max_distance = walk_speed_mps * trip_time_seconds
        self.cache_folder = cache_folder

        # Configure OSMnx caching parameters
        ox.settings.use_cache = True
        ox.settings.cache_folder = cache_folder
        ox.settings.log_console = False

    @staticmethod
    def load_stations(geojson_path: str, city_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load station entrance point locations from a GeoJSON vector layer.

        Parameters:
        -----------
        geojson_path : str
            Path to station GeoJSON file.
        city_name : Optional[str]
            Optional city identifier to tag the records.

        Returns:
        --------
        gpd.GeoDataFrame
            Standardized station point GeoDataFrame in EPSG:4326.
        """
        gdf = gpd.read_file(geojson_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(OUTPUT_CRS)
        elif gdf.crs.to_string() != OUTPUT_CRS:
            gdf = gdf.to_crs(OUTPUT_CRS)

        if city_name and 'city' not in gdf.columns:
            gdf['city'] = city_name

        # Ensure station_name column exists
        if 'station_name' not in gdf.columns:
            for candidate in ['name', 'NAME', 'STATION_NAME', 'Stop_Name', 'stop_name', 'LONGNAME']:
                if candidate in gdf.columns:
                    gdf['station_name'] = gdf[candidate]
                    break
            else:
                gdf['station_name'] = [f"Station_{i+1}" for i in range(len(gdf))]

        return gdf

    @staticmethod
    def get_utm_crs(city_name: Optional[str], gdf: gpd.GeoDataFrame) -> str:
        """Get or estimate the local metric Universal Transverse Mercator (UTM) CRS."""
        if city_name and city_name in CITY_UTM_CRS:
            return CITY_UTM_CRS[city_name]
        return gdf.estimate_utm_crs().to_string()

    def fetch_pedestrian_network(self, stations_gdf: gpd.GeoDataFrame,
                                 buffer_meters: float = GRAPH_BUFFER_METERS) -> nx.MultiDiGraph:
        """
        Retrieve and project the walkable street network topology encompassing all stations.

        Parameters:
        -----------
        stations_gdf : gpd.GeoDataFrame
            Station points in local UTM projection.
        buffer_meters : float
            Regional buffer distance in meters.

        Returns:
        --------
        nx.MultiDiGraph
            Projected pedestrian network graph with edge travel impedances.
        """
        # Create buffered boundary in geographic degrees for OSMnx query
        stations_wgs84 = stations_gdf.to_crs(OUTPUT_CRS)
        convex_hull_wgs = stations_wgs84.unary_union.convex_hull
        
        # Buffer slightly in degrees (~1.2 km)
        query_polygon = convex_hull_wgs.buffer(buffer_meters / 111000.0)

        # Download walkable street network
        G = ox.graph_from_polygon(query_polygon, network_type=NETWORK_TYPE, simplify=True)

        # Project graph to local UTM CRS of the stations
        utm_crs = stations_gdf.crs
        G_proj = ox.project_graph(G, to_crs=utm_crs)

        # Parameterize travel impedance in seconds: time = distance / velocity
        for u, v, k, data in G_proj.edges(keys=True, data=True):
            length = data.get('length', 0)
            data['time_sec'] = length / self.walk_speed_mps

        return G_proj

    def compute_station_walkshed(self, G_proj: nx.MultiDiGraph,
                                 station_point: Point) -> Optional[Polygon]:
        """
        Compute a single 10-minute pedestrian walkshed boundary from a station point.

        Parameters:
        -----------
        G_proj : nx.MultiDiGraph
            Projected pedestrian street network.
        station_point : Point
            Station origin coordinates in local UTM projection.

        Returns:
        --------
        Optional[Polygon]
            2D Polygon boundary of the walkshed catchment in projected UTM coordinates.
        """
        try:
            # Snap station point to nearest network node
            orig_node = ox.nearest_nodes(G_proj, X=station_point.x, Y=station_point.y)

            # Dijkstra shortest-path traversal within the maximum travel distance
            subgraph = nx.ego_graph(G_proj, orig_node, radius=self.max_distance,
                                    distance='length')

            if len(subgraph.nodes) < 3:
                # Fallback to circular buffer if network graph is disconnected
                return station_point.buffer(self.max_distance)

            # Extract spatial coordinates of all reachable nodes
            node_points = [Point(data['x'], data['y']) for _, data in subgraph.nodes(data=True)]
            node_multipoint = unary_union(node_points)

            # Construct convex hull enclosing the reachable node cluster
            walkshed_poly = node_multipoint.convex_hull
            return walkshed_poly
        except Exception:
            return station_point.buffer(self.max_distance)

    def generate_city_walksheds(self, stations_gdf: gpd.GeoDataFrame,
                                city_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Batch generate 10-minute pedestrian walksheds for all stations in a network.

        Parameters:
        -----------
        stations_gdf : gpd.GeoDataFrame
            Station point layer.
        city_name : Optional[str]
            Metropolitan region identifier (e.g., 'NYC', 'Chicago', 'DC').

        Returns:
        --------
        gpd.GeoDataFrame
            Walkshed polygons standardized to WGS84 (EPSG:4326) with area_km2.
        """
        utm_crs = self.get_utm_crs(city_name, stations_gdf)
        stations_proj = stations_gdf.to_crs(utm_crs)

        print(f"[{city_name or 'City'}] Fetching walkable street network...")
        G_proj = self.fetch_pedestrian_network(stations_proj)

        walkshed_geoms = []
        areas_km2 = []

        print(f"[{city_name or 'City'}] Computing 10-min walksheds for {len(stations_proj)} stations...")
        for _, row in tqdm(stations_proj.iterrows(), total=len(stations_proj)):
            poly = self.compute_station_walkshed(G_proj, row.geometry)
            walkshed_geoms.append(poly)
            # Area in km² (area in m² / 1e6)
            areas_km2.append(poly.area / 1e6 if poly is not None else np.nan)

        # Build GeoDataFrame in local UTM
        walksheds_proj = stations_proj.copy()
        walksheds_proj['geometry'] = walkshed_geoms
        walksheds_proj['area_km2'] = areas_km2
        walksheds_proj['walk_time_minutes'] = int(self.trip_time_seconds / 60)
        walksheds_proj['walk_speed_mps'] = self.walk_speed_mps
        walksheds_proj['original_crs'] = str(utm_crs)
        walksheds_proj['output_crs'] = OUTPUT_CRS

        # Reproject to standard WGS84 for publishing
        walksheds_wgs84 = walksheds_proj.to_crs(OUTPUT_CRS)
        return walksheds_wgs84
