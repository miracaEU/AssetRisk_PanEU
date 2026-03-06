"""
Integrated Flood Risk Assessment Module (Refactored)

This module provides functions to assess flood risk for infrastructure
from both river (pluvial) and coastal flooding sources, with shared
core functionality extracted into reusable functions.
"""

import sys
import time
import functools
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import xarray as xr
import rasterio
from scipy import integrate
from tqdm import tqdm
from pathlib import Path
import pystac
import pystac_client
import traceback
from pystac.extensions.projection import ProjectionExtension
import concurrent.futures

import damagescanner.download as download
from damagescanner.core import DamageScanner
from damagescanner.vector import _get_cell_area_m2
from damagescanner.osm import read_osm_data,DICT_CIS_OSM
from damagescanner.config import DICT_CIS_VULNERABILITY_FLOOD

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pystac_client.warnings import NoConformsTo, FallbackToPystac

# Suppress the specific warnings
warnings.filterwarnings("ignore", category=NoConformsTo)
warnings.filterwarnings("ignore", category=FallbackToPystac)

# Map ISO3 codes to ISO2 codes for European countries
iso3_to_iso2 = {
    "SVN": "SI",  # Slovenia
    "SVK": "SK",  # Slovakia
    "AUT": "AT",  # Austria
    "BEL": "BE",  # Belgium
    "BGR": "BG",  # Bulgaria
    "CHE": "CH",  # Switzerland
    "CYP": "CY",  # Cyprus
    "CZE": "CZ",  # Czech Republic
    "DEU": "DE",  # Germany
    "DNK": "DK",  # Denmark
    "EST": "EE",  # Estonia
    "GRC": "EL",  # Greece
    "ESP": "ES",  # Spain
    "FIN": "FI",  # Finland
    "FRA": "FR",  # France
    "HRV": "HR",  # Croatia
    "HUN": "HU",  # Hungary
    "IRL": "IE",  # Ireland
    "ISL": "IS",  # Iceland
    "ITA": "IT",  # Italy
    "LIE": "LI",  # Liechtenstein
    "LTU": "LT",  # Lithuania
    "LUX": "LU",  # Luxembourg
    "LVA": "LV",  # Latvia
    "MLT": "MT",  # Malta
    "NLD": "NL",  # Netherlands
    "NOR": "NO",  # Norway
    "POL": "PL",  # Poland
    "PRT": "PT",  # Portugal
    "ROU": "RO",  # Romania
    "SWE": "SE"   # Sweden
}

# Infrastructure damage values dictionary
# Infrastructure damage values dictionary
INFRASTRUCTURE_DAMAGE_VALUES = {
    "roads": {
        "motorway": [1106, 2895, 3931],
        "motorway_link": [1106, 2895, 3931],
        "trunk": [848, 1242, 1636],
        "trunk_link": [848, 1242, 1636],
        "primary": [917, 1137, 1357],
        "primary_link": [917, 1137, 1357],
        "secondary": [257, 452, 678],
        "secondary_link": [257, 452, 678],
        "tertiary": [203, 271, 339],
        "tertiary_link": [203, 271, 339],
        "residential": [66, 136, 305],
        "road": [66, 136, 305],
        "unclassified": [66, 136, 305],
        "track": [66, 136, 305],
        "service": [66, 136, 305],
    },
    "main_roads": {
        "motorway": [1106, 2895, 3931],
        "motorway_link": [1106, 2895, 3931],
        "trunk": [848, 1242, 1636],
        "trunk_link": [848, 1242, 1636],
        "primary": [917, 1137, 1357],
        "primary_link": [917, 1137, 1357],
        "secondary": [257, 452, 678],
        "secondary_link": [257, 452, 678],
        "tertiary": [203, 271, 339],
        "tertiary_link": [203, 271, 339],
    },
    "rail": {
        "rail": [491, 2858, 14186],
    },
    "air": {
        "aerodrome": [113, 135, 165],
        "terminal": [113, 165, 4271],
        "runway": [4133, 5511, 9078],
    },
    "telecom": {
        "mast": [67506, 76630, 111998],
        "communications_tower": [139610, 152468, 229376],
        "tower": [139610, 152468, 229376],

    },
     "education": {
        "school": [267, 713, 1294],
        "kindergarten": [267, 713, 1294],
        "college": [267, 713, 1294],
        "university": [267, 713, 1294],
        "library": [267, 713, 1294],
    },
    "healthcare": {
        "hospital": [591, 1294, 2227],
        "clinic": [591, 1294, 2227],
        "doctors": [591, 1294, 2227],
        "pharmacy": [591, 1294, 2227],
        "dentist": [591, 1294, 2227],
        "physiotherapist": [591, 1294, 2227],
        "alternative": [591, 1294, 2227],
        "laboratory": [591, 1294, 2227],
        "optometrist": [591, 1294, 2227],
        "rehabilitation": [591, 1294, 2227],
        "blood_donation": [591, 1294, 2227],
        "birthing_center": [591, 1294, 2227],
    },
    "power": {
        "line": [108, 183, 1151],
        "cable": [215, 1818, 5497],
        "minor_line": [71, 102, 103],
        "plant": [649, 1558, 11110],
        "generator": [1299, 1904, 6349],
        "substation": [1299, 1904, 6349],
        "transformer": [1299, 1904, 6349],
        "pole": [73005, 97627, 369547],
        "portal": [1299, 1904, 6349],
        "tower": [6171, 103928, 275472],
        "terminal": [1299, 1904, 6349],
        "switch": [1299, 1904, 6349],
        "catenary_mast": [67506, 76630, 111998],
    },
    "gas": {
        "pipeline": [71, 102, 103],
        "storage_tank": [30310, 808265, 1515497],
        "substation": [1299886, 19047345, 63491148],
    },
    "oil": {
        "substation": [1299886, 19047345, 63491148],
        "pipeline": [71, 102, 103],
        "petroleum_well": [303100, 404133, 505166],
        "oil_refinery": [6499430, 155817332, 1111095098],
    },
    "waste_water": {
        "wastewater_plant": [7346159, 74148254, 140950350],
        "waste_transfer_station": [581081, 774774, 968468],
    },
    "buildings": {
        "yes": [591, 1294, 2227],
        "house": [591, 1294, 2227],
        "residential": [591, 1294, 2227],
        "detached": [591, 1294, 2227],
        "hut": [591, 1294, 2227],
        "industrial": [591, 1294, 2227],
        "shed": [591, 1294, 2227],
        "apartments": [591, 1294, 2227],
    }
}

DICT_CIS_VULNERABILITY_WIND = {
    "roads": {
        "motorway": ["W7.2"],
        "motorway_link": ["W7.2"],
        "trunk": ["W7.2"],
        "trunk_link": ["W7.2"],
        "primary": ["W7.2"],
        "primary_link": ["W7.2"],
        "secondary": ["W7.2"],
        "secondary_link": ["W7.2"],
        "tertiary": ["W7.2"],
        "tertiary_link": ["W7.2"],
        "residential": ["W7.2"],
        "road": ["W7.2"],
        "unclassified": ["W7.2"],
        "track": ["W7.2"],
        "service": ["W7.2"],
    },
    "main_roads": {
        "motorway": ["W7.2"],
        "motorway_link": ["W7.2"],
        "trunk": ["W7.2"],
        "trunk_link": ["W7.2"],
        "primary": ["W7.2"],
        "primary_link": ["W7.2"],
        "secondary": ["W7.2"],
        "secondary_link": ["W7.2"],
        "tertiary": ["W7.2"],
        "tertiary_link": ["W7.2"],
    },
    "rail": {
        "rail": ["W7.2"],
    },
    "air": {
        "aerodrome": ["W7.2"],
        "terminal": ["W21.13", "W21.14"],
        "runway": ["W7.2"],
    },
    "telecom": {
        "mast": ["W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14"],
        "tower": ["W10.3", "W10.4", "W10.5", "W10.6", "W10.7", "W10.8", "W10.9"],
        "communications_tower": ["W10.3", "W10.4", "W10.5", "W10.6", "W10.7", "W10.8", "W10.9"],

    },
    "education": {
        "school": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "kindergarten": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "college": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "university": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "library": ["W21.11", "W21.12", "W21.13", "W21.14"],
    },
    "healthcare": {
        "hospital": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "clinic": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "doctors": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "pharmacy": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "dentist": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "physiotherapist": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "alternative": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "laboratory": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "optometrist": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "rehabilitation": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "blood_donation": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "birthing_center": ["W21.11", "W21.12", "W21.13", "W21.14"],
    },
    "power": {
        "line": ["W6.1", "W6.2", "W6.3"],  # Transmission line
        "cable": ["W7.2"],
        "minor_line": ["W6.1", "W6.2", "W6.3"],  # Distribution line
        #"plant": ["W1.10", "W1.11", "W1.12", "W1.13", "W1.14"],
        #"generator": ["W1.10", "W1.13"],  # Using substation curves as approximation
        #"substation": ["W1.10", "W1.13"],
        #"transformer": ["W1.10", "W1.13"],  # Using substation curves as approximation
        "pole": ["W4.33", "W4.34", "W4.35", "W4.36", "W4.37" ],
        #"portal": ["W1.10", "W1.13"],  # Using substation curves as approximation
        "tower": ["W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14"],
        #"terminal": ["W1.10", "W1.13"],  # Using substation curves as approximation
        #"switch": ["W1.10", "W1.13"],  # Using substation curves as approximation
        "catenary_mast": ["W4.33", "W4.34", "W4.35", "W4.36", "W4.37" ]  # Using a subset of pole curves
    },
}

def read_as_xr_dataset(url):
    """
    Read a GeoTIFF file as an xarray Dataset.
    
    Args:
        url (str): URL or path to the GeoTIFF file
        
    Returns:
        xarray.Dataset: The raster data as an xarray Dataset with rio extensions enabled
    """
    try:
        # Open the GeoTIFF file using rasterio
        with rasterio.open(url) as src:
            # Read the data into a numpy array
            data = src.read(1)  # Assuming single-band data
        
            # Create coordinate arrays for x and y
            x_coords = np.arange(src.bounds.left, src.bounds.right, src.transform[0])
            y_coords = np.arange(src.bounds.top, src.bounds.bottom, src.transform[4])
        
            # Create an xarray DataArray
            data_array = xr.DataArray(
                data[np.newaxis, :, :],  # Add a new axis for the band dimension
                dims=("band", "y", "x"),
                coords={
                    "band": [1],
                    "x": x_coords,
                    "y": y_coords,
                    "spatial_ref": 0  # Adding spatial_ref as a coordinate
                },
                attrs={
                    "transform": src.transform,
                    "crs": src.crs.to_string() if src.crs else None,  # Convert CRS to string
                    "nodata": src.nodata
                }
            )
        
            # Create an xarray Dataset
            dataset = xr.Dataset(
                {
                    "band_data": data_array
                },
                attrs={
                    "description": "GeoTIFF data",
                    "source": url,
                    "transform": src.transform,
                    "crs": src.crs.to_string() if src.crs else None,
                }
            )
            
            # Try to enable rio extension and set spatial dimensions
            try:
                # This ensures that rioxarray is properly imported and registered
                import rioxarray  # noqa
                
                # Explicitly set spatial dimensions
                dataset = dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                
                # Set CRS explicitly
                if src.crs:
                    dataset = dataset.rio.write_crs(src.crs.to_string(), inplace=False)
            except (ImportError, AttributeError) as e:
                print(f"Warning: rioxarray extension not available: {e}")
                # Continue without rio extension - DamageScanner should still work
        
        return dataset
    except Exception as e:
        print(f"Error reading raster file: {e}")
        return None

def find_integer_in_list(strings):
    """
    Find the first integer in a list of strings.
    
    Args:
        strings (list): List of strings to search
        
    Returns:
        int: First integer found or None if not found
    """
    for s in strings:
        try:
            # Try to convert the string to an integer
            number = int(s)
            return number
        except ValueError:
            # If conversion fails, continue to the next string
            continue
    return None  # Return None if no integer is found

def interpolate_damage(rp_values, damage_values, target_rp):
    """
    Interpolate damage value for a specific return period.
    
    Args:
        rp_values: List of return periods
        damage_values: List of corresponding damage values
        target_rp: Target return period to interpolate
        
    Returns:
        Interpolated damage value
    """
    # Find indices where target would be inserted to maintain sorted order
    idx = np.searchsorted(rp_values, target_rp)
    
    # If target is exactly at a return period value
    if idx < len(rp_values) and rp_values[idx] == target_rp:
        return damage_values[idx]
    
    # If target is smaller than smallest RP, use the smallest damage
    if idx == 0:
        return damage_values[0]
    
    # If target is larger than largest RP, use the largest damage
    if idx == len(rp_values):
        return damage_values[-1]
    
    # Otherwise, interpolate between the two closest points
    rp_low, rp_high = rp_values[idx-1], rp_values[idx]
    damage_low, damage_high = damage_values[idx-1], damage_values[idx]
    
    # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return damage_low + (target_rp - rp_low) * (damage_high - damage_low) / (rp_high - rp_low)

def prepare_vulnerability_curves(asset_type, vulnerability_path):
    """
    Prepare vulnerability curves for the specified asset type.
    
    Args:
        asset_type (str): Type of infrastructure asset
        vulnerability_path (str): Path to vulnerability curve data
        
    Returns:
        tuple: (damage_curves, multi_curves, maxdam) for use in risk calculation
    """
    # Read vulnerability data
    vul_df = pd.read_excel(vulnerability_path, sheet_name='F_Vuln_Depth').fillna(method='ffill')
    
    # Get curve IDs for this asset type
    ci_system = DICT_CIS_VULNERABILITY_FLOOD[asset_type]
    
    # Select the first curve for each subtype
    selected_curves = []
    for subtype in ci_system:
        selected_curves.append(ci_system[subtype][0])
    
    # Create damage curves dataframe
    damage_curves = vul_df[['ID number'] + selected_curves]
    damage_curves = damage_curves.iloc[4:125, :]
    damage_curves.set_index('ID number', inplace=True)
    damage_curves.index = damage_curves.index.rename('Depth')  
    damage_curves = damage_curves.astype(np.float32)
    damage_curves.columns = list(ci_system.keys())
    damage_curves = damage_curves.fillna(method='ffill')
    
    # Get all unique curves for this asset type
    unique_curves = set([x for xs in DICT_CIS_VULNERABILITY_FLOOD[asset_type].values() for x in xs])
    
    # Create multi_curves dictionary
    multi_curves = {}
    for unique_curve in unique_curves:
        curve_creation = damage_curves.copy()
        get_curve_values = vul_df[unique_curve].iloc[4:125].values
        
        for curve in curve_creation:
            curve_creation.loc[:, curve] = get_curve_values
            
        multi_curves[unique_curve] = curve_creation.astype(np.float32)
    
    # Create maximum damage values dataframe
    asset_maxdam_dict = INFRASTRUCTURE_DAMAGE_VALUES[asset_type]
    maxdam_dict = {key: values[1] for key, values in asset_maxdam_dict.items()}
    maxdam = pd.DataFrame.from_dict(maxdam_dict, orient='index').reset_index()
    maxdam.columns = ['object_type', 'damage']
    
    return damage_curves, multi_curves, maxdam

def prepare_vulnerability_curves_wind(asset_type, vulnerability_path):
    """
    Prepare vulnerability curves for the specified asset type for windstorm assessment.
    
    Args:
        asset_type (str): Type of infrastructure asset
        vulnerability_path (str): Path to vulnerability curve data
        
    Returns:
        tuple: (damage_curves, multi_curves, maxdam) for use in risk calculation
    """
    # Read vulnerability data - use 'W_Vuln_Speed' sheet for wind vulnerability
    vul_df = pd.read_excel(vulnerability_path, sheet_name='W_Vuln_V10m_3sec').fillna(method='ffill')
    
    # Get curve IDs for this asset type
    ci_system = DICT_CIS_VULNERABILITY_WIND[asset_type]
    
    # Select the first curve for each subtype
    selected_curves = []
    for subtype in ci_system:
        selected_curves.append(ci_system[subtype][0])
    
    # Create damage curves dataframe
    damage_curves = vul_df[['ID number'] + selected_curves]
    damage_curves = damage_curves.iloc[4:125, :]
    damage_curves.set_index('ID number', inplace=True)
    damage_curves.index = damage_curves.index.rename('Speed')  
    damage_curves = damage_curves.astype(np.float32)
    damage_curves.columns = list(ci_system.keys())
    damage_curves = damage_curves.fillna(method='ffill')
    
    # Get all unique curves for this asset type
    unique_curves = set([x for xs in DICT_CIS_VULNERABILITY_WIND[asset_type].values() for x in xs])
    
    # Create multi_curves dictionary
    multi_curves = {}
    for unique_curve in unique_curves:
        curve_creation = damage_curves.copy()
        get_curve_values = vul_df[unique_curve].iloc[4:125].values
        
        for curve in curve_creation:
            curve_creation.loc[:, curve] = get_curve_values
            
        multi_curves[unique_curve] = curve_creation.astype(np.float32)
    
    # Create maximum damage values dataframe
    asset_maxdam_dict = INFRASTRUCTURE_DAMAGE_VALUES[asset_type]
    maxdam_dict = {key: values[1] for key, values in asset_maxdam_dict.items()}
    maxdam = pd.DataFrame.from_dict(maxdam_dict, orient='index').reset_index()
    maxdam.columns = ['object_type', 'damage']
    
    return damage_curves, multi_curves, maxdam

def convert_mixed_geometries_to_polygons(features, asset_type):
    """
    Convert point and linestring geometries to polygons for asset types with mixed geometry types.
    Only converts geometries for object types that have at least some polygon representations.
    Respects specific object types that should maintain their original geometry.
    
    Args:
        features (gpd.GeoDataFrame): Infrastructure features
        asset_type (str): Type of infrastructure asset
        
    Returns:
        gpd.GeoDataFrame: Features with consistent polygon geometries where appropriate
    """
    # Only apply for certain asset types
    if asset_type not in ['education', 'healthcare', 'telecom','power','gas', 'oil']:
        return features
    
    # Define object types that should NOT be converted for each asset type
    preserve_geometry = {
        'power': {
            'line': 'LineString',      # Should always remain as lines
            'tower': 'Point',          # Should always remain as points
            'pole': 'Point',           # Should always remain as points
            'catenary_mast': 'Point',  # Should always remain as points
            'cable': 'LineString',     # Should always remain as lines
            'minor_line': 'LineString' # Should always remain as lines
        },
        'gas': {
            'pipeline': 'LineString'   # Should always remain as lines
        },
        'oil': {
            'pipeline': 'LineString'   # Should always remain as lines
        },
        'telecom': {    
            'mast': 'Point',           # Should always remain as points
            'communications_tower': 'Point'  # Should always remain as points
        },
    }
    
    # Get the preserve list for this asset type
    preserve_list = preserve_geometry.get(asset_type, {})
    
    # Add geometry type information
    features['geom_type'] = features.geometry.geom_type
    
    # Create a mask for features that should preserve their geometry
    preserve_mask = pd.Series(False, index=features.index)
    for obj_type, geom_type in preserve_list.items():
        # Mark features with this object_type to preserve if they have the right geometry
        type_mask = (features['object_type'] == obj_type) & (features['geom_type'] == geom_type)
        preserve_mask = preserve_mask | type_mask
    
    # Get polygon features to calculate median areas (only for non-preserved features)
    polygon_features = features.loc[
        (~preserve_mask) & features.geom_type.isin(['Polygon', 'MultiPolygon'])
    ].to_crs(3035)
    
    # If no polygon features exist, return original features
    if len(polygon_features) == 0:
        features = features.drop(['geom_type'], axis=1)
        return features
        
    polygon_features['square_m2'] = polygon_features.area
    
    # Calculate median area by object type
    square_m2_object_type = polygon_features[['object_type', 'square_m2']].groupby('object_type').median()
    
    # Default area if median cannot be calculated (1000 sq meters ~ small building)
    default_area = 1000
    
    # Find object types that have mixed geometries (linestrings + polygons)
    # Only consider non-preserved features
    non_preserved_features = features[~preserve_mask]
    mixed_geom_types = non_preserved_features.groupby(['object_type', 'geom_type']).size().unstack().fillna(0)
    
    # Identify object types that have both linestrings and polygons
    linestrings_to_polygonize = []
    if 'LineString' in mixed_geom_types.columns and any(col in mixed_geom_types.columns for col in ['Polygon', 'MultiPolygon']):
        for obj_type in mixed_geom_types.index:
            # Skip if this object type should be preserved
            if obj_type in preserve_list and preserve_list[obj_type] == 'LineString':
                continue
                
            line_count = mixed_geom_types.loc[obj_type, 'LineString'] if 'LineString' in mixed_geom_types.columns else 0
            poly_count = sum(mixed_geom_types.loc[obj_type, col] for col in ['Polygon', 'MultiPolygon'] 
                            if col in mixed_geom_types.columns)
            
            # If this object type has both linestrings and polygons, add to conversion list
            if line_count > 0 and poly_count > 0:
                linestrings_to_polygonize.append(obj_type)
    
    # Convert linestrings to polygons
    if linestrings_to_polygonize:
        print(f"Converting linestrings to polygons for {asset_type}: {linestrings_to_polygonize}")
        
        # Get linestrings to convert
        all_linestrings_to_polygonize = features.loc[
            (features.object_type.isin(linestrings_to_polygonize)) & 
            (features.geom_type == 'LineString') &
            (~preserve_mask)  # Ensure we don't convert preserved features
        ]
        
        if len(all_linestrings_to_polygonize) > 0:
            # Define function to convert linestring to polygon
            def polygonize_linestring(linestring):
                try:
                    # Simple conversion for closed linestrings
                    if linestring.is_closed:
                        return shapely.geometry.Polygon(linestring)
                    else:
                        # For open linestrings, create a small buffer
                        return linestring.buffer(0.0001)
                except Exception:
                    # Fallback: create a small buffer
                    return linestring.buffer(0.0001)
            
            # Apply conversion
            new_geometries = all_linestrings_to_polygonize.geometry.apply(polygonize_linestring).values
            
            # Update geometries
            features.loc[
                (features.object_type.isin(linestrings_to_polygonize)) & 
                (features.geom_type == 'LineString') &
                (~preserve_mask),  # Ensure we don't convert preserved features
                'geometry'
            ] = new_geometries
    
    # Get the points to convert (only for object types that also have polygons)
    points_to_polygonize = []
    if 'Point' in mixed_geom_types.columns and any(col in mixed_geom_types.columns for col in ['Polygon', 'MultiPolygon']):
        for obj_type in mixed_geom_types.index:
            # Skip if this object type should be preserved
            if obj_type in preserve_list and preserve_list[obj_type] == 'Point':
                continue
                
            point_count = mixed_geom_types.loc[obj_type, 'Point'] if 'Point' in mixed_geom_types.columns else 0
            poly_count = sum(mixed_geom_types.loc[obj_type, col] for col in ['Polygon', 'MultiPolygon'] 
                            if col in mixed_geom_types.columns)
            
            # If this object type has both points and polygons, add to conversion list
            if point_count > 0 and poly_count > 0:
                points_to_polygonize.append(obj_type)
    
    if points_to_polygonize:
        all_assets_to_polygonize = features.loc[
            (features.object_type.isin(points_to_polygonize)) & 
            (features.geom_type == 'Point') &
            (~preserve_mask)  # Ensure we don't convert preserved features
        ].to_crs(3035)
        
        if len(all_assets_to_polygonize) > 0:
            print(f"Converting {len(all_assets_to_polygonize)} points to polygons for {asset_type}: {points_to_polygonize}")
            
            # Define function to polygonize points
            def polygonize_point_per_asset(asset):
                # Get buffer length (half of width/length)
                if asset.object_type in square_m2_object_type.index:
                    area = square_m2_object_type.loc[asset.object_type].values[0]
                else:
                    area = default_area
                    
                buffer_length = np.sqrt(area) / 2
                
                # Buffer the point to create a square polygon
                return asset.geometry.buffer(buffer_length, cap_style='square')
            
            # Apply the conversion
            new_geometries = all_assets_to_polygonize.apply(
                lambda asset: polygonize_point_per_asset(asset), axis=1
            ).set_crs(3035).to_crs(3035).values
            
            # Update the geometries
            features.loc[
                (features.object_type.isin(points_to_polygonize)) & 
                (features.geom_type == 'Point') &
                (~preserve_mask),  # Ensure we don't convert preserved features
                'geometry'
            ] = new_geometries
    
    # Remove the temporary geom_type column
    features = features.drop(['geom_type'], axis=1)
    
    return features


def standardize_geometry(geom, asset_type):
    """
    Standardize geometry for mixed-type assets.
    
    Args:
        geom (shapely.geometry.base.BaseGeometry): Input geometry
        asset_type (str): Type of infrastructure asset
        
    Returns:
        shapely.geometry.base.BaseGeometry: Standardized geometry
    """
    # Determine appropriate standardization based on asset type
    # if asset_type in ['roads', 'main_roads', 'rail', 'power', 'gas', 'oil']:
    #     # Linear features should be simplified to LineStrings
    #     if isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
    #         # Convert polygon to its centerline/skeleton
    #         return shapely.geometry.LineString(shapely.ops.polylabel(geom))
    #     else:
    #         return geom

    if asset_type in ['roads', 'main_roads', 'rail', 'power']:
        return geom

    if asset_type in ['air', 'education', 'healthcare', 'buildings', 'waste_water']:
        # Area features should be standardized to Polygons
        if isinstance(geom, shapely.geometry.LineString):
            # Buffer line to create a polygon
            return shapely.geometry.LineString(geom).buffer(0.0001)
        elif isinstance(geom, shapely.geometry.Point):
            # Buffer point to create a polygon
            return shapely.geometry.Point(geom).buffer(0.0001)
        else:
            return geom
    elif asset_type in ['telecom']:
        # Point features should be standardized to Points
        if isinstance(geom, (shapely.geometry.Point, shapely.geometry.MultiPoint)):
            return geom
        elif isinstance(geom, shapely.geometry.LineString):
            # Take centroid of line
            return shapely.geometry.Point(geom.centroid)
        elif isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            # Take representative point of polygon
            return shapely.geometry.Point(geom.centroid)
        
    else:
        # For unknown asset types, return as is
        return shapely.geometry.Point(geom.centroid)

def filter_inappropriate_curves(results, multi_curves, object_curve_exclusions):
    """
    Filter out inappropriate vulnerability curves for specific object types by setting them to NaN.
    
    Args:
        results (pd.DataFrame): Damage calculation results
        multi_curves (dict): Dictionary of vulnerability curves
        object_curve_exclusions (dict): Dictionary mapping object types to lists of curves to exclude
        
    Returns:
        pd.DataFrame: Filtered damage results
    """
    # Make a copy to avoid modifying the original
    filtered_results = results.copy()
    
    # Only process if we have exclusions defined
    if not object_curve_exclusions:
        return filtered_results
    
    # Get all curve columns (as strings)
    curve_cols = [str(curve) for curve in multi_curves.keys()]
    
    # Process each object type that has exclusions
    for object_type, excluded_curves in object_curve_exclusions.items():
        # Convert excluded curves to strings for consistency
        excluded_curves_str = [str(curve) for curve in excluded_curves]
        
        # Create mask for rows with this object type
        mask = filtered_results['object_type'] == object_type
        
        # If no rows of this type, continue
        if not mask.any():
            continue
        
        # Set excluded curve values to NaN for this object type
        for curve in excluded_curves_str:
            if curve in curve_cols and curve in filtered_results.columns:
                filtered_results.loc[mask, curve] = np.nan
    
    return filtered_results

def load_infrastructure_data(country_iso3, asset_type, standardize_geom=True):
    """
    Load infrastructure data for a country and asset type.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        asset_type (str): Type of infrastructure to assess
        standardize_geom (bool): Whether to standardize geometries for mixed-type assets
        
    Returns:
        gpd.GeoDataFrame: Loaded infrastructure features
    """
    # Download country infrastructure data if not already available
    # osm_data_path = Path("/scistor/ivm/data_catalogue/open_street_map/country_osm")
    # infrastructure_path = osm_data_path / f"{country_iso3}.osm.pbf"
    
    infrastructure_path = Path("C:/MIRACA/infrastructure_data") / f"{country_iso3}_{asset_type}.parquet"

    
    print(f"Reading feature data for {country_iso3} and {asset_type}...")
    
    # Load infrastructure features
    #features = read_osm_data(infrastructure_path, asset_type=asset_type)
    
    features = gpd.read_parquet(infrastructure_path).reset_index(drop=True)
    
    print(len(features), "features loaded for", asset_type)
    
    # # First convert points to realistic polygons for certain asset types
    #features = convert_mixed_geometries_to_polygons(features, asset_type)

    # # Then standardize geometries if requested (using existing function)
    # if standardize_geom:
    #     print("Standardizing geometries...")
    #     features['geometry'] = features['geometry'].apply(lambda geom: standardize_geometry(geom, asset_type))
        
    # # Drop any invalid geometries
    valid_mask = features['geometry'].is_valid
    if not valid_mask.all():
        print(f"Warning: Dropping {(~valid_mask).sum()} invalid geometries")
        features = features[valid_mask]
    
    return features

def get_country_bounds(country_iso3, nuts2_path=None, hazard_type=None):
    """
    Get country boundaries from NUTS2 data or use a simplified approach.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        nuts2_path (str, optional): Path to NUTS2 regions file
        
    Returns:
        tuple: (country_bounds, country_iso2)
    """
    if nuts2_path:
        if (hazard_type == 'river') or (hazard_type == 'windstorm'):
            nuts2 = gpd.read_file(nuts2_path).to_crs(4326)
        elif hazard_type == 'coastal':
            nuts2 = gpd.read_file(nuts2_path).to_crs(3035)
        
        # Convert ISO3 to ISO2 using the mapping dictionary
        if country_iso3 in iso3_to_iso2:
            country_iso2 = iso3_to_iso2[country_iso3]
        else:
            print(f"Warning: No ISO2 mapping found for {country_iso3}, using first two letters")
            country_iso2 = country_iso3[:2]
            
        country_bounds = nuts2.loc[(nuts2.LEVL_CODE == 0) & (nuts2.CNTR_CODE == country_iso2)].bounds
        return country_bounds, country_iso2
    else:
        # Use a simplified approach if NUTS2 data not available
        print("NUTS2 data not provided, using simplified approach for country bounds")
        return None, None

# Process coverage based on geometry type
def process_coverage(row, cell_area_m2=None):
    coverage = row['coverage']
    geom_type = row['geom_type']
    
    if isinstance(coverage, (list, tuple, np.ndarray)):
        if geom_type in ['Polygon', 'MultiPolygon']:
            # For polygon features, each list element is a fraction that needs to be converted to m²
            # Sum of fractions * cell area gives total area
            return sum(coverage) * cell_area_m2 if len(coverage) > 0 else 0
        else:
            # For LineString features, each list element is already a length in meters
            return sum(coverage) if len(coverage) > 0 else 0
    elif isinstance(coverage, (int, float)):
        # For single values (which could be from points or simplified calculations)
        if geom_type in ['Polygon', 'MultiPolygon']:
            # Convert relative overlay to square meters
            return float(coverage) * cell_area_m2
        else:
            # For point features, just return the original value
            return float(coverage)
    elif coverage is None or pd.isna(coverage):
        return 0
    else:
        # For any other type, try to get a numeric value
        try:
            return float(coverage)
        except (TypeError, ValueError):
            return 0

def calculate_damage(hazard, features, damage_curves, multi_curves, maxdam, asset_type):
    """
    Calculate damage for assets exposed to a hazard.
    
    Args:
        hazard: Hazard dataset
        features: Infrastructure features
        damage_curves: Vulnerability curves
        multi_curves: Multiple vulnerability curves
        maxdam: Maximum damage values
        asset_type: Type of infrastructure
        
    Returns:
        pd.DataFrame: Damage calculation results
    """
    # Ensure rioxarray is properly imported
    try:
        if not hasattr(hazard, 'rio'):
            hazard = hazard.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            if 'crs' in hazard.attrs:
                hazard = hazard.rio.write_crs(hazard.attrs['crs'], inplace=False)
    except (ImportError, AttributeError):
        pass
    
    # Calculate damage
    results = DamageScanner(
        hazard, 
        features, 
        curves=damage_curves, 
        maxdam=maxdam
    ).calculate(asset_type=asset_type, multi_curves=multi_curves, disable_progress=False)

    return results

def calculate_eae_ead(damages_by_rp, lengths_by_rp, protection_standard=0):
    """
    Calculate Expected Annual Exposure (EAE) and Expected Annual Damage (EAD).
    
    Args:
        damages_by_rp (dict): Dictionary mapping return periods to damage values
        lengths_by_rp (dict): Dictionary mapping return periods to length values
        protection_standard (float, optional): Protection standard return period
        
    Returns:
        tuple: (ead_value, eae_value)
    """
    # If no valid damage values, return zeros
    if not damages_by_rp:
        return 0, 0
    
    # Sort return periods and get corresponding damages and lengths
    sorted_rps = sorted(damages_by_rp.keys())
    sorted_damages = [damages_by_rp[rp] for rp in sorted_rps]
    sorted_lengths = [lengths_by_rp.get(rp, 0) for rp in sorted_rps]
    
    # If protection standard is not exactly at one of our return periods,
    # interpolate values at the protection standard
    if protection_standard > 0 and protection_standard not in sorted_rps:
        # Add interpolated damage at design standard
        interpolated_damage = interpolate_damage(sorted_rps, sorted_damages, protection_standard)
        
        # Add interpolated length at design standard
        interpolated_length = interpolate_damage(sorted_rps, sorted_lengths, protection_standard)
        
        # Insert the interpolated values in the sorted lists
        insert_idx = np.searchsorted(sorted_rps, protection_standard)
        sorted_rps.insert(insert_idx, protection_standard)
        sorted_damages.insert(insert_idx, interpolated_damage)
        sorted_lengths.insert(insert_idx, interpolated_length)
    
    # Filter to only include return periods at or above the protection standard
    filtered_rps = [rp for rp in sorted_rps if rp >= protection_standard]
    filtered_damages = [damages_by_rp[rp] if rp in damages_by_rp else 
                       interpolate_damage(sorted_rps, sorted_damages, rp) 
                       for rp in filtered_rps]
    filtered_lengths = [lengths_by_rp.get(rp, 0) if rp in lengths_by_rp else 
                       interpolate_damage(sorted_rps, sorted_lengths, rp) 
                       for rp in filtered_rps]
    
    # Convert return periods to probabilities for integration
    filtered_probs = [1/rp for rp in filtered_rps]
    
    # Calculate risk using numerical integration
    if len(filtered_damages) >= 2:
        # Reverse for integration (ascending probabilities)
        ead_value = np.trapezoid(
            y=filtered_damages[::-1], 
            x=filtered_probs[::-1]
        )
        
        # Calculate expected annual exposure (EAE)
        eae_value = np.trapezoid(
            y=filtered_lengths[::-1],
            x=filtered_probs[::-1]
        )
    elif len(filtered_damages) == 1:
        # Just one point
        ead_value = filtered_probs[0] * filtered_damages[0]
        eae_value = filtered_probs[0] * filtered_lengths[0]
    else:
        # No valid points
        ead_value = 0
        eae_value = 0
    
    return ead_value, eae_value

def format_risk_results(largest_rp, all_risks, all_exposures, multi_curves, mean_damages, lengths, hazard_type, protection_standard=None):
    """
    Format risk results into the final output DataFrame.
    
    Args:
        largest_rp (pd.DataFrame): DataFrame with largest return period data
        all_risks (pd.DataFrame): DataFrame with risk values for each curve
        all_exposures (pd.DataFrame): DataFrame with exposure values for each curve
        multi_curves (dict): Dictionary of vulnerability curves
        mean_damages (dict): Dictionary with mean damage values by return period
        lengths (dict): Dictionary with length values by return period
        hazard_type (str): Type of hazard ('river' or 'coastal')
        protection_standards (dict, optional): Dictionary mapping osm_id to protection standard
        
    Returns:
        gpd.GeoDataFrame: Final risk assessment results
    """
    # Format columns if needed
    if isinstance(largest_rp.columns, pd.MultiIndex):
        largest_rp.columns = largest_rp.columns.get_level_values(1)
    
    # Remove existing curve columns
    curve_cols = [col for col in largest_rp.columns if col in multi_curves.keys()]
    if curve_cols:
        largest_rp = largest_rp.drop(curve_cols, axis=1)
    
    # Add calculated risk values
    largest_rp.loc[:, multi_curves.keys()] = all_risks.values
    
    # Calculate total risk (mean across all non-NaN curves)
    largest_rp['EAD'] = largest_rp[list(multi_curves.keys())].mean(axis=1, skipna=True)
    
    # Add expected annual exposure (EAE)
    largest_rp['EAE'] = all_exposures.mean(axis=1, skipna=True).values

    # Add hazard type identifier
    largest_rp['hazard_type'] = hazard_type
    
    # Add protection standard for river flooding
    if hazard_type == 'river' and protection_standard is not None:
        # Add protection standard column
        largest_rp['protection_standard'] = largest_rp['osm_id'].map(protection_standard)

    # Add mean damage and length columns for each return period
    all_return_periods = sorted(list(set(list(mean_damages.keys()) + list(lengths.keys()))))
    
    rp = 1000
    

    for rp in all_return_periods:
        # Add mean damage
        if rp in mean_damages:
            largest_rp[f'mean_damage_{rp}'] = largest_rp['osm_id'].apply(lambda x: mean_damages[rp][x] if x in mean_damages[rp].keys() else np.nan)
        
        # Add length
        if rp in lengths:
            largest_rp[f'exposure_{rp}'] = largest_rp['osm_id'].apply(lambda x: lengths[rp][x] if x in lengths[rp].keys() else np.nan)

    # Create final output DataFrame with essential columns
    risk_columns = ['osm_id', 'LAU', 'NUTS2', 'EAD', 'EAE', 'hazard_type']

   # Add protection standard column if it exists
    if 'protection_standard' in largest_rp.columns:
        risk_columns.append('protection_standard')

    # Add length columns and mean damage columns to the list of columns to keep
    length_columns = [f'exposure_{rp}' for rp in all_return_periods if f'exposure_{rp}' in largest_rp.columns]
    mean_damage_columns = [f'mean_damage_{rp}' for rp in all_return_periods if f'mean_damage_{rp}' in largest_rp.columns]
    
    risk_columns.extend(length_columns)
    risk_columns.extend(mean_damage_columns)
    
    # Filter to available columns
    available_columns = [col for col in risk_columns if col in largest_rp.columns]
    risk_dataframe = largest_rp[available_columns]
    
    # Convert to GeoDataFrame if it's not already
    if not isinstance(risk_dataframe, gpd.GeoDataFrame) and 'geometry' in risk_dataframe.columns:
        risk_dataframe = gpd.GeoDataFrame(risk_dataframe, geometry='geometry')
    
    return risk_dataframe

# Create a wrapper function that receives just the return period and hazard data
def process_return_period_wrapper(common_data, args):
    """Wrapper that unpacks common data and specific return period data"""
    return_period, hazard_data = args
    features, damage_curves, multi_curves, maxdam, asset_type, object_curve_exclusions = common_data
    
    # Calculate damage
    results = calculate_damage(
        hazard_data, 
        features, 
        damage_curves, 
        multi_curves, 
        maxdam, 
        asset_type
    )
    
    # Filter out inappropriate curves for specific object types
    if object_curve_exclusions:
        results = filter_inappropriate_curves(results, multi_curves, object_curve_exclusions)
    
    # Calculate mean damage across all curves
    mean_damage_col = f'mean_damage_{return_period}'
    curve_values = [str(k) for k in multi_curves.keys()]
    results[mean_damage_col] = results[curve_values].mean(axis=1)

    hazard_resolution = abs(hazard_data.rio.resolution()[0])
    # Check if CRS is already in meters
    if hasattr(hazard_data, 'rio') and hazard_data.rio.crs:
        import pyproj
        crs = pyproj.CRS.from_user_input(hazard_data.rio.crs)
        if crs.axis_info[0].unit_name == "metre":
            # Calculate cell area directly if in meters
            cell_area_m2 = abs(hazard_data.x[1].values - hazard_data.x[0].values) * \
                            abs(hazard_data.y[0].values - hazard_data.y[1].values)
        else:
            # Get cell area using helper function
            cell_area_m2 = _get_cell_area_m2(features, hazard_resolution)
    else:
        # Fallback if rio accessor not available
        cell_area_m2 = _get_cell_area_m2(features, hazard_resolution)

    # Calculate total exposed length (coverage)
    length_col = f'length_{return_period}'
    
    # Add geometry type to results
    results['geom_type'] = results['osm_id'].map(
        dict(zip(features['osm_id'], features.geometry.geom_type))
    )    
    
    results[length_col] = results.apply(lambda row: process_coverage(row, cell_area_m2), axis=1)

    # Remove temporary geom_type column
    results = results.drop(columns=['geom_type'], errors='ignore')
    
    # Return all results for this return period
    return return_period, results

def assess_river_flood_risk(country_iso3, asset_type, nuts2_path=None, 
                           protection_standard_path=None, vulnerability_path=None,
                           standardize_geom=True, object_curve_exclusions=None):
    """
    Assess river flood risk for a specific country and asset type.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        asset_type (str): Type of infrastructure to assess
        nuts2_path (str, optional): Path to NUTS2 regions file
        protection_standard_path (str, optional): Path to flood protection standard file
        vulnerability_path (str, optional): Path to vulnerability curves file
        standardize_geom (bool): Whether to standardize geometries for mixed-type assets
        object_curve_exclusions (dict, optional): Dictionary mapping object types to lists of curves to exclude
        
    Returns:
        geopandas.GeoDataFrame: Risk assessment results with infrastructure features
    """

    # Load infrastructure data with standardized geometries
    features = load_infrastructure_data(country_iso3, asset_type, standardize_geom)
    
    # Define return periods for river flooding
    return_periods =[10, 20, 30, 40, 50, 75, 100, 200, 500]
    
    # Get country bounds
    country_bounds, country_iso2 = get_country_bounds(country_iso3, nuts2_path, hazard_type='river')
    
    # Download hazard data for each return period
    print(f"Downloading hazard data for {country_iso3}...")
    hazard_dict = {}
    
    data_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\River_floods")

    for return_period in tqdm(return_periods, desc="Loading hazard data"):
        try:
            hazard_map = xr.open_dataset(data_path / f"Europe_RP{return_period}_filled_depth.tif", engine="rasterio")
            
            if country_bounds is not None:
                hazard_dict[return_period] = hazard_map.rio.clip_box(
                    minx=country_bounds.minx.values[0],
                    miny=country_bounds.miny.values[0],
                    maxx=country_bounds.maxx.values[0],
                    maxy=country_bounds.maxy.values[0]
                )
            else:
                hazard_dict[return_period] = hazard_map
                
        except Exception as e:
            print(f"Error loading hazard data for return period {return_period}: {e}")
    
    # If no hazard data, return empty GeoDataFrame
    if not hazard_dict:
        print("No hazard data could be loaded. Aborting river flood risk assessment.")
        return gpd.GeoDataFrame()
    
    # If vulnerability path not provided, use default from Zenodo
    if vulnerability_path is None:
        vulnerability_path = "https://zenodo.org/records/10203846/files/Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx?download=1"

    # Prepare vulnerability curves
    damage_curves, multi_curves, maxdam = prepare_vulnerability_curves(asset_type, vulnerability_path)
    
    # Load protection standards if path provided
    asset_region_ds = {}
    
    if protection_standard_path:
        print("Overlaying protection standard map with country...")
        protection_standard_map = xr.open_dataset(protection_standard_path, engine="rasterio")
        protection_standard_map = protection_standard_map.coarsen(x=10, y=10, boundary="trim").mean()
        
        if hasattr(protection_standard_map, 'rio'):
            protection_standard_map.rio.write_crs("EPSG:3035", inplace=True)
        
        NUTS_EU_3035 = gpd.read_file(nuts2_path)
        country_bounds_3035 = NUTS_EU_3035.loc[(NUTS_EU_3035.LEVL_CODE == 0) & (NUTS_EU_3035.CNTR_CODE == country_iso2)].bounds

        protection_standard_map = protection_standard_map.rio.clip_box(
                            minx=country_bounds_3035.minx.values[0],
                            miny=country_bounds_3035.miny.values[0],
                            maxx=country_bounds_3035.maxx.values[0],
                            maxy=country_bounds_3035.maxy.values[0]
                        )

        features_prot = features.to_crs(3035)
        features_prot.geometry = features_prot.centroid

        protection_standard = DamageScanner(
            protection_standard_map, 
            features_prot, 
            curves=pd.DataFrame(), 
            maxdam=pd.DataFrame()
        ).exposure(asset_type=asset_type)
        

        protection_standard['design_standard'] = protection_standard['values'].apply(
            lambda x: np.max(x) if len(x) > 0 else 0
        )
        
        # Create lookup dictionary for protection standards
        asset_region_ds = dict(zip(protection_standard.osm_id.values, protection_standard.design_standard.values))
    else:
        print("Protection standard data not provided, assuming no protection")
    
    # Perform damage assessment for each return period
    print(f"Calculating damage for {len(hazard_dict)} return periods...")
    
    risk = {}  # Store damage results for each return period
    mean_damages = {}  # Store mean damage values by return period
    lengths = {}  # Store length values by return period
    
   # Package the common data that doesn't change per return period
    common_data = (features, damage_curves, multi_curves, maxdam, asset_type, object_curve_exclusions)
    
    # Create a partial function with the common data already filled in
    process_fn = functools.partial(process_return_period_wrapper, common_data)
    
    # Create work items (just return period and hazard data)
    work_items = [(rp, hazard_dict[rp]) for rp in hazard_dict.keys()]

    start_time = time.time()
    print("Starting parallel damage calculation...")

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Process in parallel with progress tracking
        results = list(tqdm(
            executor.map(process_fn, work_items),
            total=len(work_items),
            desc="Calculating damages in parallel"
        ))
    
    # Process the results
    for return_period, result_df in results:
        # Store results
        risk[return_period] = result_df
        # Create lookup dictionaries for mean damage and length
        mean_damage_col = f'mean_damage_{return_period}'
        length_col = f'length_{return_period}'
        
        mean_damages[return_period] = dict(zip(result_df['osm_id'], result_df[mean_damage_col]))
        lengths[return_period] = dict(zip(result_df['osm_id'], result_df[length_col]))
    
    parallel_time = time.time() - start_time
    print(f"Parallel execution time: {parallel_time:.2f} seconds")

    # Calculate risk with protection standards
    RP_list = sorted(list(hazard_dict.keys()))
    df_risk = pd.concat(risk, axis=1)
    largest_rp = df_risk.loc[:, pd.IndexSlice[RP_list[-1], :]]
    
    # Create lookup for osm_id
    lookup_osm_id = dict(zip(features.index, features.osm_id))
    
    # Calculate risk and exposure for each curve
    collect_risks = {}
    collect_exposures = {}
    
    for curve in tqdm(multi_curves.keys(),total=len(multi_curves), desc="Calculating risks per curve"):
        # Get the risk dataframe for this curve
        subrisk = df_risk.loc[:, pd.IndexSlice[:, curve]].fillna(0)
        
        # Calculate risk and exposure for each asset
        risk_values = []
        exposure_values = []
        
        for idx, row in subrisk.iterrows():
            # Get design standard for this asset
            design_standard = asset_region_ds.get(lookup_osm_id[idx], 0)
            
            # Get damage and length values by return period
            rp_damages = {}
            rp_lengths = {}
            
            for rp in RP_list:
                # Get damage value
                damage = row[rp]
                if not pd.isna(damage).values:
                    rp_damages[rp] = damage.values[0]
                
                # Get length value
                try:
                    length_col = f'length_{rp}'
                    rp_lengths[rp] = lengths[rp][lookup_osm_id[idx]]
                except:
                    rp_lengths[rp] = 0
                
            # Calculate EAD and EAE
            ead, eae = calculate_eae_ead(rp_damages, rp_lengths, design_standard)
            
            risk_values.append(ead)
            exposure_values.append(eae)
        
        collect_risks[curve] = risk_values
        collect_exposures[curve] = exposure_values
    
    # Create DataFrames with risk and exposure values
    all_risks = pd.DataFrame.from_dict(collect_risks)
    all_exposures = pd.DataFrame.from_dict(collect_exposures)
    
    # Format and return results
    return format_risk_results(
        largest_rp, 
        all_risks, 
        all_exposures, 
        multi_curves, 
        mean_damages,  # Now passing mean_damages correctly
        lengths, 
        'river',
        protection_standard=asset_region_ds
    )


def assess_coastal_flood_risk(country_iso3, asset_type, nuts2_path, 
                             stac_catalog_url=None, 
                             vulnerability_path=None, 
                             standardize_geom=True,
                             object_curve_exclusions=None,
                             time_horizon="2010",  # Add this parameter
                             climate_scenario="None"):  # Add this parameter
    """
    Assess coastal flood risk for a specific country and asset type.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        asset_type (str): Type of infrastructure to assess
        nuts2_path (str): Path to NUTS2 regions file
        stac_catalog_url (str, optional): URL to STAC catalog
        vulnerability_path (str, optional): Path to vulnerability curves file
        standardize_geom (bool): Whether to standardize geometries for mixed-type assets
        object_curve_exclusions (dict, optional): Dictionary mapping object types to lists of curves to exclude
        
    Returns:
        geopandas.GeoDataFrame: Risk assessment results with infrastructure features
    """
    # Load infrastructure data with standardized geometries
    features = load_infrastructure_data(country_iso3, asset_type, standardize_geom)

    # Convert to EPSG:3035 for consistency with other data
    features = features.to_crs(3035)
    
    # Create spatial index for quick queries
    feature_tree = shapely.STRtree(features.geometry)
    
    # Get country bounds
    country_bounds, country_iso2 = get_country_bounds(country_iso3, nuts2_path, hazard_type='coastal')   
    
    # If vulnerability path not provided, use default from Zenodo
    if vulnerability_path is None:
        vulnerability_path = "https://zenodo.org/records/10203846/files/Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx?download=1"
    
    # Prepare vulnerability curves
    print("Preparing vulnerability curves...")
    damage_curves, multi_curves, maxdam = prepare_vulnerability_curves(asset_type, vulnerability_path)
    
    # Connect to STAC catalog
    if stac_catalog_url is None:
        stac_catalog_url = "https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json"
    
    try:
        catalog = pystac_client.Client.open(stac_catalog_url)
        collection = catalog.get_child(id='cfhp_all')
    except Exception as e:
        print(f"Error connecting to STAC catalog: {e}")
        return gpd.GeoDataFrame()
    
    # Process coastal flood maps and calculate damage
    print(f"Processing coastal flood maps for time horizon: {time_horizon}, climate scenario: {climate_scenario}...")

    # Dictionaries to store results by return period
    risk = {}  # Store damage results
    mean_damages = {}  # Store mean damage values
    lengths = {}  # Store length values

    for flood_item in list(collection.get_items()):
        flood_name = '_'.join(flood_item.id.split('\\')).split('.')[0]
       
        # Filter flood scenarios to match specified time horizon and climate scenario
        if 'static' in flood_name:
            continue    
                
        if time_horizon not in flood_name:
            continue        
        
        if climate_scenario not in flood_name:
            continue

        if ('LOW_DEFENDED' not in flood_name):
            continue
        
        # Extract return period from flood name
        return_period = find_integer_in_list(flood_name.split('_')[:-1])
        if return_period is None:
            continue
        
        # Process each flood map
        collect_damages = []
        
        for iter_, flood_map in tqdm(enumerate(flood_item.assets),
                                   total=len(flood_item.assets),
                                   desc=flood_name, leave=True):
            if iter_ == 0:
                continue
            
            # Get the flood chunk
            flood_chunk = flood_item.assets.get(flood_map)
            
            # Access the Projection extension on the asset
            projection = ProjectionExtension.ext(flood_chunk)
            
            # Create a bbox and check if features intersect
            [chunk_bbox] = projection.geometry['coordinates']
            chunk_geom = shapely.Polygon(chunk_bbox)
            
            if len(feature_tree.query(chunk_geom)) == 0:
                continue
            
            # Open hazard data
            
            try:
                hazard = read_as_xr_dataset(flood_chunk.href)

                hazard = hazard.rio.clip_box(
                                    minx=country_bounds.minx.values[0],
                                    miny=country_bounds.miny.values[0],
                                    maxx=country_bounds.maxx.values[0],
                                    maxy=country_bounds.maxy.values[0]
                            ).load()
                
            except Exception as e:
                print(f"Error loading hazard data for {flood_name}: {e}")
                continue

            if hazard.band_data.max() == 0:
                continue

            if hazard is None:
                continue
            
            # Calculate damage
            try:
                # Calculate damage using shared function
                results = calculate_damage(
                    hazard, 
                    features, 
                    damage_curves, 
                    multi_curves, 
                    maxdam, 
                    asset_type
                )

                # np.set_printoptions(suppress=True, formatter={'float': lambda x: f"{int(x)}"})
                
                # Filter out inappropriate curves for specific object types
                if object_curve_exclusions:
                    results = filter_inappropriate_curves(results, multi_curves, object_curve_exclusions)

                # If no results, skip this chunk
                if results.empty:
                    print(f"No results for chunk {flood_name} with return period {return_period}")
                    continue

                # Calculate mean damage across all curves for this return period
                mean_damage_col = f'mean_damage_{return_period}'
                curve_values = [str(k) for k in multi_curves.keys()]
                results[mean_damage_col] = results[curve_values].mean(axis=1, skipna=True)
                    
                # Get resolution from hazard data
                hazard_resolution = abs(hazard.rio.resolution()[0])

                # Get CRS from hazard data
                hazard_crs = hazard.rio.crs             
                
                # Check if CRS is already in meters
                if hasattr(hazard, 'rio') and hazard.rio.crs:
                    import pyproj
                    crs = pyproj.CRS.from_user_input(hazard.rio.crs)
                    if crs.axis_info[0].unit_name == "metre":
                        # Calculate cell area directly if in meters
                        cell_area_m2 = abs(hazard.x[1].values - hazard.x[0].values) * \
                                    abs(hazard.y[0].values - hazard.y[1].values)
                    else:
                        # Get cell area using helper function
                        cell_area_m2 = _get_cell_area_m2(features, hazard_resolution, hazard_crs=hazard_crs)
                else:
                    # Fallback if rio accessor not available
                    cell_area_m2 = _get_cell_area_m2(features, hazard_resolution, hazard_crs=hazard_crs)
                
                # Add geometry type to results
                results['geom_type'] = results['osm_id'].map(
                    dict(zip(features['osm_id'], features.geometry.geom_type))
                )
                
                # Calculate total exposed length/area
                length_col = f'length_{return_period}'
                               
                results[length_col] = results.apply(lambda row: process_coverage(row, cell_area_m2), axis=1)
                
                # Remove temporary geom_type column
                results = results.drop(columns=['geom_type'], errors='ignore')
                
                collect_damages.append(results)
                
            except Exception as e:
                print(f"Error calculating damage for chunk: {e}")
                traceback_str = traceback.format_exc()
                print(f"Traceback: {traceback_str}")
                continue
                            
        if len(collect_damages) > 0:
            combined_damage = pd.concat(collect_damages)
            
            # Aggregate by osm_id
            agg_dict = {
                'geometry': 'first', 
                'object_type': 'first', 
                'LAU': 'first', 
                'NUTS2': 'first'
            }
            
            curve_values = [str(k) for k in multi_curves.keys()]
            for curve in curve_values:
                if curve in combined_damage.columns:
                    agg_dict[curve] = 'sum'  # Sum total damage
            
            mean_damage_col = f'mean_damage_{return_period}'
            length_col = f'length_{return_period}'
            
            if mean_damage_col in combined_damage.columns:
                agg_dict[mean_damage_col] = 'sum'  # Sum total damage
            if length_col in combined_damage.columns:
                agg_dict[length_col] = 'sum'  # Sum total exposed length
            
            combined_damage = combined_damage.groupby('osm_id', as_index=False).agg(agg_dict)
            
            print(len(combined_damage), "assets flooded for return period", return_period)

            risk[return_period] = combined_damage
            
            # Create lookup dictionaries
            if mean_damage_col in combined_damage.columns:
                mean_damages[return_period] = dict(zip(
                    combined_damage['osm_id'], 
                    combined_damage[mean_damage_col]
                ))
            
            if length_col in combined_damage.columns:
                lengths[return_period] = dict(zip(
                    combined_damage['osm_id'], 
                    combined_damage[length_col]
                ))
                
    # Check if we have any damage data
    if not risk:
        print("No damage data could be calculated for coastal flooding")
        return gpd.GeoDataFrame()
    
    # Calculate risk with numerical integration
    print("Calculating coastal flood risk...")
    
    RP_list = sorted(list(risk.keys()))
    # Calculate risk with protection standards
    df_risk = pd.concat(risk, axis=1)
    largest_rp = df_risk.loc[:, pd.IndexSlice[RP_list[-1], :]]
    
    # Create lookup for osm_id
    lookup_osm_id = dict(zip(features.index, features.osm_id))
    
    # Calculate risk and exposure for each curve
    collect_risks = {}
    collect_exposures = {}
    
    for curve in tqdm(multi_curves.keys(),total=len(multi_curves), desc="Calculating risks per curve"):
        # Get the risk dataframe for this curve
        subrisk = df_risk.loc[:, pd.IndexSlice[:, curve]].fillna(0)
        
        # Calculate risk and exposure for each asset
        risk_values = []
        exposure_values = []
        
        for idx, row in subrisk.iterrows():
            
            # Get damage and length values by return period
            rp_damages = {}
            rp_lengths = {}
            
            for rp in RP_list:
                # Get damage value
                damage = row[rp]
                if not pd.isna(damage).values:
                    rp_damages[rp] = damage.values[0]
                
                # Get length value
                try:
                    length_col = f'length_{rp}'
                    rp_lengths[rp] = lengths[rp][lookup_osm_id[idx]]
                except:
                    rp_lengths[rp] = 0
                
            # Calculate EAD and EAE
            ead, eae = calculate_eae_ead(rp_damages, rp_lengths)
            
            risk_values.append(ead)
            exposure_values.append(eae)
        
        collect_risks[curve] = risk_values
        collect_exposures[curve] = exposure_values
    
    # Create DataFrames with risk and exposure values
    all_risks = pd.DataFrame.from_dict(collect_risks)
    all_exposures = pd.DataFrame.from_dict(collect_exposures)
    
        # Format and return results
    return format_risk_results(
        largest_rp, 
        all_risks, 
        all_exposures, 
        multi_curves, 
        mean_damages, 
        lengths, 
        'coastal'
    )

def assess_windstorm_risk(country_iso3, asset_type, nuts2_path=None, 
                         vulnerability_path=None, standardize_geom=True,
                         object_curve_exclusions=None):
    """
    Assess windstorm risk for a specific country and asset type.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        asset_type (str): Type of infrastructure to assess
        nuts2_path (str, optional): Path to NUTS2 regions file
        vulnerability_path (str, optional): Path to vulnerability curves file
        standardize_geom (bool): Whether to standardize geometries for mixed-type assets
        object_curve_exclusions (dict, optional): Dictionary mapping object types to lists of curves to exclude
        
    Returns:
        geopandas.GeoDataFrame: Risk assessment results with infrastructure features
    """
    # Load infrastructure data with standardized geometries
    features = load_infrastructure_data(country_iso3, asset_type, standardize_geom)
        
    # only keep "line","tower","catenary_mast","pole","minor_line" if asset_type is "power"
    if asset_type == "power":
        features = features[features['object_type'].isin(
            ['line', 'tower', 'catenary_mast', 'pole', 'minor_line']
        )]
       
    # Define return periods for windstorms based on available files
    return_periods = [5, 10, 25, 50, 100, 250, 500]
    
    # Get country bounds
    country_bounds, country_iso2 = get_country_bounds(country_iso3, nuts2_path, hazard_type='windstorm')
    
    # Download hazard data for each return period
    print(f"Loading windstorm hazard data for {country_iso3}...")
    hazard_dict = {}
    
    data_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\Windstorms")

    for return_period in tqdm(return_periods, desc="Loading hazard data"):
        try:
            hazard_file = data_path / f"{return_period}yr_wisc_nao_0.59.tif"
            hazard_map = xr.open_dataset(hazard_file, engine="rasterio")
            
            if country_bounds is not None:
                hazard_dict[return_period] = hazard_map.rio.clip_box(
                    minx=country_bounds.minx.values[0],
                    miny=country_bounds.miny.values[0],
                    maxx=country_bounds.maxx.values[0],
                    maxy=country_bounds.maxy.values[0]
                )
            else:
                hazard_dict[return_period] = hazard_map
                
        except Exception as e:
            print(f"Error loading hazard data for return period {return_period}: {e}")
    
    # If no hazard data, return empty GeoDataFrame
    if not hazard_dict:
        print("No hazard data could be loaded. Aborting windstorm risk assessment.")
        return gpd.GeoDataFrame()
    
    # If vulnerability path not provided, use default from Zenodo
    if vulnerability_path is None:
        vulnerability_path = "https://zenodo.org/records/10203846/files/Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx?download=1"
    
    # Prepare vulnerability curves - now using WIND curves instead of FLOOD
    print("Preparing wind vulnerability curves...")
    
    # Modify prepare_vulnerability_curves to use WIND curves
    damage_curves, multi_curves, maxdam = prepare_vulnerability_curves_wind(asset_type, vulnerability_path)
    
    # Perform damage assessment for each return period
    print(f"Calculating damage for {len(hazard_dict)} return periods...")
    
    risk = {}  # Store damage results for each return period
    mean_damages = {}  # Store mean damage values by return period
    lengths = {}  # Store length values by return period
   
    # Package the common data that doesn't change per return period
    common_data = (features, damage_curves, multi_curves, maxdam, asset_type, object_curve_exclusions)
    
    # Create a partial function with the common data already filled in
    process_fn = functools.partial(process_return_period_wrapper, common_data)
    
    # Create work items (just return period and hazard data)
    work_items = [(rp, hazard_dict[rp]) for rp in hazard_dict.keys()]

    start_time = time.time()
    print("Starting parallel damage calculation...")

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Process in parallel with progress tracking
        results = list(tqdm(
            executor.map(process_fn, work_items),
            total=len(work_items),
            desc="Calculating damages in parallel"
        ))
    
    # Process the results
    for return_period, result_df in results:
        # Store results
        risk[return_period] = result_df
        
        # Create lookup dictionaries for mean damage and length
        mean_damage_col = f'mean_damage_{return_period}'
        length_col = f'length_{return_period}'
        
        mean_damages[return_period] = dict(zip(result_df['osm_id'], result_df[mean_damage_col]))
        lengths[return_period] = dict(zip(result_df['osm_id'], result_df[length_col]))
    
    parallel_time = time.time() - start_time
    print(f"Parallel execution time: {parallel_time:.2f} seconds")

    # Calculate risk with windstorm return periods
    RP_list = sorted(list(hazard_dict.keys()))
    df_risk = pd.concat(risk, axis=1)
    largest_rp = df_risk.loc[:, pd.IndexSlice[RP_list[-1], :]]
    
    # Create lookup for osm_id
    lookup_osm_id = dict(zip(features.index, features.osm_id))
    
    # Calculate risk and exposure for each curve
    collect_risks = {}
    collect_exposures = {}
    
    for curve in tqdm(multi_curves.keys(),total=len(multi_curves)):
        # Get the risk dataframe for this curve
        subrisk = df_risk.loc[:, pd.IndexSlice[:, curve]].fillna(0)
        
        # Calculate risk and exposure for each asset
        risk_values = []
        exposure_values = []
        
        for idx, row in subrisk.iterrows():
            # For windstorms, we don't have protection standards like for rivers,
            # so we use 0 (no protection)
            design_standard = 0
            
            # Get damage and length values by return period
            rp_damages = {}
            rp_lengths = {}
            
            for rp in RP_list:
                # Get damage value
                damage = row[rp]
                if not pd.isna(damage).values:
                    rp_damages[rp] = damage.values[0]
                
                # Get length value
                try:
                    length_col = f'length_{rp}'
                    rp_lengths[rp] = lengths[rp][lookup_osm_id[idx]]
                except:
                    rp_lengths[rp] = 0
                
            # Calculate EAD and EAE
            ead, eae = calculate_eae_ead(rp_damages, rp_lengths, design_standard)
            
            risk_values.append(ead)
            exposure_values.append(eae)
        
        collect_risks[curve] = risk_values
        collect_exposures[curve] = exposure_values
    
    # Create DataFrames with risk and exposure values
    all_risks = pd.DataFrame.from_dict(collect_risks)
    all_exposures = pd.DataFrame.from_dict(collect_exposures)
    
    # Format and return results
    return format_risk_results(
        largest_rp, 
        all_risks, 
        all_exposures, 
        multi_curves, 
        mean_damages,
        lengths, 
        'windstorm'  # Indicate this is windstorm hazard
    )

def miraca_risk(country_iso3, asset_type, hazard_type, 
                nuts2_path=None, protection_standard_path=None, 
                vulnerability_path=None, stac_catalog_url=None,
                object_curve_exclusions=None,
                coastal_time_horizon="2010",  # Add this parameter
                coastal_climate_scenario="None"):  # Add this parameter
    """
    Integrated MIRACA risk assessment for a specific country, asset type, and hazard.
    
    Args:
        country_iso3 (str): ISO3 code of the country (e.g., 'PRT' for Portugal)
        asset_type (str): Type of infrastructure to assess (e.g., 'rail', 'roads')
        hazard_type (str): Type of hazard to assess ('river', 'coastal', 'windstorm', or combinations)
        nuts2_path (str, optional): Path to NUTS2 regions GeoJSON file
        protection_standard_path (str, optional): Path to flood protection standard raster
        vulnerability_path (str, optional): Path to vulnerability curves Excel file
        stac_catalog_url (str, optional): URL to STAC catalog for coastal flooding
        object_curve_exclusions (dict, optional): Dictionary mapping object types to lists of curves to exclude
        coastal_time_horizon (str, optional): Time horizon for coastal flooding ('2010', '2050', '2080')
        coastal_climate_scenario (str, optional): Climate scenario for coastal flooding ('None', 'SSP245', 'SSP585')
        
    Returns:
        geopandas.GeoDataFrame: Risk assessment results with infrastructure features
    """

    print(f"Starting MIRACA risk assessment for {country_iso3}, {asset_type}, {hazard_type}")
    
    results = []
    
    # Check if asset type is supported
    if asset_type not in INFRASTRUCTURE_DAMAGE_VALUES.keys():
        raise ValueError(f"Asset type '{asset_type}' not supported. Valid types are: {list(INFRASTRUCTURE_DAMAGE_VALUES.keys())}")
    
    # Check if hazard type is valid
    valid_hazards = ['river', 'coastal', 'windstorm', 'all']
    if hazard_type not in valid_hazards and 'both' not in hazard_type:
        raise ValueError(f"Hazard type '{hazard_type}' not supported. Valid types are: {valid_hazards}")
    
    # Process river flooding if requested
    if hazard_type in ['river', 'both', 'all']:
        try:
            print("\n=== RIVER FLOOD RISK ASSESSMENT ===")
            river_risk = assess_river_flood_risk(
                country_iso3=country_iso3,
                asset_type=asset_type,
                nuts2_path=nuts2_path,
                protection_standard_path=protection_standard_path,
                vulnerability_path=vulnerability_path,
                object_curve_exclusions=object_curve_exclusions
            )
            results.append(river_risk)
            print(f"River flood risk assessment complete: {len(river_risk)} features processed")
        except Exception as e:
            print(f"Error in river flood risk assessment: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
    
    # Process coastal flooding if requested
    if hazard_type in ['coastal', 'both', 'all']:
        try:
            print(f"\n=== COASTAL FLOOD RISK ASSESSMENT ({coastal_time_horizon}, {coastal_climate_scenario}) ===")
            coastal_risk = assess_coastal_flood_risk(
                country_iso3=country_iso3,
                asset_type=asset_type,
                nuts2_path=nuts2_path, 
                stac_catalog_url=stac_catalog_url,
                vulnerability_path=vulnerability_path,
                object_curve_exclusions=object_curve_exclusions,
                time_horizon=coastal_time_horizon,
                climate_scenario=coastal_climate_scenario
            )
            results.append(coastal_risk)
            print(f"Coastal flood risk assessment complete: {len(coastal_risk)} features processed")
        except Exception as e:
            print(f"Error in coastal flood risk assessment: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
    
    # Process windstorm if requested
    if hazard_type in ['windstorm', 'all']:
        try:
            print("\n=== WINDSTORM RISK ASSESSMENT ===")
            windstorm_risk = assess_windstorm_risk(
                country_iso3=country_iso3,
                asset_type=asset_type,
                nuts2_path=nuts2_path,
                vulnerability_path=vulnerability_path,
                object_curve_exclusions=object_curve_exclusions
            )
            results.append(windstorm_risk)
            print(f"Windstorm risk assessment complete: {len(windstorm_risk)} features processed")
        except Exception as e:
            print(f"Error in windstorm risk assessment: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
    
    # Combine results if we have multiple hazard types
    if len(results) > 1:
        combined_risk = pd.concat(results)
        combined_risk = gpd.GeoDataFrame(combined_risk, geometry='geometry')
    elif len(results) == 1:
        combined_risk = results[0]
    else:
        # Return empty GeoDataFrame if no results
        combined_risk = gpd.GeoDataFrame()
    
    print(f"MIRACA risk assessment complete. Total features: {len(combined_risk)}")
    return combined_risk


# Example usage
if __name__ == "__main__":
    hazard_type = 'coastal' #sys.argv[2]  # e.g., 'river', 'coastal', or 'both'
    country_iso3 = 'ESP'
    asset_type = 'power'

    #features = load_infrastructure_data(country_iso3, asset_type, standardize_geom=True)

    #vulnerability_path = Path("/scistor/ivm/eks510/Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx")
    #vulnerability_path = "https://zenodo.org/records/10203846/files/Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx?download=1"
    vulnerability_path =  "Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx"

    for country_iso3 in ['PRT', 'ESP']: #sys.argv[1].split(','):  #
        for asset_type in ['power','roads','rail','telecom','education','healthcare']: #sys.argv[3].split(','):  # e.g., 'rail', 'roads', 'power'
            
            print(f"\nProcessing {country_iso3} - {asset_type} - {hazard_type}")             # Add command line arguments for coastal scenarios if needed
            coastal_time_horizon = sys.argv[4] if len(sys.argv) > 4 else "2010"
            coastal_climate_scenario = sys.argv[5] if len(sys.argv) > 5 else "None"   

    # for coastal_time_horizon in ['2050', '2100']: #
    #     for coastal_climate_scenario in ['SSP245','SSP585']: #

            # Define curve exclusions for different asset types
            if hazard_type == 'river' or hazard_type == 'coastal':
                asset_curve_exclusions = {
                    'power': {
                        'tower': ["F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7","F2.1", "F2.2", "F2.3","F5.1","F6.1", "F6.2"],	 
                        'plant': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],  
                        'line': ["F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7","F2.1", "F2.2", "F2.3","F5.1","F10.1"],   
                        'minor_line': ["F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7","F2.1", "F2.2", "F2.3","F5.1","F10.1"],   
                        'substation': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],  
                        'generator': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],
                        'transformer': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],
                        'portal': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],
                        'terminal': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],
                        'switch': ['F1.6', 'F6.1','F6.2','F10.1', "F2.1", "F2.2", "F2.3","F5.1","F10.1"],
                        'pole': ["F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7","F2.1", "F2.2", "F2.3","F5.1"]
                    },
                    # Add more asset types and their exclusions as needed
                }
            elif hazard_type == 'windstorm':
                asset_curve_exclusions = {
                    'power': {
                        'tower': ["W6.1", "W6.2", "W6.3","W7.2","W1.10", "W1.11", "W1.12", "W1.13", "W1.14","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],	 
                        'plant': ["W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],  
                        'line': ["W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],   
                        'minor_line': ["W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],    
                        'substation':  ["W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],   
                        'generator': ["W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"], 
                        'transformer': ["W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"], 
                        'portal': ["W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"], 
                        'terminal': ["W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"], 
                        'switch': ["W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2","W4.33", "W4.34", "W4.35", "W4.36", "W4.37"], 
                        'pole': ["W1.10", "W1.13","W1.11", "W1.12","W1.14","W3.5", "W3.6", "W3.7", "W3.8", "W3.9", "W3.10", "W3.11", "W3.12", "W3.13", "W3.14","W6.1", "W6.2", "W6.3","W7.2"], 
                    }
                    }

            # Get the exclusions for the current asset type (if any)
            current_exclusions = asset_curve_exclusions.get(asset_type, {})

            # Save results
            result_dir = Path(r"C:\MIRACA\risk")
            
            # Include scenario info in filename if coastal flooding is assessed
            scenario_suffix = ""
            if hazard_type in ['coastal', 'both', 'all'] and (coastal_time_horizon != "2010" or coastal_climate_scenario != "None"):
                scenario_suffix = f"_{coastal_time_horizon}_{coastal_climate_scenario}"
            
            result_file = result_dir / f"{country_iso3}_{asset_type}_{hazard_type}{scenario_suffix}_risk.parquet"

            # Create directory if it doesn't exist
            if not result_dir.exists():
                result_dir.mkdir(parents=True, exist_ok=True)

            # # Check if result file exists
            if result_file.exists():
                print(f"Results already exist at {result_file}. Skipping risk assessment.")
                continue
                #sys.exit(0)  # Use sys.exit() instead of exit()

            print(f"Running risk assessment for {country_iso3} {asset_type} with {hazard_type} hazard...")
            
            # Example for infrastructure risk assessment
            risk_results = miraca_risk(
                country_iso3=country_iso3,
                asset_type=asset_type,
                hazard_type=hazard_type,
                nuts2_path="NUTS_RG_20M_2024_3035.geojson",
                protection_standard_path="floodProtection_v2019_paper3.tif",
                vulnerability_path=vulnerability_path,
                object_curve_exclusions=current_exclusions,
                coastal_time_horizon=coastal_time_horizon,
                coastal_climate_scenario=coastal_climate_scenario
            )
            
            # Save results to parquet
            if len(risk_results) > 0:
                print(f"Saving risk results to {result_file}")
                risk_results.to_parquet(str(result_file))  # Convert Path to string for compatibility
            else:
                print(f"No risk results found for {country_iso3} {asset_type} with {hazard_type} hazard. Skipping save.")

            # Print summary statistics
            if len(risk_results) > 0:
                print("\nRisk summary statistics:")
                print(f"Mean risk: {risk_results['EAD'].mean()}")
                print(f"Max risk: {risk_results['EAD'].max()}")
                print(f"Total risk: {risk_results['EAD'].sum()}")
                print(f"Mean exposure: {risk_results['EAE'].mean()}")
                print(f"Total exposure: {risk_results['EAE'].sum()}")
            else:
                print("No risk results found for the given parameters.")