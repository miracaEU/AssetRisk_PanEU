"""
Landslide Susceptibility Exposure Assessment Module

This module provides functions to assess infrastructure exposure to landslide
susceptibility using overlay analysis.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from tqdm import tqdm
from pathlib import Path
import traceback

import damagescanner.download as download
from damagescanner.core import DamageScanner
from damagescanner.vector import _get_cell_area_m2
from damagescanner.osm import read_osm_data

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Map ISO3 codes to ISO2 codes for European countries
iso3_to_iso2 = {
    "SVN": "SI",  # Slovenia
    "SVK": "SK",  # Slovakia
    "XKO": "XK",  # Kosovo
    "ALB": "AL",  # Albania
    "AUT": "AT",  # Austria
    "BIH": "BA",  # Bosnia and Herzegovina
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
    "MNE": "ME",  # Montenegro
    "MKD": "MK",  # North Macedonia
    "MLT": "MT",  # Malta
    "NLD": "NL",  # Netherlands
    "NOR": "NO",  # Norway
    "POL": "PL",  # Poland
    "PRT": "PT",  # Portugal
    "ROU": "RO",  # Romania
    "SRB": "RS",  # Serbia
    "SWE": "SE"   # Sweden
}

def load_infrastructure_data(country_iso3, asset_type, standardize_geom=True):
    """
    Load infrastructure data for a country and asset type.
    """
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

def get_country_bounds(country_iso3, nuts2_path=None):
    """
    Get country boundaries from NUTS2 data.
    """
    if nuts2_path and Path(nuts2_path).exists():
        nuts2 = gpd.read_file(nuts2_path).to_crs(3035)
        
        # Convert ISO3 to ISO2 using the mapping dictionary
        if country_iso3 in iso3_to_iso2:
            country_iso2 = iso3_to_iso2[country_iso3]
        else:
            print(f"Warning: No ISO2 mapping found for {country_iso3}, using first two letters")
            country_iso2 = country_iso3[:2]
            
        country_data = nuts2.loc[(nuts2.LEVL_CODE == 0) & (nuts2.CNTR_CODE == country_iso2)]
        if not country_data.empty:
            country_bounds = country_data.bounds
            return country_bounds, country_iso2
        else:
            print(f"Warning: No country bounds found for {country_iso2} in NUTS2 data")
            return None, country_iso2
    else:
        print("NUTS2 data not provided or not found, using full landslide dataset")
        return None, None

def load_landslide_susceptibility(landslide_path, country_bounds=None):
    """
    Load landslide susceptibility data from .asc file.
    
    Args:
        landslide_path (str): Path to the landslide susceptibility .asc file
        country_bounds (pandas.Series, optional): Country bounds for clipping
        
    Returns:
        xarray.Dataset: Landslide susceptibility data
    """
    print(f"Loading landslide susceptibility data from {landslide_path}...")
    
    try:
        # Load the .asc file using rasterio engine
        landslide_data = xr.open_dataset(landslide_path, engine="rasterio")
        
        # Ensure it has proper CRS info
        #if not hasattr(landslide_data, 'rio') or landslide_data.rio.crs is None:
            # Assume WGS84 if no CRS is specified
        landslide_data = landslide_data.rio.write_crs('EPSG:3035')
        landslide_data = landslide_data.rio.set_crs('EPSG:3035')

        # Clip to country bounds if provided
        if country_bounds is not None:
            print("Clipping landslide data to country bounds...")
            landslide_data = landslide_data.rio.clip_box(
                minx=country_bounds.minx.values[0],
                miny=country_bounds.miny.values[0],
                maxx=country_bounds.maxx.values[0],
                maxy=country_bounds.maxy.values[0]
            )
        
        print(f"Loaded landslide data with shape: {landslide_data.dims}")
        print(f"Susceptibility value range: {float(landslide_data.band_data.min().values)} - {float(landslide_data.band_data.max().values)}")
        
        return landslide_data
        
    except Exception as e:
        print(f"Error loading landslide susceptibility data: {e}")
        raise

def calculate_susceptibility_stats(row):
    """
    Calculate susceptibility statistics for a single infrastructure asset.
    
    Args:
        row (pandas.Series): Row from exposure results containing 'values' and 'coverage'
        
    Returns:
        dict: Dictionary with min, avg, max susceptibility and max_category_length
    """
    values = row['values']
    coverage = row['coverage']
    
    # Handle case where no values are present
    if values is None or len(values) == 0:
        return {
            'min_susceptibility': np.nan,
            'avg_susceptibility': np.nan,
            'max_susceptibility': np.nan,
            'max_category_length': 0.0
        }
    
    # Convert to numpy arrays and handle scalar case
    values = np.array(values)
    coverage = np.array(coverage)
    
    # Handle scalar coverage case
    if coverage.ndim == 0:
        coverage = np.array([coverage])
    if values.ndim == 0:
        values = np.array([values])
    
    # Ensure both arrays have the same length
    if len(values) != len(coverage):
        # If coverage is shorter, repeat the last value
        if len(coverage) == 1:
            coverage = np.repeat(coverage[0], len(values))
        else:
            # Truncate to minimum length
            min_len = min(len(values), len(coverage))
            values = values[:min_len]
            coverage = coverage[:min_len]
    
    # Remove NaN values and corresponding coverage
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return {
            'min_susceptibility': np.nan,
            'avg_susceptibility': np.nan,
            'max_susceptibility': np.nan,
            'max_category_length': 0.0
        }
    
    valid_values = values[valid_mask]
    valid_coverage = coverage[valid_mask]
    
    # Calculate basic statistics
    min_susc = float(np.min(valid_values))
    max_susc = float(np.max(valid_values))
    
    # Calculate weighted average (weighted by coverage)
    if np.sum(valid_coverage) > 0:
        avg_susc = float(np.average(valid_values, weights=valid_coverage))
    else:
        avg_susc = float(np.mean(valid_values))
    
    # Calculate length/area of maximum susceptibility category
    max_category_mask = valid_values == max_susc
    max_category_length = float(np.sum(valid_coverage[max_category_mask]))
    
    return {
        'min_susceptibility': min_susc,
        'avg_susceptibility': avg_susc,
        'max_susceptibility': max_susc,
        'max_category_length': max_category_length
    }

def process_coverage_landslide(row, cell_area_m2=None):
    """
    Process coverage for landslide assessment based on geometry type.
    """
    coverage = row['coverage']
    geom_type = row.geometry.geom_type
    
    if isinstance(coverage, (list, tuple, np.ndarray)):
        if geom_type in ['Polygon', 'MultiPolygon']:
            return sum(coverage) * cell_area_m2 if len(coverage) > 0 else 0
        else:
            return sum(coverage) if len(coverage) > 0 else 0
    elif isinstance(coverage, (int, float)):
        if geom_type in ['Polygon', 'MultiPolygon']:
            return coverage * cell_area_m2  # Fixed: removed sum() for single float
        else:
            return coverage  # Fixed: removed sum() for single float
    elif coverage is None or pd.isna(coverage):
        return 0
    else:
        try:
            return float(coverage)
        except (TypeError, ValueError):
            return 0

def assess_landslide_exposure(country_iso3, asset_type, landslide_path, nuts2_path=None):
    """
    Assess infrastructure exposure to landslide susceptibility.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        asset_type (str): Type of infrastructure to assess
        landslide_path (str): Path to landslide susceptibility .asc file
        nuts2_path (str, optional): Path to NUTS2 regions file
        
    Returns:
        geopandas.GeoDataFrame: Exposure assessment results
    """
    
    # Load infrastructure data
    features = load_infrastructure_data(country_iso3, asset_type)
    print(f"Loaded {len(features)} infrastructure features")
    
    features = features.to_crs(epsg=3035)  # Ensure features are in WGS84

    if len(features) == 0:
        print("No infrastructure features found. Returning empty results.")
        return gpd.GeoDataFrame()
    
    # Get country bounds
    country_bounds, country_iso2 = get_country_bounds(country_iso3, nuts2_path)
    
    # Load landslide susceptibility data
    landslide_data = load_landslide_susceptibility(landslide_path, country_bounds)

    
    print("Calculating exposure using DamageScanner...")
    
    # Use DamageScanner to get exposure (overlay infrastructure with susceptibility)
    try:
        exposure_results = DamageScanner(
            landslide_data,
            features,
            curves=pd.DataFrame(),  # Empty DataFrame since we're not calculating damage
            maxdam=pd.DataFrame(),  # Dummy maxdam DataFrame
        ).exposure(asset_type=asset_type)
        
        print(f"Exposure calculation completed for {len(exposure_results)} features")
        
    except Exception as e:
        print(f"Error in exposure calculation: {e}")
        traceback.print_exc()
        return gpd.GeoDataFrame()
    
    # Calculate susceptibility statistics for each asset
    print("Calculating susceptibility statistics...")
    
    # Apply susceptibility statistics calculation
    stats_list = []
    for idx, row in tqdm(exposure_results.iterrows(), total=len(exposure_results), desc="Processing assets"):
        stats = calculate_susceptibility_stats(row)
        stats_list.append(stats)
    
    # Convert to DataFrame and merge with exposure results
    stats_df = pd.DataFrame(stats_list)
    
    # Combine exposure results with statistics
    results = pd.concat([exposure_results.reset_index(drop=True), stats_df], axis=1)
    
    # Calculate total exposure length/area
    hazard_resolution = abs(landslide_data.rio.resolution()[0])
    
    # Get cell area for coverage calculation
    if hasattr(landslide_data, 'rio') and landslide_data.rio.crs:
        import pyproj
        crs = pyproj.CRS.from_user_input(landslide_data.rio.crs)
        if crs.axis_info[0].unit_name == "metre":
            cell_area_m2 = abs(landslide_data.x[1].values - landslide_data.x[0].values) * \
                            abs(landslide_data.y[0].values - landslide_data.y[1].values)
        else:
            cell_area_m2 = _get_cell_area_m2(features, hazard_resolution)
    else:
        cell_area_m2 = _get_cell_area_m2(features, hazard_resolution)
    
    # Calculate total exposure length/area

    print(results.apply(
        lambda row: process_coverage_landslide(row, cell_area_m2), axis=1
    ))

    results['total_exposure'] = results.apply(
        lambda row: process_coverage_landslide(row, cell_area_m2), axis=1
    )
    
    # Add hazard type identifier
    results['hazard_type'] = 'landslide'
    
    # Select final columns for output
    output_columns = [
        'osm_id','LAU','NUTS2', 'geometry', 'object_type', 'hazard_type',
        'min_susceptibility', 'avg_susceptibility', 'max_susceptibility',
        'max_category_length', 'total_exposure'
    ]
    
    # Filter to available columns
    available_columns = [col for col in output_columns if col in results.columns]
    final_results = results[available_columns]
    
    # Convert to GeoDataFrame if it's not already
    if not isinstance(final_results, gpd.GeoDataFrame) and 'geometry' in final_results.columns:
        final_results = gpd.GeoDataFrame(final_results, geometry='geometry')
    
    return final_results

# Example usage
if __name__ == "__main__":
    country_iso3 = sys.argv[1] if len(sys.argv) > 1 else 'PRT'  # Default to Portugal
    asset_type = sys.argv[2] if len(sys.argv) > 2 else 'roads'  # Default to roads
    
    # Landslide susceptibility data path
    landslide_path = r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\Landslides\elsus_v2.asc"
    
    # NUTS2 path for country bounds (optional)
    nuts2_path = "NUTS_RG_20M_2024_3035.geojson"
    
    # Output directory
    result_dir = Path(r"C:\MIRACA\exposure")
    result_file = result_dir / f"{country_iso3}_{asset_type}_landslide_exposure.parquet"
    
    # Create directory if it doesn't exist
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if result file exists
    if result_file.exists():
        print(f"Results already exist at {result_file}. Skipping exposure assessment.")
        sys.exit(0)
    
    print(f"Running landslide exposure assessment for {country_iso3} {asset_type}...")
    print(f"Using landslide data from: {landslide_path}")
    
    # Run landslide exposure assessment
    try:
        exposure_results = assess_landslide_exposure(
            country_iso3=country_iso3,
            asset_type=asset_type,
            landslide_path=landslide_path,
            nuts2_path=nuts2_path
        )
        
        if len(exposure_results) > 0:
            # Save results
            exposure_results.to_parquet(str(result_file))
            print(f"Results saved to: {result_file}")
            
            # Print summary statistics  
            print("\nLandslide exposure summary statistics:")
            print(f"Number of assets assessed: {len(exposure_results)}")
            print(f"Mean min susceptibility: {exposure_results['min_susceptibility'].mean():.3f}")
            print(f"Mean avg susceptibility: {exposure_results['avg_susceptibility'].mean():.3f}")
            print(f"Mean max susceptibility: {exposure_results['max_susceptibility'].mean():.3f}")
            print(f"Total exposure length/area: {exposure_results['total_exposure'].sum():.2f}")
            print(f"Total max category length/area: {exposure_results['max_category_length'].sum():.2f}")
            
            # Print susceptibility distribution
            print("\nSusceptibility category distribution (max values):")
            print(exposure_results['max_susceptibility'].value_counts().sort_index())
            
        else:
            print("No exposure results generated.")
            
    except Exception as e:
        print(f"Error running landslide exposure assessment: {e}")
        traceback.print_exc()
        sys.exit(1)