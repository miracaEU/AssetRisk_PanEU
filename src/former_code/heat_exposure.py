#!/usr/bin/env python3
"""
Heat Exposure Analysis for Infrastructure - Batch Processing Script

Usage:
    python heat_exposure_analysis.py <country_iso3> <infrastructure_type> [data_path] [nuts2_path]

Example:
    python heat_exposure_analysis.py PRT rail
    python heat_exposure_analysis.py ESP roads /path/to/heat/data /path/to/nuts2.geojson
"""

import sys
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from damagescanner.vector import VectorExposure
import damagescanner.download as download
from damagescanner.core import DamageScanner
from damagescanner.osm import read_osm_data
from damagescanner.config import DICT_CIS_VULNERABILITY_FLOOD
import warnings
from shapely.geometry import LineString, Point

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", 
    message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.*",
    category=UserWarning
)

def parse_arguments():
    """Parse command line arguments."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    country_iso3 = sys.argv[1].upper()
    infrastructure_type = sys.argv[2].lower()
    
    # Optional arguments with defaults
    if len(sys.argv) > 3:
        data_path = Path(sys.argv[3])
    else:
        data_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\Extreme_heat")
    
    if len(sys.argv) > 4:
        nuts2_path = sys.argv[4]
    else:
        nuts2_path = "NUTS_RG_20M_2024_3035.geojson"
    
    # Validate infrastructure type
    valid_infra_types = ["roads", "rail", "air", "telecom", "education", "healthcare", "power"]
    if infrastructure_type not in valid_infra_types:
        print(f"Error: Infrastructure type '{infrastructure_type}' not supported.")
        print(f"Valid types: {', '.join(valid_infra_types)}")
        sys.exit(1)
    
    return country_iso3, infrastructure_type, data_path, nuts2_path

def get_country_name(country_iso3):
    """Get full country name from ISO3 code."""
    country_mapping = {
        'ESP': 'Spain', 'PRT': 'Portugal', 'FRA': 'France', 'ITA': 'Italy', 'DEU': 'Germany',
        'GBR': 'United Kingdom', 'NLD': 'Netherlands', 'BEL': 'Belgium', 'CHE': 'Switzerland', 
        'AUT': 'Austria', 'POL': 'Poland', 'CZE': 'Czech Republic', 'SVK': 'Slovakia', 
        'HUN': 'Hungary', 'SVN': 'Slovenia', 'HRV': 'Croatia', 'BIH': 'Bosnia and Herzegovina',
        'MNE': 'Montenegro', 'ALB': 'Albania', 'GRC': 'Greece', 'ROU': 'Romania', 'BGR': 'Bulgaria',
        'TUR': 'Turkey', 'RUS': 'Russia', 'UKR': 'Ukraine', 'BLR': 'Belarus', 'LTU': 'Lithuania',
        'LVA': 'Latvia', 'EST': 'Estonia', 'FIN': 'Finland', 'SWE': 'Sweden', 'NOR': 'Norway',
        'DNK': 'Denmark', 'ISL': 'Iceland', 'IRL': 'Ireland', 'LUX': 'Luxembourg', 'MLT': 'Malta',
        'CYP': 'Cyprus'
    }
    return country_mapping.get(country_iso3, country_iso3)

def analyze_extreme_heat_files(directory):
    """
    Analyze and organize extreme heat NetCDF files.
    
    Parameters:
    -----------
    directory : str or Path
        Path to the directory containing NetCDF files
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with organized information about each NetCDF file
    """
    # Convert to Path object if string
    directory = Path(directory)
    
    # Find all NetCDF files
    nc_files = list(directory.glob('*.nc'))
    
    # Initialize list to store file metadata
    file_metadata = []
    
    # Regular expression patterns
    projection_pattern = r'hot_days-projections-monthly-(\d+)deg-rcp_(\d+_\d+)-(\w+)-(\w+)-(\w+)-grid'
    reanalysis_pattern = r'hot_days-reanalysis-monthly-(\d+)deg-grid-(\d+)-(\d+)'
    
    for file_path in nc_files:
        filename = file_path.name
        
        # Try projection pattern first
        proj_match = re.search(projection_pattern, filename)
        
        if proj_match:
            # For projection files
            threshold = proj_match.group(1)
            rcp = proj_match.group(2)
            regional_model = proj_match.group(3)
            global_model = proj_match.group(4)
            realization = proj_match.group(5)
            type_data = "projections"
            scenario = f"rcp_{rcp}"
        else:
            # Try reanalysis pattern
            reanalysis_match = re.search(reanalysis_pattern, filename)
            
            if reanalysis_match:
                # For reanalysis files
                threshold = reanalysis_match.group(1)
                start_year = reanalysis_match.group(2)
                end_year = reanalysis_match.group(3)
                type_data = "reanalysis"
                scenario = "historical"
                regional_model = "reanalysis"
                global_model = "reanalysis"
                realization = "n/a"
            else:
                # Skip files that don't match either pattern
                print(f"Warning: Could not parse filename format for {filename}")
                continue
        
        # Open dataset to get time range
        try:
            ds = xr.open_dataset(file_path)
            time_range = f"{pd.to_datetime(ds.time.values[0]).strftime('%Y-%m')}" + \
                         f" to {pd.to_datetime(ds.time.values[-1]).strftime('%Y-%m')}"
            
            # Get spatial information
            lat_range = f"{ds.lat.min().values:.2f} to {ds.lat.max().values:.2f}"
            lon_range = f"{ds.lon.min().values:.2f} to {ds.lon.max().values:.2f}"
            
            # Get main variable name and attributes
            main_var = list(ds.data_vars)[0] if ds.data_vars else "No variables"
            
            # Close dataset
            ds.close()
            
        except Exception as e:
            time_range = f"Error: {str(e)}"
            lat_range = "Unknown"
            lon_range = "Unknown"
            main_var = "Unknown"
        
        # Store metadata
        metadata = {
            'filename': filename,
            'type': type_data,
            'temp_threshold': f"{threshold}°C",
            'scenario': scenario,
            'regional_model': regional_model,
            'global_model': global_model,
            'realization': realization,
            'time_range': time_range,
            'lat_range': lat_range,
            'lon_range': lon_range,
            'main_variable': main_var,
            'file_path': file_path
        }
        
        file_metadata.append(metadata)
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(file_metadata)
    
    # Sort by scenario, threshold, and model
    if not df.empty:
        df = df.sort_values(['scenario', 'temp_threshold', 'regional_model', 'global_model'])
        
        # Print a summary of what was found
        print(f"Found files by scenario: {df['scenario'].value_counts().to_dict()}")
        print(f"Found files by temperature threshold: {df['temp_threshold'].value_counts().to_dict()}")
    
    return df

def get_country_bounds(country_iso3, nuts2_path=None):
    """
    Get country boundaries from NUTS2 data or use a simplified approach.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        nuts2_path (str, optional): Path to NUTS2 regions file
        
    Returns:
        tuple: (country_bounds, country_iso2)
    """
    # ISO3 to ISO2 mapping
    iso3_to_iso2 = {
        'ESP': 'ES', 'PRT': 'PT', 'FRA': 'FR', 'ITA': 'IT', 'DEU': 'DE',
        'GBR': 'UK', 'NLD': 'NL', 'BEL': 'BE', 'CHE': 'CH', 'AUT': 'AT',
        'POL': 'PL', 'CZE': 'CZ', 'SVK': 'SK', 'HUN': 'HU', 'SVN': 'SI',
        'HRV': 'HR', 'BIH': 'BA', 'MNE': 'ME', 'ALB': 'AL', 'GRC': 'EL',
        'ROU': 'RO', 'BGR': 'BG', 'TUR': 'TR', 'RUS': 'RU', 'UKR': 'UA',
        'BLR': 'BY', 'LTU': 'LT', 'LVA': 'LV', 'EST': 'EE', 'FIN': 'FI',
        'SWE': 'SE', 'NOR': 'NO', 'DNK': 'DK', 'ISL': 'IS', 'IRL': 'IE',
        'LUX': 'LU', 'MLT': 'MT', 'CYP': 'CY'
    }
    
    if nuts2_path and Path(nuts2_path).exists():
        nuts2 = gpd.read_file(nuts2_path).to_crs(4326)
        
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
        print("NUTS2 data not provided or not found, using simplified approach for country bounds")
        return None, None

def group_heat_files(heat_files_df):
    """
    Group heat files by scenario and temperature threshold.
    
    Parameters:
    -----------
    heat_files_df : pd.DataFrame
        DataFrame with file metadata from analyze_extreme_heat_files
    
    Returns:
    --------
    dict
        Nested dictionary organized by scenario and temperature threshold
    """
    grouped_files = {}
    
    # Group by scenario and temperature threshold
    for scenario in heat_files_df['scenario'].unique():
        grouped_files[scenario] = {}
        
        for threshold in heat_files_df['temp_threshold'].unique():
            # Get files for this scenario and threshold
            subset = heat_files_df[(heat_files_df['scenario'] == scenario) & 
                                   (heat_files_df['temp_threshold'] == threshold)]
            
            # Filter to only include the required files
            grouped_files[scenario][threshold] = subset
    
    return grouped_files

def extract_monthly_hazard_maps(file_path, time_window, country_iso3, nuts2_path, warm_months_only=True):
    """
    Extract monthly hazard maps for a specific time window and convert to proper geospatial format.
    
    Parameters:
    -----------
    file_path : Path
        Path to the NetCDF file
    time_window : tuple
        (start_year, end_year) for the time window to extract
    country_iso3 : str
        ISO3 code of the country to clip to
    nuts2_path : str
        Path to NUTS2 regions file
    warm_months_only : bool
        If True, only return data for warm months (April-October)
    
    Returns:
    --------
    dict
        Dictionary mapping (year, month) to properly formatted hazard maps with correct spatial dimensions
    """
    # Get country bounds
    country_bounds, country_iso2 = get_country_bounds(country_iso3, nuts2_path)
    
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Convert time array to pandas datetime
    times = pd.to_datetime(ds.time.values)
    
    # Extract year and month
    years = times.year
    months = times.month
    
    # Filter to time window
    time_mask = (years >= time_window[0]) & (years <= time_window[1])
    
    # Filter to warm months if requested
    if warm_months_only:
        warm_months = [4, 5, 6, 7, 8, 9, 10]
        month_mask = np.isin(months, warm_months)
        full_mask = time_mask & month_mask
    else:
        full_mask = time_mask
    
    # Get the filtered times
    filtered_times = times[full_mask]
    
    # Extract the data variable name (should be the first data variable)
    var_name = list(ds.data_vars)[0]
    
    # Store attributes for later use
    global_attrs = ds.attrs.copy() if hasattr(ds, 'attrs') else {}
    
    # Create dictionary to store hazard maps
    hazard_maps = {}
    
    for t in filtered_times:
        year = t.year
        month = t.month
        
        # Extract data for this timestep
        monthly_data = ds[var_name].sel(time=t)
        
        # Clip to country bounds if available
        if country_bounds is not None:
            try:
                # Add CRS info before clipping
                monthly_data = monthly_data.rio.write_crs("EPSG:4326")
                
                # Clip to country bounds
                monthly_data = monthly_data.rio.clip_box(
                    minx=country_bounds.minx.values[0],
                    miny=country_bounds.miny.values[0],
                    maxx=country_bounds.maxx.values[0],
                    maxy=country_bounds.maxy.values[0]
                )
            except Exception as e:
                # If clipping fails, just continue with the original data
                print(f"Warning: Clipping failed for {year}-{month}, using original extent")
        
        # Create a new dataset with band, x, y dimensions
        lats = monthly_data.lat.values if 'lat' in monthly_data.coords else monthly_data.y.values
        lons = monthly_data.lon.values if 'lon' in monthly_data.coords else monthly_data.x.values
        data_values = monthly_data.values
        
        # Create a new dataset with band, y, x dimensions
        new_ds = xr.Dataset(
            data_vars={
                "band_data": (["band", "y", "x"], data_values[np.newaxis, :, :])
            },
            coords={
                "band": [1],
                "y": lats,
                "x": lons
            }
        )
        
        # Add CRS information
        new_ds = new_ds.rio.write_crs("EPSG:4326")
        
        # Add original attributes
        for key, value in global_attrs.items():
            new_ds.attrs[key] = value
        
        # Add temporal attributes
        new_ds.attrs['year'] = year
        new_ds.attrs['month'] = month
        
        # Add country info
        new_ds.attrs['country_iso3'] = country_iso3
        if country_iso2:
            new_ds.attrs['country_iso2'] = country_iso2
        
        # Add to dictionary
        hazard_maps[(year, month)] = new_ds
    
    # Close the dataset
    ds.close()
    
    return hazard_maps

def run_exposure_analysis(hazard_map, infrastructure_data, temp_threshold, scenario, year, month):
    """
    Run exposure analysis for a single hazard map using direct xarray data.
    
    Parameters:
    -----------
    hazard_map : xarray.Dataset
        The hazard map for a specific month
    infrastructure_data : geopandas.GeoDataFrame
        Infrastructure vector data
    temp_threshold : str
        Temperature threshold (e.g., "30°C")
    scenario : str
        Climate scenario (e.g., "historical", "rcp_4_5")
    year : int
        Year of the hazard map
    month : int
        Month of the hazard map
    
    Returns:
    --------
    pd.DataFrame
        Exposure results with added metadata
    """
    # Initialize DamageScanner with the hazard map and infrastructure data
    scanner = DamageScanner(
        hazard_data=hazard_map,  # Direct xarray data
        feature_data=infrastructure_data,
        curves=pd.DataFrame(),
        maxdam=pd.DataFrame()
    )
    
    # Run exposure analysis
    exposed_assets = scanner.exposure(
        asset_type="rail",  # This will be adjusted dynamically in main
        disable_progress=False
    )
    
    # Add metadata
    if not exposed_assets.empty:
        exposed_assets['temp_threshold'] = temp_threshold
        exposed_assets['scenario'] = scenario
        exposed_assets['year'] = year
        exposed_assets['month'] = month
    
    return exposed_assets

def calculate_exposure_statistics(exposure_results, time_window_name):
    """
    Calculate statistics from exposure results for a specific time window.
    
    Parameters:
    -----------
    exposure_results : pd.DataFrame
        Combined exposure results
    time_window_name : str
        Name of the time window
    
    Returns:
    --------
    pd.DataFrame
        Statistics by infrastructure segment
    """
    exposure_results['values'] = exposure_results['values'].apply(lambda x: x[0] if len(x) > 0 else 0)

    # Group by segment and calculate statistics
    stats = exposure_results.groupby(['osm_id', 'month']).agg({
                'values': lambda x: np.mean(x)
            }).reset_index()

    # Rename column and add time window
    stats.rename(columns={'values': f'avg_days_{time_window_name}'}, inplace=True)
    
    return stats

def calculate_relative_change_safe(baseline_stats, future_stats, baseline_name, future_name):
    """
    Calculate relative change between baseline and future periods, handling zero values.
    
    Parameters:
    -----------
    baseline_stats : pd.DataFrame
        Statistics for baseline period
    future_stats : pd.DataFrame
        Statistics for future period
    baseline_name : str
        Name of baseline time window
    future_name : str
        Name of future time window
    
    Returns:
    --------
    pd.DataFrame
        Relative change statistics
    """
    # Merge the datasets
    merged = pd.merge(
        baseline_stats, 
        future_stats, 
        on=['osm_id', 'month', 'month_name'],
        suffixes=('_baseline', '_future')
    )
    
    # Calculate relative change
    baseline_cols = [col for col in merged.columns if f'avg_days_{baseline_name}' in col]
    future_cols = [col for col in merged.columns if f'avg_days_{future_name}' in col]
    
    # Process each statistic (mean, min, max)
    stats_to_process = {'mean'}
    if len(future_cols) > 1:  # More than just mean
        stats_to_process.update(['min', 'max'])
    
    for stat in stats_to_process:
        baseline_col = f'avg_days_{baseline_name}_{stat}'
        future_col = f'avg_days_{future_name}_{stat}'
        
        if baseline_col in merged.columns and future_col in merged.columns:
            change_col = f'rel_change_{baseline_name}_to_{future_name}_{stat}'
            
            # Calculate absolute change
            merged[f'abs_change_{baseline_name}_to_{future_name}_{stat}'] = merged[future_col] - merged[baseline_col]
            
            # Calculate relative change, handling zero baseline values
            # Initialize with NaN
            merged[change_col] = np.nan
            
            # Case 1: Normal calculation where baseline > 0
            mask_normal = merged[baseline_col] > 0
            merged.loc[mask_normal, change_col] = (
                (merged.loc[mask_normal, future_col] - merged.loc[mask_normal, baseline_col]) / 
                merged.loc[mask_normal, baseline_col]
            )
            
            # Case 2: Baseline = 0, future > 0 (large increase)
            mask_zero_to_positive = (merged[baseline_col] == 0) & (merged[future_col] > 0)
            merged.loc[mask_zero_to_positive, change_col] = 10.0  # Arbitrary large value
            
            # Case 3: Baseline = 0, future = 0 (no change)
            mask_zero_to_zero = (merged[baseline_col] == 0) & (merged[future_col] == 0)
            merged.loc[mask_zero_to_zero, change_col] = 0.0
    
    return merged

def aggregate_model_results(all_results):
    """
    Aggregate results across different climate models.
    
    Parameters:
    -----------
    all_results : dict
        The nested dictionary containing all results
    
    Returns:
    --------
    dict
        Aggregated results with average, min, and max values
    """
    aggregated_results = {}
    
    # Month number to name mapping
    month_names = {
        4: 'April', 5: 'May', 6: 'June', 7: 'July',
        8: 'August', 9: 'September', 10: 'October'
    }
    
    for scenario, threshold_data in all_results.items():
        aggregated_results[scenario] = {}
        
        for threshold, model_data in threshold_data.items():
            aggregated_results[scenario][threshold] = {}
            
            # Get all time windows across all models
            all_time_windows = set()
            for model_name, time_window_data in model_data.items():
                all_time_windows.update(time_window_data.keys())
            
            # Process each time window
            for window_name in all_time_windows:
                # Collect all stats dataframes for this time window
                window_stats = []
                
                for model_name, time_window_data in model_data.items():
                    if window_name in time_window_data:
                        # Convert month numbers to month names
                        df = time_window_data[window_name].copy()
                        df['month_name'] = df['month'].map(month_names)
                        window_stats.append(df)
                
                if not window_stats:
                    continue
                
                # Merge all dataframes on osm_id and month
                merged = window_stats[0].copy()
                avg_col = f'avg_days_{window_name}'
                
                # For historical data or recent period (which is also from reanalysis),
                # don't need to aggregate models
                if scenario == 'historical' or window_name == 'recent':
                    # Just rename the average column
                    merged.rename(columns={avg_col: f'{avg_col}_mean'}, inplace=True)
                else:
                    # Rename first dataframe column
                    merged.rename(columns={avg_col: f'{avg_col}_model1'}, inplace=True)
                    
                    # Merge with other dataframes
                    for i, df in enumerate(window_stats[1:], start=2):
                        merged = pd.merge(
                            merged, 
                            df, 
                            on=['osm_id', 'month', 'month_name'],
                            suffixes=('', f'_model{i}')
                        )
                    
                    # Get all model columns
                    model_cols = [col for col in merged.columns if col.startswith(f'avg_days_{window_name}')]
                    
                    # Calculate aggregate statistics
                    merged[f'{avg_col}_mean'] = merged[model_cols].mean(axis=1)
                    merged[f'{avg_col}_min'] = merged[model_cols].min(axis=1)
                    merged[f'{avg_col}_max'] = merged[model_cols].max(axis=1)
                
                # Keep only necessary columns
                if scenario == 'historical' or window_name == 'recent':
                    keep_cols = ['osm_id', 'month', 'month_name', f'{avg_col}_mean']
                else:
                    keep_cols = ['osm_id', 'month', 'month_name', 
                                f'{avg_col}_mean', f'{avg_col}_min', f'{avg_col}_max']
                
                # Create a dataframe with the selected columns
                final_df = merged[keep_cols].copy()
                
                # Calculate yearly totals
                yearly_totals = final_df.groupby('osm_id').agg({
                    col: 'sum' for col in final_df.columns if col.startswith('avg_days')
                }).reset_index()
                
                # Add a special 'Yearly' row for each osm_id
                for osm_id in yearly_totals['osm_id'].unique():
                    yearly_row = yearly_totals[yearly_totals['osm_id'] == osm_id].iloc[0].to_dict()
                    yearly_row['month'] = 0
                    yearly_row['month_name'] = 'Yearly'
                    final_df = pd.concat([final_df, pd.DataFrame([yearly_row])], ignore_index=True)
                
                aggregated_results[scenario][threshold][window_name] = final_df
    
    return aggregated_results

def create_final_output(aggregated_results, relative_changes, features_gdf):
    """
    Create a single final output file with multi-index columns.
    
    Parameters:
    -----------
    aggregated_results : dict
        Aggregated results by scenario, threshold, and time window
    relative_changes : dict
        Relative changes between baseline and future periods
    features_gdf : geopandas.GeoDataFrame
        The original features geodataframe with geometries
    
    Returns:
    --------
    geopandas.GeoDataFrame
        Final combined output with geometries
    """
    # Get all unique infrastructure segments
    all_segments = set()
    for scenario, threshold_data in aggregated_results.items():
        for threshold, window_data in threshold_data.items():
            for window_name, stats in window_data.items():
                all_segments.update(stats['osm_id'].unique())
    
    # Define month order with Yearly first
    all_months = ['Yearly', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
    
    # Create base DataFrame with one row per segment
    base_df = pd.DataFrame({'osm_id': list(all_segments)})
    
    # Column multi-index tuples
    column_tuples = []
    
    # Process the results to create a data dictionary with multi-index columns
    data_dict = {}
    
    # Add absolute values
    for scenario, threshold_data in aggregated_results.items():
        for threshold, window_data in threshold_data.items():
            threshold_clean = threshold.replace('°', '')
            
            for window_name, stats in window_data.items():
                # Add mean, min, max columns where applicable
                if scenario == 'historical' or window_name == 'recent':
                    stats_to_add = ['mean']
                else:
                    stats_to_add = ['mean', 'min', 'max']
                
                # Process each month separately
                for month in all_months:
                    month_stats = stats[stats['month_name'] == month].copy() if not stats.empty else pd.DataFrame()
                    
                    for stat in stats_to_add:
                        # Column name in source dataframe
                        source_col = f'avg_days_{window_name}_{stat}'
                        
                        # Only proceed if source column exists and there's data for this month
                        if source_col in month_stats.columns and not month_stats.empty:
                            # Create multi-index tuple: (period, metric, value)
                            column_tuple = (month, f'days_{threshold_clean}_{scenario}_{window_name}_{stat}')
                            column_tuples.append(column_tuple)
                            
                            # Create a mapping from osm_id to value
                            value_dict = dict(zip(month_stats['osm_id'], month_stats[source_col]))
                            
                            # Add the values to the data dictionary
                            data_dict[column_tuple] = base_df['osm_id'].map(value_dict)
    
    # Add relative changes
    for scenario, threshold_data in relative_changes.items():
        for threshold, comparison_data in threshold_data.items():
            threshold_clean = threshold.replace('°', '')
            
            for comparison_name, changes in comparison_data.items():
                # Determine which stats to add based on the time windows involved
                if 'recent_to_' in comparison_name or 'baseline_' in comparison_name:
                    stats_to_add = ['mean']
                else:
                    stats_to_add = ['mean', 'min', 'max']
                
                # Process each month separately
                for month in all_months:
                    month_changes = changes[changes['month_name'] == month].copy() if not changes.empty else pd.DataFrame()
                    
                    for stat in stats_to_add:
                        # Column names in source dataframe
                        rel_col = f'rel_change_{comparison_name}_{stat}'
                        abs_col = f'abs_change_{comparison_name}_{stat}'
                        
                        # Add relative change if available
                        if rel_col in month_changes.columns and not month_changes.empty:
                            column_tuple = (month, f'rel_change_{threshold_clean}_{scenario}_{comparison_name}_{stat}')
                            column_tuples.append(column_tuple)
                            value_dict = dict(zip(month_changes['osm_id'], month_changes[rel_col]))
                            data_dict[column_tuple] = base_df['osm_id'].map(value_dict)
                            
                        # Add absolute change if available
                        if abs_col in month_changes.columns and not month_changes.empty:
                            column_tuple = (month, f'abs_change_{threshold_clean}_{scenario}_{comparison_name}_{stat}')
                            column_tuples.append(column_tuple)
                            value_dict = dict(zip(month_changes['osm_id'], month_changes[abs_col]))
                            data_dict[column_tuple] = base_df['osm_id'].map(value_dict)
    
    # Create DataFrame from the dictionary with multi-index columns
    multi_columns = pd.MultiIndex.from_tuples(column_tuples, names=['period', 'metric'])
    result_df = pd.DataFrame(data_dict, index=base_df.index, columns=multi_columns)
    
    # Add osm_id
    result_df['osm_id'] = base_df['osm_id']
    
    # Create geometry dictionary lookup from features_gdf
    geometry_dict = dict(zip(features_gdf['osm_id'], features_gdf['geometry']))
    
    # Add geometry directly using dictionary lookup
    # Add it as a multiindex column from the start
    result_df[('attributes', 'geometry')] = result_df['osm_id'].map(geometry_dict)
    
    # Set osm_id as index
    result_df = result_df.set_index('osm_id')
    
    # Extract geometry for GeoDataFrame creation
    geometry = result_df[('attributes', 'geometry')]
    result_df = result_df.drop(('attributes', 'geometry'), axis=1)
    
    # Create GeoDataFrame
    final_gdf = gpd.GeoDataFrame(result_df, geometry=geometry, crs=features_gdf.crs)
    
    # Reorder columns to put Yearly first, then months in order
    period_order = {month: i for i, month in enumerate(['Yearly', 'April', 'May', 'June', 'July', 'August', 'September', 'October'])}
    columns = list(final_gdf.columns)
    columns.sort(key=lambda x: (x[0] != 'attributes', period_order.get(x[0], 999), x[1]))
    final_gdf = final_gdf[columns]
    
    return final_gdf

def main():
    """Main processing function."""
    # Parse arguments
    
    for country_iso3 in ['PRT','ESP']:

        country_iso3_fake, infrastructure_type, data_path, nuts2_path = parse_arguments()
        country_full_name = get_country_name(country_iso3)
        
        print(f"Processing heat exposure analysis for {country_full_name} ({country_iso3}) - {infrastructure_type}")
        print(f"Data path: {data_path}")
        print(f"NUTS2 path: {nuts2_path}")
        
        # Create output directory
        output_dir = Path(f"output_{country_iso3}_{infrastructure_type}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Define the time windows for comparison
        time_windows = {
            "recent": (1990, 2016),
            "near_future": (2021, 2040),
            "mid_future": (2041, 2060),
            "far_future": (2061, 2080),
            "distant_future": (2081, 2100)
        }
        
        # Define the warm season months (April to October)
        warm_months = [4, 5, 6, 7, 8, 9, 10]
        
        try:
            # Download infrastructure data
            print(f"Downloading {infrastructure_type} data for {country_iso3}...")
            infrastructure_path = download.get_country_geofabrik(country_iso3)
            
            # Load infrastructure features
            print(f"Loading {infrastructure_type} features...")
            features = read_osm_data(infrastructure_path, asset_type=infrastructure_type)
            features_point = features.copy()
            features_point.geometry = features.geometry.centroid        
            
            if features.empty:
                print(f"No {infrastructure_type} features found for {country_iso3}. Exiting.")
                return
            
            print(f"Loaded {len(features)} {infrastructure_type} features")
            
        except Exception as e:
            print(f"Error loading infrastructure data: {e}")
            return
        
        # Analyze and group NetCDF files
        print("Analyzing NetCDF files...")
        if not data_path.exists():
            print(f"Error: Data path {data_path} does not exist")
            return
            
        heat_files_info = analyze_extreme_heat_files(data_path)
        
        if heat_files_info.empty:
            print("No NetCDF files found in the specified directory")
            return
            
        grouped_heat_files = group_heat_files(heat_files_info)
        print(f"Found {len(heat_files_info)} files across {len(grouped_heat_files)} scenarios")
        
        # Store results for each scenario and threshold
        all_results = {}
        
        # Process each scenario and threshold
        for scenario, threshold_data in grouped_heat_files.items():
            all_results[scenario] = {}
            
            for threshold, files_df in threshold_data.items():
                all_results[scenario][threshold] = {}
                
                # Process each model
                for _, file_row in files_df.iterrows():
                    file_path = file_row['file_path']
                    model_name = f"{file_row['regional_model']}_{file_row['global_model']}"
                    
                    print(f"\nProcessing: {scenario} | {threshold} | {model_name}")
                    
                    # Store results for each time window
                    time_window_results = {}
                    
                    # Process each time window
                    for window_name, (start_year, end_year) in tqdm(time_windows.items(), desc="Time windows", leave=True):
                        # Skip future windows for historical data and vice versa
                        if scenario == "historical" and "future" in window_name:
                            continue
                        if scenario != "historical" and window_name not in ["near_future", "mid_future", "far_future", "distant_future"]:
                            continue
                        
                        try:
                            # Extract monthly hazard maps
                            monthly_hazards = extract_monthly_hazard_maps(
                                            file_path, 
                                            (start_year, end_year), 
                                            country_iso3, 
                                            nuts2_path, 
                                            warm_months_only=True
                                        )
                            
                            if not monthly_hazards:
                                print(f"  No hazard data found for {window_name}")
                                continue
                            
                            # Run exposure analysis for each month
                            exposure_results = []
                            
                            for i, ((year, month), hazard_map) in enumerate(monthly_hazards.items()):
                                # Create point geometry for infrastructure (centroids)

                                
                                # Run exposure analysis
                                exposed = run_exposure_analysis(
                                    hazard_map, features_point, threshold, scenario, year, month
                                )
                                
                                if not exposed.empty:
                                    exposure_results.append(exposed)
                            
                            # Combine results
                            if exposure_results:
                                combined_results = pd.concat(exposure_results, ignore_index=True)
                                
                                # Calculate statistics
                                stats = calculate_exposure_statistics(combined_results, window_name)
                                
                                # Store results
                                time_window_results[window_name] = stats
                            else:
                                print(f"  No exposure results found for {window_name}")
                                
                        except Exception as e:
                            print(f"  Error processing {window_name}: {e}")
                            continue
                    
                    # Store model results
                    all_results[scenario][threshold][model_name] = time_window_results
                    print(f"Completed processing for {model_name}")
        
        # Save intermediate results
        print("Saving intermediate results...")
        
        # Create a metadata file
        metadata = {
            'country_iso3': country_iso3,
            'country_name': country_full_name,
            'infrastructure_type': infrastructure_type,
            'scenarios': list(all_results.keys()),
            'thresholds': [threshold for scenario in all_results for threshold in all_results[scenario]],
            'models': [model for scenario in all_results for threshold in all_results[scenario] 
                    for model in all_results[scenario][threshold]],
            'time_windows': list(time_windows.keys())
        }
            
        print("Aggregating results across climate models...")
        aggregated_results = aggregate_model_results(all_results)
        
        print("Calculating relative changes...")
        relative_changes = {}
        
        # Check if historical data exists
        if 'historical' in aggregated_results:
            # Process each future scenario
            for scenario, threshold_data in aggregated_results.items():
                if scenario == "historical":
                    continue
                    
                relative_changes[scenario] = {}
                
                for threshold, window_data in threshold_data.items():
                    relative_changes[scenario][threshold] = {}
                    
                    # Get historical data for this threshold
                    if threshold in aggregated_results['historical']:
                        historical_data = aggregated_results['historical'][threshold]
                        
                        # Calculate relative changes for each future window
                        for future_window in ["near_future", "mid_future", "far_future", "distant_future"]:
                            if future_window in window_data:
                                # Compare with recent period
                                if "recent" in historical_data:
                                    print(f"  Comparing recent to {future_window}")
                                    rel_change = calculate_relative_change_safe(
                                        historical_data["recent"],
                                        window_data[future_window],
                                        "recent",
                                        future_window
                                    )
                                    relative_changes[scenario][threshold][f"recent_to_{future_window}"] = rel_change
                    else:
                        print(f"  Warning: No historical data for threshold {threshold}")
        else:
            print("Warning: No historical data found. Skipping relative change calculations.")
        
        # Create final output
        print("Creating final output...")
        final_output = create_final_output(
            aggregated_results, 
            relative_changes, 
            features
        )
        
        # Extract yearly data
        yearly_output = final_output[['Yearly']].copy()
        yearly_output.columns = yearly_output.columns.droplevel(0)
        
        # Save final results
        output_filename = f"{country_iso3}_{infrastructure_type}_heat_exposure.parquet"
        yearly_output.reset_index().to_parquet(output_dir / output_filename)
        
        print(f"Analysis completed successfully!")
        print(f"Results saved to: {output_dir / output_filename}")
        print(f"Total infrastructure segments analyzed: {len(yearly_output)}")

if __name__ == "__main__":
    main()