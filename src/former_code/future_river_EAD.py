import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import itertools
import os
import time
from datetime import datetime

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

def calculate_eae_ead(damages_by_rp, lengths_by_rp, protection_standard=0):
    """
    Calculate Expected Annual Exposure (EAE) and Expected Annual Damage (EAD).
    """
    if not damages_by_rp:
        return 0, 0
    
    sorted_rps = sorted(damages_by_rp.keys())
    sorted_damages = [damages_by_rp[rp] for rp in sorted_rps]
    sorted_lengths = [lengths_by_rp.get(rp, 0) for rp in sorted_rps]
    
    if protection_standard > 0 and protection_standard not in sorted_rps:
        interpolated_damage = interpolate_damage(sorted_rps, sorted_damages, protection_standard)
        interpolated_length = interpolate_damage(sorted_rps, sorted_lengths, protection_standard)
        
        insert_idx = np.searchsorted(sorted_rps, protection_standard)
        sorted_rps.insert(insert_idx, protection_standard)
        sorted_damages.insert(insert_idx, interpolated_damage)
        sorted_lengths.insert(insert_idx, interpolated_length)
    
    filtered_rps = [rp for rp in sorted_rps if rp >= protection_standard]
    filtered_damages = [damages_by_rp[rp] if rp in damages_by_rp else 
                       interpolate_damage(sorted_rps, sorted_damages, rp) 
                       for rp in filtered_rps]
    filtered_lengths = [lengths_by_rp.get(rp, 0) if rp in lengths_by_rp else 
                       interpolate_damage(sorted_rps, sorted_lengths, rp) 
                       for rp in filtered_rps]
    
    filtered_probs = [1/rp for rp in filtered_rps]
    
    if len(filtered_damages) >= 2:
        ead_value = np.trapz(y=filtered_damages[::-1], x=filtered_probs[::-1])
        eae_value = np.trapz(y=filtered_lengths[::-1], x=filtered_probs[::-1])
    elif len(filtered_damages) == 1:
        ead_value = filtered_probs[0] * filtered_damages[0]
        eae_value = filtered_probs[0] * filtered_lengths[0]
    else:
        ead_value = 0
        eae_value = 0
    
    return ead_value, eae_value

def adjust_return_periods_and_protection(original_rps, protection_standard, basin_changes, temp_scenario):
    """
    Adjust return periods and protection standard based on climate change factors for a given temperature scenario.
    
    Args:
        original_rps: List of original return periods [10, 20, 30, 40, 50, 75, 100, 200, 500]
        protection_standard: Original protection standard (return period)
        basin_changes: Row from basin dataframe with climate change factors
        temp_scenario: Temperature scenario ('15', '20', '30', '40')
    
    Returns:
        Tuple of (adjusted_return_periods, adjusted_protection_standard)
    """
    # Get climate change factors for anchor points (10, 100, 500 years)
    change_10 = basin_changes.get(f'10_rp_change_{temp_scenario}', 0)*-1
    change_100 = basin_changes.get(f'100_rp_change_{temp_scenario}', 0)*-1
    change_500 = basin_changes.get(f'500_rp_change_{temp_scenario}', 0)*-1
    
    # Handle NaN values (no change)
    if pd.isna(change_10):
        change_10 = 0
    if pd.isna(change_100):
        change_100 = 0
    if pd.isna(change_500):
        change_500 = 0
   
    # Convert percentage changes to factors
    factor_10 = 1 + (change_10 / 100)
    factor_100 = 1 + (change_100 / 100)
    factor_500 = 1 + (change_500 / 100)
    
    # Anchor points for interpolation
    anchor_rps = [10, 100, 500]
    anchor_factors = [factor_10, factor_100, factor_500]
    
    # Interpolate factors for all return periods
    interp_func = interp1d(anchor_rps, anchor_factors, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    
    # Adjust return periods
    adjusted_rps = []
    for rp in original_rps:
        factor = float(interp_func(rp))
        adjusted_rp = rp * factor
        # Ensure return periods don't become negative or too small
        adjusted_rp = max(adjusted_rp, 1.0)
        adjusted_rps.append(adjusted_rp)
    
    # Adjust protection standard using the same interpolation
    adjusted_protection_standard = protection_standard
    if protection_standard > 0:
        factor = float(interp_func(protection_standard))
        adjusted_protection_standard = protection_standard * factor
        # Ensure protection standard doesn't become negative or too small
        adjusted_protection_standard = max(adjusted_protection_standard, 1.0)
    
    return adjusted_rps, adjusted_protection_standard

def process_country_climate_risks(country_file_path, basin_file_path):
    """
    Process climate-adjusted risks for a country.
    
    Args:
        country_file_path: Path to country damage file (e.g., 'AUT_roads_river_risk.parquet')
        basin_file_path: Path to basin climate factors file ('basins_with_return_periods.parquet')
    
    Returns:
        DataFrame with original and climate-adjusted risks
    """
    
    print(f"Loading damage data from {country_file_path}...")
    damage_df = gpd.read_parquet(country_file_path).to_crs(3035)
    print(f"Loading basin data from {basin_file_path}...")
    basin_df = gpd.read_parquet(basin_file_path).to_crs(3035)

    # Original return periods
    original_rps = [10, 20, 30, 40, 50, 75, 100, 200, 500]
    temp_scenarios = ['15', '20', '30', '40']  # 1.5°, 2.0°, 3.0°, 4.0°C

    # Create spatial index for fast basin lookup
    print("Building spatial index for basins...")
    basin_sindex = basin_df.sindex

    # Clip extreme values at 99.9th percentile
    for rp, temp in itertools.product([10, 100, 500], temp_scenarios):
        col_name = f'{rp}_rp_change_{temp}'
        if col_name in basin_df.columns:
            percentile_999 = basin_df[col_name].quantile(0.999)
            basin_df[col_name] = basin_df[col_name].clip(upper=percentile_999)
        
    # Prepare basin lookup data
    basin_lookup = {}
    for idx, basin_row in basin_df.iterrows():
        basin_lookup[idx] = {
            'HYBAS_ID': basin_row['HYBAS_ID'],
            **{f'{rp}_rp_change_{temp}': basin_row.get(f'{rp}_rp_change_{temp}', np.nan) 
               for rp in [10, 100, 500] for temp in temp_scenarios}
        }

    print(f"Processing {len(damage_df)} infrastructure objects...")

    results = []

    for idx, row in damage_df.reset_index().iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(damage_df)} objects...")
        
        # Fast basin lookup using spatial index
        geometry = row['geometry']
        centroid = geometry.centroid if hasattr(geometry, 'centroid') else geometry
        
        # Find potential basin matches using spatial index
        possible_matches_idx = list(basin_sindex.intersection(centroid.bounds))
        
        # Find actual basin containing the point
        basin_data = None
        for basin_idx in possible_matches_idx:
            if basin_df.iloc[basin_idx]['geometry'].contains(centroid):
                basin_data = basin_lookup[basin_idx]
                break

        # If no basin found, use default values
        if basin_data is None:
            basin_data = {
                'HYBAS_ID': None,
                **{f'{rp}_rp_change_{temp}': np.nan for rp in [10, 100, 500] for temp in temp_scenarios}
            }
            
        # Extract original damage and exposure values
        damages_by_rp = {}
        lengths_by_rp = {}
        
        for rp in original_rps:
            damage_col = f'mean_damage_{rp}'
            exposure_col = f'exposure_{rp}'
            
            if damage_col in row and not pd.isna(row[damage_col]):
                damages_by_rp[rp] = row[damage_col]
            if exposure_col in row and not pd.isna(row[exposure_col]):
                lengths_by_rp[rp] = row[exposure_col]
        
        # Calculate original risks
        original_protection_standard = row.get('protection_standard', 0)
        original_ead, original_eae = calculate_eae_ead(
            damages_by_rp, lengths_by_rp, original_protection_standard
        )
        
        result_row = {
            'osm_id': row.get('osm_id'),
            'object_type': row.get('object_type'),
            'HYBAS_ID': basin_data['HYBAS_ID'],
            'protection_standard': original_protection_standard,
            'original_EAD': original_ead,
            'original_EAE': original_eae
        }
        
        # Calculate climate-adjusted risks for each temperature scenario
        for temp in temp_scenarios:
            # Adjust return periods and protection standard based on climate change
            adjusted_rps, adjusted_protection_standard = adjust_return_periods_and_protection(
                original_rps, original_protection_standard, basin_data, temp
            )
            
            # Create new damage/length mappings with adjusted return periods
            adjusted_damages_by_rp = {}
            adjusted_lengths_by_rp = {}
            
            for i, orig_rp in enumerate(original_rps):
                adj_rp = adjusted_rps[i]
                if orig_rp in damages_by_rp:
                    adjusted_damages_by_rp[adj_rp] = damages_by_rp[orig_rp]
                if orig_rp in lengths_by_rp:
                    adjusted_lengths_by_rp[adj_rp] = lengths_by_rp[orig_rp]
            
            # Calculate climate-adjusted risks with adjusted protection standard
            climate_ead, climate_eae = calculate_eae_ead(
                adjusted_damages_by_rp, adjusted_lengths_by_rp, adjusted_protection_standard
            )
            
            # Store results
            temp_label = f"{float(temp)/10:.1f}C"
            result_row[f'EAD_{temp_label}'] = climate_ead
            result_row[f'EAE_{temp_label}'] = climate_eae
            result_row[f'protection_standard_{temp_label}'] = adjusted_protection_standard
            result_row[f'EAD_change_{temp_label}'] = ((climate_ead - original_ead) / original_ead * 100) if original_ead > 0 else 0
            result_row[f'EAE_change_{temp_label}'] = ((climate_eae - original_eae) / original_eae * 100) if original_eae > 0 else 0

        results.append(result_row)

    print("Climate risk calculation completed!")
    return pd.DataFrame(results)

def get_river_files(directory):
    """
    Get all river risk files from the directory.
    
    Args:
        directory: Path to the directory containing risk files
        
    Returns:
        List of paths to river risk files
    """
    river_files = []
    directory = Path(directory)
    
    for file in directory.glob("*.parquet"):
        if "_river_risk.parquet" in file.name:
            river_files.append(file)
    
    return sorted(river_files)

def organize_files_by_country_and_infra(file_list):
    """
    Organize files by country and infrastructure type.
    
    Args:
        file_list: List of file paths
        
    Returns:
        Dictionary organized by country and infrastructure type
    """
    organized = {}
    
    for file_path in file_list:
        filename = file_path.name
        parts = filename.replace("_river_risk.parquet", "").split("_")
        
        if len(parts) >= 2:
            country = parts[0]
            infra_type = "_".join(parts[1:])  # Handle multi-word infrastructure types
            
            if country not in organized:
                organized[country] = {}
            if infra_type not in organized[country]:
                organized[country][infra_type] = []
                
            organized[country][infra_type].append(file_path)
    
    return organized

def main():
    """
    Main function to process all river risk files.
    """
    
    # Configuration
    data_directory = Path(r"X:\eks510\MIRACA_results")
    basin_file_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\12_repositories\AssetRisk_PanEU\src\basins_with_return_periods.parquet")
    
    # Check if paths exist
    if not data_directory.exists():
        print(f"Error: Data directory does not exist: {data_directory}")
        return
    
    if not basin_file_path.exists():
        print(f"Error: Basin file does not exist: {basin_file_path}")
        return
    
    # Get all river files
    print("Scanning for river risk files...")
    river_files = get_river_files(data_directory)
    print(f"Found {len(river_files)} river risk files")
    
    if not river_files:
        print("No river risk files found!")
        return
    
    # Organize files by country and infrastructure type
    organized_files = organize_files_by_country_and_infra(river_files)
    
    # Process each file
    total_files = len(river_files)
    processed_files = 0
    failed_files = []
    
    start_time = time.time()
    
    for country, infra_types in organized_files.items():
        print(f"\n{'='*50}")
        print(f"Processing country: {country}")
        print(f"{'='*50}")
        
        for infra_type, files in infra_types.items():
            for file_path in files:
                processed_files += 1
                print(f"\n[{processed_files}/{total_files}] Processing: {file_path.name}")
                
                try:
                    # Process the file
                    results_df = process_country_climate_risks(file_path, basin_file_path)
                    
                    # Generate output filename
                    original_name = file_path.stem  # Remove .parquet extension
                    new_name = original_name.replace("_river_risk", "_river_climate_risk")
                    output_path = file_path.parent / f"{new_name}.parquet"
                    
                    # Save results
                    print(f"Saving results to: {output_path}")
                    results_df.to_parquet(output_path, index=False)
                    
                    # Print summary statistics
                    print(f"Processed {len(results_df)} objects")
                    max_changes = {}
                    for temp in ['1.5C', '2.0C', '3.0C', '4.0C']:
                        if f'EAD_{temp}' in results_df.columns:
                            max_change = results_df[f'EAD_{temp}'].max()
                            max_changes[temp] = max_change
                    
                    if max_changes:
                        print("Maximum EAD values by temperature scenario:")
                        for temp, max_val in max_changes.items():
                            print(f"  {temp}: {max_val:.2e}")
                    
                except Exception as e:
                    print(f"ERROR processing {file_path.name}: {str(e)}")
                    failed_files.append(file_path.name)
                    continue
                
                # Progress update
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / processed_files
                remaining_files = total_files - processed_files
                estimated_time_remaining = remaining_files * avg_time_per_file
                
                print(f"Progress: {processed_files}/{total_files} files completed")
                print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {processed_files}")
    print(f"Successful: {processed_files - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()