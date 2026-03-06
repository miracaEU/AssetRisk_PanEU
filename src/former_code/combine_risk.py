import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from scipy import integrate
from damagescanner.core import DamageScanner
from damagescanner.config import DICT_CIS_VULNERABILITY_FLOOD
from damagescanner.osm import read_osm_data
from damagescanner import download
from tqdm import tqdm

iso3_to_iso2 = {"PRT": "PT", "FRA": "FR", "DEU": "DE", "ESP": "ES", "ITA": "IT", "NLD": "NL", "BEL": "BE", "POL": "PL", "CZE": "CZ", "HUN": "HU", "SVK": "SK", "AUT": "AT", "CHE": "CH", "SVN": "SI", "HRV": "HR", "ROU": "RO", "BGR": "BG", "GRC": "EL", "FIN": "FI", "SWE": "SE", "DNK": "DK", "EST": "EE", "LVA": "LV", "LTU": "LT", "IRL": "IE", "LUX": "LU", "MLT": "MT", "CYP": "CY", "NOR": "NO", "ISL": "IS"}

def interpolate_damage(rp_values, damage_values, target_rp):
    idx = np.searchsorted(rp_values, target_rp)
    
    if idx < len(rp_values) and rp_values[idx] == target_rp:
        return damage_values[idx]
    
    if idx == 0:
        return damage_values[0]
    
    if idx == len(rp_values):
        return damage_values[-1]
    
    rp_low, rp_high = rp_values[idx-1], rp_values[idx]
    damage_low, damage_high = damage_values[idx-1], damage_values[idx]
    
    return damage_low + (target_rp - rp_low) * (damage_high - damage_low) / (rp_high - rp_low)

def calculate_eae_ead(damages_by_rp, lengths_by_rp, protection_standard=0):
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
        ead_value = np.trapezoid(y=filtered_damages[::-1], x=filtered_probs[::-1])
        eae_value = np.trapezoid(y=filtered_lengths[::-1], x=filtered_probs[::-1])
    elif len(filtered_damages) == 1:
        ead_value = filtered_probs[0] * filtered_damages[0]
        eae_value = filtered_probs[0] * filtered_lengths[0]
    else:
        ead_value = 0
        eae_value = 0
    
    return ead_value, eae_value

def load_protection_standards(country_iso3, asset_type):
    
    try:
        print("Loading protection standards...")
        
        # Load infrastructure features
        infrastructure_path = download.get_country_geofabrik(country_iso3)
        
        print(f"Reading feature data for {country_iso3} and {asset_type}...")
        features = read_osm_data(infrastructure_path, asset_type=asset_type)
        
        # Load protection standard map
        protection_standard_map = xr.open_dataset("floodProtection_v2019_paper3.tif", engine="rasterio")
        protection_standard_map = protection_standard_map.coarsen(x=10, y=10, boundary="trim").mean()
        
        if hasattr(protection_standard_map, 'rio'):
            protection_standard_map.rio.write_crs("EPSG:3035", inplace=True)
        
        # Get country bounds
        country_iso2 = iso3_to_iso2[country_iso3]
        nuts2 = gpd.read_file("NUTS_RG_20M_2024_3035.geojson")
        country_bounds_3035 = nuts2.loc[(nuts2.LEVL_CODE == 0) & (nuts2.CNTR_CODE == country_iso2)].bounds

        protection_standard_map = protection_standard_map.rio.clip_box(
            minx=country_bounds_3035.minx.values[0],
            miny=country_bounds_3035.miny.values[0],
            maxx=country_bounds_3035.maxx.values[0],
            maxy=country_bounds_3035.maxy.values[0]
        )

        protection_standard = DamageScanner(
            protection_standard_map, 
            features.to_crs(3035), 
            curves=pd.DataFrame(), 
            maxdam=pd.DataFrame()
        ).exposure(asset_type=asset_type)
        
        protection_standard['design_standard'] = protection_standard['values'].apply(
            lambda x: np.max(x) if len(x) > 0 else 0
        )
        
        # Create lookup dictionary for protection standards
        asset_region_ds = dict(zip(protection_standard.osm_id.values, protection_standard.design_standard.values))
        print(f"Loaded protection standards for {len(asset_region_ds)} assets")
        
        return asset_region_ds
        
    except Exception as e:
        print(f"Warning: Could not load protection standards: {e}")
        print("Assuming no protection")
        return {}

if __name__ == "__main__":
    country_iso3 = sys.argv[1]
    asset_type = sys.argv[2]
    
    return_periods = [10, 20, 30, 40, 50, 75, 100, 200, 500]
    
    # Load all return period results
    input_dir = Path(r"X:\eks510\MIRACA_results\river_rp_results")
    all_results = {}
    
    for rp in return_periods:
        file_path = input_dir / f"{country_iso3}_{asset_type}_river_RP{rp}.csv"
        if file_path.exists():
            all_results[rp] = pd.read_csv(file_path)
            print(f"Loaded RP{rp}: {len(all_results[rp])} features")
    
    if not all_results:
        print("No return period results found!")
        sys.exit(1)
    
    # Get vulnerability curve names
    ci_system = DICT_CIS_VULNERABILITY_FLOOD[asset_type]
    unique_curves = set([x for xs in ci_system.values() for x in xs])
    curve_names = list(unique_curves)
    
    # Load protection standards
    asset_region_ds = load_protection_standards(country_iso3, asset_type)
    
    # Get largest RP for base data
    largest_rp = max(all_results.keys())
    largest_rp_data = all_results[largest_rp].copy()
    
    # Get all unique asset IDs
    all_asset_ids = set()
    for results in all_results.values():
        all_asset_ids.update(results['osm_id'])
    
    # Calculate risk for each curve and each asset
    collect_risks = {}
    collect_exposures = {}
    mean_damages = {}
    lengths = {}
    
    # Organize data by return period
    for rp in return_periods:
        if rp in all_results:
            results = all_results[rp]
            mean_damages[rp] = dict(zip(results['osm_id'], results[f'mean_damage_{rp}']))
            lengths[rp] = dict(zip(results['osm_id'], results[f'length_{rp}']))
    
    for curve in tqdm(curve_names, total=len(curve_names), desc="Processing curves"):
        risk_values = []
        exposure_values = []
        
        for asset_id in all_asset_ids:
            # Get protection standard for this asset
            design_standard = asset_region_ds.get(asset_id, 0)
            
            # Get damage and length values by return period
            rp_damages = {}
            rp_lengths = {}
            
            for rp in return_periods:
                if rp in all_results:
                    results = all_results[rp]
                    asset_rows = results[results['osm_id'] == asset_id]
                    
                    if len(asset_rows) > 0:
                        if str(curve) in asset_rows.columns:
                            damage = asset_rows[str(curve)].iloc[0]
                            if not pd.isna(damage):
                                rp_damages[rp] = damage
                        
                        if rp in lengths and asset_id in lengths[rp]:
                            rp_lengths[rp] = lengths[rp][asset_id]
            
            # Calculate EAD and EAE
            ead, eae = calculate_eae_ead(rp_damages, rp_lengths, design_standard)
            
            risk_values.append(ead)
            exposure_values.append(eae)
        
        collect_risks[curve] = risk_values
        collect_exposures[curve] = exposure_values
    
    # Create final results
    final_results = pd.DataFrame()
    final_results['osm_id'] = list(all_asset_ids)
    
    # Add object_type and geometry from largest RP data
    asset_info = largest_rp_data[['osm_id', 'object_type', 'geometry']].drop_duplicates()
    final_results = final_results.merge(asset_info, on='osm_id', how='left')
    
    # Convert geometry from WKT to shapely objects
    from shapely import wkt
    final_results['geometry'] = final_results['geometry'].apply(wkt.loads)
    
    # Convert to GeoDataFrame
    final_results = gpd.GeoDataFrame(final_results, geometry='geometry')
    
    # Add risk values for each curve
    for curve in curve_names:
        final_results[str(curve)] = collect_risks[curve]
    
    # Calculate mean EAD and EAE
    all_risks = pd.DataFrame.from_dict(collect_risks)
    all_exposures = pd.DataFrame.from_dict(collect_exposures)
    
    final_results['EAD'] = all_risks.mean(axis=1, skipna=True)
    final_results['EAE'] = all_exposures.mean(axis=1, skipna=True)
    
    # Add hazard type
    final_results['hazard_type'] = 'river'
    
    # Add protection standards
    if asset_region_ds:
        final_results['protection_standard'] = final_results['osm_id'].map(asset_region_ds)
    
    # Add damage and exposure values for each return period
    for rp in return_periods:
        if rp in mean_damages:
            final_results[f'mean_damage_{rp}'] = final_results['osm_id'].map(mean_damages[rp])
        if rp in lengths:
            final_results[f'exposure_{rp}'] = final_results['osm_id'].map(lengths[rp])
    
    # Save results
    output_dir = Path("/scistor/ivm/eks510/MIRACA_results")
    output_file = output_dir / f"{country_iso3}_{asset_type}_river_risk.parquet"
    final_results.to_parquet(output_file, index=False)
    
    print(f"Saved final results to {output_file}")
    print(f"Total features: {len(final_results)}")
    print(f"Mean EAD: {final_results['EAD'].mean():.2f}")
    print(f"Total EAD: {final_results['EAD'].sum():.2f}")