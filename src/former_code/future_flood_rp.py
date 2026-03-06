import os
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from exactextract import exact_extract
import matplotlib.pyplot as plt

def main():
    """
    Process basin data with return period changes and save as parquet.
    """
    
    # File paths
    file_path_10 = Path("disEnsemble_highExtremes_10.nc")
    file_path_100 = Path("disEnsemble_highExtremes_100.nc")
    file_path_500 = Path("disEnsemble_highExtremes_500.nc")
    basin_file_path = Path("hybas_eu_lev08_v1c.shp")
    nuts_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\12_repositories\AssetRisk_PanEU\book\NUTS_RG_20M_2024_3035.geojson")
    
    # Output path for parquet file
    output_parquet_path = Path("basins_abs_shift_return_periods.parquet")
    
    # ISO3 to ISO2 country code mapping
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
    
    print("Loading datasets...")
    # Load return period datasets
    rp10 = xr.open_dataset(file_path_10)
    rp100 = xr.open_dataset(file_path_100)
    rp500 = xr.open_dataset(file_path_500)
    
    # Set spatial dimensions and CRS for rp500 (apply to all if needed)
    rp500 = rp500.rio.set_spatial_dims(x_dim='x', y_dim='y')
    rp500 = rp500.rio.write_crs(3035)
    
    print("Loading geographic data...")
    # Load NUTS and basins data
    nuts = gpd.read_file(nuts_path)
    nuts = nuts.loc[nuts.CNTR_CODE.isin(list(iso3_to_iso2.values()))]
    
    basins = gpd.read_file(basin_file_path).to_crs(3035)
    basins = basins.clip(nuts)
    basins = basins.reset_index(drop=True)
    
    print("Processing return period changes...")
    # Define temperature scenarios
    temp_scenarios = ['15', '20', '30', '40']
    
    # Process each return period dataset
    rp_datasets = [rp10, rp100, rp500]
    rp_prefixes = ['10', '100', '500']
    
    for iter_, (rpmap, prefix) in enumerate(zip(rp_datasets, rp_prefixes)):
        print(f"Processing {prefix}-year return period...")
        
        # Set spatial dimensions and CRS for all datasets
        rpmap = rpmap.rio.set_spatial_dims(x_dim='x', y_dim='y')
        rpmap = rpmap.rio.write_crs(3035)
        
        # Get baseline return level
        baseline = rpmap['baseline_return_level']
        
        for temp in temp_scenarios:
            print(f"  Processing temperature scenario {temp}°C...")
            
            # Get baseline_rp_shift for this temperature
            rp_shift_var = f'baseline_rp_shift_{temp}'
            rp_shift = rpmap[rp_shift_var]
            
            # Assign abs difference: (new - old) / old 
            # new = baseline_rp_shift, old = baseline_return_level
            perc_change = rp_shift #((rp_shift - baseline) / baseline) 
            
            # Extract mean values for each basin
            df = exact_extract(
                perc_change.transpose('y', 'x'), 
                basins, 
                'median', 
                output='pandas',
                progress=True
            )
            
            # Rename column with appropriate prefix
            new_column_name = f"{prefix}_rp_change_{temp}"
            df = df.rename(columns={"median": new_column_name})
            
            # Merge with basins
            basins = basins.merge(df, left_index=True, right_index=True)
    
    print("Saving to parquet file...")
    # Save as parquet file
    basins.to_parquet(output_parquet_path)
    print(f"Basin data with return periods saved to: {output_parquet_path}")
        
    # Print summary information
    print("\nSummary:")
    print(f"Total basins processed: {len(basins)}")
    print(f"New columns added: {[col for col in basins.columns if any(prefix in col for prefix in ['10_', '100_', '500_'])]}")
    print(f"File size: {output_parquet_path.stat().st_size / (1024*1024):.2f} MB")
    
    return basins

if __name__ == "__main__":
    basins_result = main()