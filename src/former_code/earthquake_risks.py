"""
Earthquake Risk Assessment Module

This module provides functions to assess earthquake risk for infrastructure
using fragility curves and probabilistic damage estimation.
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
import traceback
import concurrent.futures

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

# Earthquake vulnerability dictionary
DICT_CIS_VULNERABILITY_EARTHQUAKE = {
    "roads": {
        "motorway": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "motorway_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "trunk": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "trunk_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "primary": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "primary_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "secondary": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "secondary_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "tertiary": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "tertiary_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "residential": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "road": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "unclassified": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "track": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "service": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
    },
    "main_roads": {
        "motorway": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "motorway_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "trunk": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "trunk_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "primary": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "primary_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "secondary": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "secondary_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "tertiary": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
        "tertiary_link": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
    },
    "rail": {
        "rail": ["E8.1", "E8.2", "E8.3", "E8.4", "E8.5", "E8.6", "E8.7", "E8.8", "E8.9", "E8.10",
                "E8.11", "E8.12", "E8.13", "E8.14", "E8.15", "E8.16", "E8.17", "E8.18", "E8.19", "E8.20"],
    },
    "air": {
        "aerodrome": ["E9.2", "E9.3", "E9.4"],
        "terminal": ["E9.2", "E9.3", "E9.4"],
        "runway": ["E7.2", "E7.3", "E7.4", "E7.5", "E7.6", "E7.7", "E7.8", "E7.9", "E7.10"],
    },
    "telecom": {
        "mast": ["E11.1"],
        "communications_tower": ["E3.1", "E3.2"],
        "tower": ["E3.1", "E3.2"],

    },
    "education": {
        "school": ["E21.26-C", "E21.27-C", "E21.29-C", "E21.30-C", "E21.31-C", "E21.32-C", 
                  "E21.33-C", "E21.34-C", "E21.35-C", "E21.36-C", "E21.37-C", "E21.38-C", "E21.39-C", 
                  "E21.40-C", "E21.41-C", "E21.42-C", "E21.43-C", "E21.48-C", "E21.49-C", "E21.50-C", 
                  "E21.51-C", "E21.52-C", "E21.53-C", "E21.54-C", "E21.55-C", "E21.56-C", "E21.57-C", 
                  "E21.58-C", "E21.59-C", "E21.60-C", "E21.61-C"],
        "kindergarten": ["E21.26-C", "E21.27-C", "E21.29-C", "E21.30-C", "E21.31-C", "E21.32-C", 
                  "E21.33-C", "E21.34-C", "E21.35-C", "E21.36-C", "E21.37-C", "E21.38-C", "E21.39-C", 
                  "E21.40-C", "E21.41-C", "E21.42-C", "E21.43-C", "E21.48-C", "E21.49-C", "E21.50-C", 
                  "E21.51-C", "E21.52-C", "E21.53-C", "E21.54-C", "E21.55-C", "E21.56-C", "E21.57-C", 
                  "E21.58-C", "E21.59-C", "E21.60-C", "E21.61-C"],
        "college": ["E21.26-C", "E21.27-C", "E21.29-C", "E21.30-C", "E21.31-C", "E21.32-C", 
                  "E21.33-C", "E21.34-C", "E21.35-C", "E21.36-C", "E21.37-C", "E21.38-C", "E21.39-C", 
                  "E21.40-C", "E21.41-C", "E21.42-C", "E21.43-C", "E21.48-C", "E21.49-C", "E21.50-C", 
                  "E21.51-C", "E21.52-C", "E21.53-C", "E21.54-C", "E21.55-C", "E21.56-C", "E21.57-C", 
                  "E21.58-C", "E21.59-C", "E21.60-C", "E21.61-C"],
        "university": ["E21.26-C", "E21.27-C", "E21.29-C", "E21.30-C", "E21.31-C", "E21.32-C", 
                  "E21.33-C", "E21.34-C", "E21.35-C", "E21.36-C", "E21.37-C", "E21.38-C", "E21.39-C", 
                  "E21.40-C", "E21.41-C", "E21.42-C", "E21.43-C", "E21.48-C", "E21.49-C", "E21.50-C", 
                  "E21.51-C", "E21.52-C", "E21.53-C", "E21.54-C", "E21.55-C", "E21.56-C", "E21.57-C", 
                  "E21.58-C", "E21.59-C", "E21.60-C", "E21.61-C"],
        "library": ["E21.26-C", "E21.27-C", "E21.29-C", "E21.30-C", "E21.31-C", "E21.32-C", 
                  "E21.33-C", "E21.34-C", "E21.35-C", "E21.36-C", "E21.37-C", "E21.38-C", "E21.39-C", 
                  "E21.40-C", "E21.41-C", "E21.42-C", "E21.43-C", "E21.48-C", "E21.49-C", "E21.50-C", 
                  "E21.51-C", "E21.52-C", "E21.53-C", "E21.54-C", "E21.55-C", "E21.56-C", "E21.57-C", 
                  "E21.58-C", "E21.59-C", "E21.60-C", "E21.61-C"],
    },
    "healthcare": {
        "hospital": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "clinic": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "doctors": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "pharmacy": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "dentist": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "physiotherapist": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "alternative": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "laboratory": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "optometrist": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "rehabilitation": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "blood_donation": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
        "birthing_center": ["E21.67-C", "E21.68-C", "E21.69-C", "E21.70-C", "E21.71-C", "E21.72-C"],
    },
    "power": {
        "line": ["E6.1", "E6.2", "E6.3", "E6.4"],
        "cable": ["E6.1", "E6.2", "E6.3", "E6.4"],
        "minor_line": ["E6.1", "E6.2", "E6.3", "E6.4"],
        "plant": ["E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6", "E1.7", "E1.8"],
        "generator": ["E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6", "E1.7", "E1.8"],
        "substation": ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"],
        "transformer": ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"],
        "pole": ["E4.1", "E4.2", "E4.3", "E4.4"],
        "portal": ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"],
        "tower": ["E3.1", "E3.2"],
        "terminal": ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"],
        "switch": ["E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E2.7", "E2.8", "E2.9"],
        "catenary_mast": ["E4.1", "E4.2", "E4.3", "E4.4"],
    },
}

# Infrastructure damage values dictionary (same as your original)
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
}

def _generate_fragility_curves_from_parameters(param_df, pga_range=None):
    """
    Generate fragility curves from median and beta parameters.
    
    The fragility function is: P(DS ≥ ds | PGA) = Φ((ln(PGA) - ln(median)) / beta)
    where Φ is the standard normal cumulative distribution function.
    """
    from scipy.stats import norm
    
    if pga_range is None:
        # Default PGA range from 0.01g to 2.0g
        pga_range = np.logspace(-2, np.log10(2.0), 100)
    
    # Initialize output DataFrame
    fragility_curves = pd.DataFrame(index=pga_range)
    
    # Get all unique curve_id and damage_state combinations (excluding beta columns)
    damage_state_columns = [col for col in param_df.columns if not str(col[1]).endswith('_beta')]
    
    for col in damage_state_columns:
        curve_id, damage_state = col
        beta_col = (curve_id, f'{damage_state}_beta')
        
        if beta_col not in param_df.columns:
            print(f"Warning: No beta column found for {curve_id} - {damage_state}")
            continue
        
        # Extract median and beta values
        median_val = param_df.iloc[0][col]
        beta_val = param_df.iloc[0][beta_col]
        
        # Skip if values are invalid
        if pd.isna(median_val) or pd.isna(beta_val) or median_val <= 0 or beta_val <= 0:
            continue
        
        # Handle comma decimal separator if present
        if isinstance(median_val, str):
            median_val = float(median_val.replace(',', '.'))
        if isinstance(beta_val, str):
            beta_val = float(beta_val.replace(',', '.'))
            
        # Calculate fragility curve: P(DS >= ds | PGA) = Φ((ln(PGA) - ln(median)) / beta)
        ln_pga = np.log(pga_range)
        ln_median = np.log(median_val)
        
        # Calculate exceedance probabilities
        exceedance_probs = norm.cdf((ln_pga - ln_median) / beta_val)
        
        # Store in output DataFrame with proper multi-index column
        fragility_curves[(curve_id, damage_state)] = exceedance_probs
    
    # Set up proper multi-index columns
    fragility_curves.columns = pd.MultiIndex.from_tuples(fragility_curves.columns)
    
    fragility_curves.to_csv("fragility_curves.csv", index_label="PGA")

    return fragility_curves


def _get_damage_per_object_fragility(asset, fragility_curves, curve_id, cell_area_m2, damage_ratios=None):
    """
    Compute damage for a single asset using fragility curves.
    Dynamically handles any number of damage states (1-5) based on available curve data.
    """
    # Default damage ratios - handles up to 5 damage states including severe
    if damage_ratios is None:
        damage_ratios = {
            'minor': 0.05,     # slight damage (5% of replacement cost)
            'moderate': 0.2,   # moderate damage (20% of replacement cost)
            'extensive': 0.7,  # extensive damage (70% of replacement cost)
            'severe': 0.85,    # severe damage (85% of replacement cost)
            'complete': 1.0,   # complete damage (100% of replacement cost)
            'collapse': 1.0    # collapse (100% - could be set higher for additional costs)
        }
    
    # Get coverage based on geometry type
    if asset.geometry.geom_type in ("Polygon", "MultiPolygon"):
        coverage = np.array(asset["coverage"]) * cell_area_m2
    elif asset.geometry.geom_type in ("LineString", "MultiLineString"):
        coverage = asset["coverage"]
    elif asset.geometry.geom_type in ("Point"):
        coverage = 1
    else:
        raise ValueError(f"Geometry type {asset.geometry.geom_type} not supported")
    
    # Get PGA values for this asset
    pga_values = np.array(asset["values"])
    
    # Find all available damage states for this curve_id
    available_states = []
    state_columns = {}
    
    for col in fragility_curves.columns:
        if col[0] == curve_id:  # col is (curve_id, damage_state)
            state_name = col[1]
            available_states.append(state_name)
            state_columns[state_name] = col
    
    if not available_states:
        available_curves = fragility_curves.columns.get_level_values(0).unique()
        raise KeyError(f"Fragility curve {curve_id} not found. Available curves: {list(available_curves)}")
    
    # Sort damage states by severity (using damage ratios as proxy for ordering)
    # States not in damage_ratios get default ordering
    state_severity = {state: damage_ratios.get(state, 0.5) for state in available_states}
    sorted_states = sorted(available_states, key=lambda x: state_severity[x])
    
    
    # Check for unmapped states
    unmapped_states = [state for state in available_states if state not in damage_ratios]
    if unmapped_states:
        print(f"Warning: Unmapped damage states found: {unmapped_states}")
    
    # Initialize array to store expected damage ratios for each PGA value
    expected_damage_ratios = np.zeros_like(pga_values, dtype=float)
    
    # Calculate expected damage ratio for each PGA value
    for i, pga in enumerate(pga_values):
        # Skip invalid PGA values
        if pd.isna(pga) or pga < 0:
            expected_damage_ratios[i] = 0.0
            continue
            
        # Get exceedance probabilities for all available damage states
        exceedance_probs = {}
        for state in sorted_states:
            try:
                prob = np.interp(
                    pga, 
                    fragility_curves.index, 
                    fragility_curves[state_columns[state]].values
                )
                exceedance_probs[state] = max(0.0, min(1.0, prob))  # Clamp to [0,1]
            except Exception as e:
                #print(f"Error interpolating {state} for PGA {pga}: {e}")
                exceedance_probs[state] = 0.0
        
        
        # Convert exceedance probabilities to individual damage state probabilities
        # States are ordered from least to most severe
        individual_probs = {}
        prev_exceedance = 0.0
        
        # No damage probability = 1 - probability of exceeding mildest damage state
        prob_no_damage = 1.0 - exceedance_probs[sorted_states[0]] if sorted_states else 1.0
        individual_probs['no_damage'] = max(0.0, prob_no_damage)
        
        # For each damage state: P(exactly this state) = P(exceed this) - P(exceed next worse state)
        for j, state in enumerate(sorted_states):
            if j == len(sorted_states) - 1:
                # Most severe state: just use its exceedance probability
                individual_probs[state] = exceedance_probs[state]
            else:
                # Intermediate state: difference between this and next worse state
                individual_probs[state] = max(0.0, exceedance_probs[state] - exceedance_probs[sorted_states[j + 1]])
        
        # Normalize probabilities to sum to 1
        total_prob = sum(individual_probs.values())
        if total_prob > 0:
            for key in individual_probs:
                individual_probs[key] /= total_prob
    
        # Calculate expected damage ratio using available damage states
        expected_damage_ratio = individual_probs['no_damage'] * 0.0
        for state in sorted_states:
            if state in damage_ratios:
                expected_damage_ratio += individual_probs[state] * damage_ratios[state]
            else:
                # If damage ratio not defined, assume proportional to position in severity order
                fallback_ratio = (sorted_states.index(state) + 1) / len(sorted_states)
                expected_damage_ratio += individual_probs[state] * fallback_ratio
                print(f"Warning: Using fallback damage ratio {fallback_ratio:.2f} for state '{state}'")
        
        # if i == 0:
        #     print(f"  Expected damage ratio: {expected_damage_ratio:.3f}")
        
        expected_damage_ratios[i] = expected_damage_ratio
    
    # Apply coverage weighting and calculate total damage
    total_damage = np.sum(expected_damage_ratios * coverage) * asset["maximum_damage"]
    
    return total_damage


def _estimate_damage_fragility(features, fragility_curves, curve_id, cell_area_m2, damage_ratios=None):
    """
    Estimate total damage per asset using fragility curves.
    """
    features["damage"] = features.progress_apply(
        lambda _object: _get_damage_per_object_fragility(
            _object, fragility_curves, curve_id, cell_area_m2, damage_ratios
        ),
        axis=1,
    )
    return features


def prepare_fragility_curves(asset_type, fragility_path):
    """
    Prepare fragility curves for earthquake assessment.
    Handles both pre-computed curves and parametric curves (median/beta).
    Supports variable numbers of damage states per asset type (1-5).
    
    Args:
        asset_type (str): Type of infrastructure asset
        fragility_path (str): Path to fragility curve data
        
    Returns:
        tuple: (fragility_curves, multi_curves, maxdam) for use in risk calculation
    """
    # Read fragility data from Excel with multi-header
    try:
        fragility_df = pd.read_excel(fragility_path, sheet_name='E_Frag_PGA', header=[0,1])
        print("Successfully read Excel file with multi-header")
    except Exception as e:
        print(f"Error reading fragility curves: {e}")
        raise
    
    # Get curve IDs for this asset type
    ci_system = DICT_CIS_VULNERABILITY_EARTHQUAKE[asset_type]
    
    # Get all unique curves for this asset type
    unique_curves = set([x for xs in DICT_CIS_VULNERABILITY_EARTHQUAKE[asset_type].values() for x in xs])
    
    # Check if this is parametric data (median/beta) or pre-computed curves
    # Look at the first relevant column for this asset type to determine format
    sample_column = None
    for col in fragility_df.columns:
        curve_id = col[0]
        if curve_id in unique_curves:
            sample_column = col
            break
    
    if sample_column is None:
        available_curves = fragility_df.columns.get_level_values(0).unique()
        raise ValueError(f"No curves found for asset type {asset_type}. "
                        f"Required: {unique_curves}, Available: {list(available_curves)}")
    
    # Check if this sample column contains parametric data
    sample_column_values = [str(val).lower() for val in fragility_df[sample_column].values if pd.notna(val)]
    is_parametric = any('median' in val for val in sample_column_values) and any('beta' in val for val in sample_column_values)
    
    if is_parametric:
        print(f"Detected parametric fragility curves for {asset_type}")
        
        # For parametric curves, we need to extract median/beta parameters and generate curves
        # Filter to only relevant curves for this asset type
        relevant_columns = []
        for col in fragility_df.columns:
            curve_id = col[0]
            if curve_id in unique_curves:
                relevant_columns.append(col)
        
        if not relevant_columns:
            available_curves = fragility_df.columns.get_level_values(0).unique()
            raise ValueError(f"No parametric curves found for asset type {asset_type}. "
                           f"Required: {unique_curves}, Available: {list(available_curves)}")
        
        # Extract parameter data for relevant curves
        # Collect all parameters first, then create DataFrame in one go
        param_dict = {}
        
        for col in relevant_columns:
            curve_id, damage_state = col
            
            # Get the data from this column
            column_data = fragility_df[col]
            
            # Find median and beta values in this column
            median_val = None
            beta_val = None
            
            for cell_value in column_data:
                # Check if this cell contains our target values
                if pd.isna(cell_value):
                    continue
                    
                # Get the row index for this cell
                cell_index = column_data[column_data == cell_value].index[0]
                
                # Check if the previous row contains "median" or "beta"
                if cell_index > 0:
                    prev_cell = column_data.iloc[cell_index - 1] if cell_index - 1 < len(column_data) else None
                    if prev_cell is not None:
                        prev_str = str(prev_cell).lower()
                        if 'median' in prev_str and median_val is None:
                            median_val = cell_value
                        elif 'beta' in prev_str and beta_val is None:
                            beta_val = cell_value
            
            # Store parameters if both found
            if median_val is not None and beta_val is not None:
                param_dict[(curve_id, damage_state)] = median_val
                param_dict[(curve_id, f'{damage_state}_beta')] = beta_val
            else:
                print(f"Warning: Could not find median/beta for {curve_id} - {damage_state}")
        
        # Create DataFrame from collected parameters
        parametric_data = pd.DataFrame([param_dict])
        parametric_data.columns = pd.MultiIndex.from_tuples(parametric_data.columns)
        
        # Generate fragility curves from parameters
        complete_fragility_df = _generate_fragility_curves_from_parameters(parametric_data)
        
    else:
        print(f"Detected pre-computed fragility curves for {asset_type}")
        
        # Extract PGA values from the first column
        pga_values = fragility_df.iloc[:, 0].values
        pga_values = pd.to_numeric(pga_values, errors='coerce')
        
        # Remove any NaN values and corresponding rows
        valid_pga_mask = ~pd.isna(pga_values)
        pga_values = pga_values[valid_pga_mask]
        
        # Get data columns (everything except first column which is PGA)
        data_columns = fragility_df.iloc[:, 1:]
        data_values = data_columns.values[valid_pga_mask]
        
        # Convert to numeric
        data_values = pd.DataFrame(data_values).apply(pd.to_numeric, errors='coerce').values
        
        # Get the multi-index from column headers
        multi_index = data_columns.columns
        
        # Create the complete fragility DataFrame
        complete_fragility_df = pd.DataFrame(data_values, index=pga_values, columns=multi_index)
    
    # Standardize damage state names (flexible mapping for various naming conventions)
    damage_state_mapping = {
        'Slight': 'minor',
        'Minor': 'minor',
        'DS1': 'minor',        # DS1 = minor/slight damage
        'Moderate': 'moderate',
        'DS2': 'moderate',     # DS2 = moderate damage
        'Extensive': 'extensive',
        'DS3': 'extensive',    # DS3 = extensive damage
        'Severe': 'severe',
        'DS4': 'severe',       # DS4 = severe damage
        'Complete': 'complete',
        'DS5': 'complete'      # DS5 = complete damage (total loss)
    }
    
    # Rename columns to standardize damage states
    new_columns = []
    for curve_id, damage_state in complete_fragility_df.columns:
        standardized_state = damage_state_mapping.get(damage_state, damage_state.lower())
        new_columns.append((curve_id, standardized_state))
    
    complete_fragility_df.columns = pd.MultiIndex.from_tuples(new_columns)
    
    # Filter to only include curves needed for this asset type
    filtered_data = {}
    curves_found = set()
    
    for curve_id in unique_curves:
        curve_columns = [col for col in complete_fragility_df.columns if col[0] == curve_id]
        if curve_columns:
            curves_found.add(curve_id)
            for col in curve_columns:
                filtered_data[col] = complete_fragility_df[col].values
    
    if not filtered_data:
        available_curves = complete_fragility_df.columns.get_level_values(0).unique()
        raise ValueError(f"No fragility curves found for asset type {asset_type}. "
                        f"Required: {unique_curves}, Available: {list(available_curves)}")
    
    # Create the filtered fragility DataFrame
    fragility_curves = pd.DataFrame(filtered_data, index=complete_fragility_df.index)
    fragility_curves.columns = pd.MultiIndex.from_tuples(fragility_curves.columns)
    fragility_curves = fragility_curves.fillna(method='ffill').fillna(0)
    
    # Report what was found
    curves_by_states = {}
    for curve_id in curves_found:
        states = [col[1] for col in fragility_curves.columns if col[0] == curve_id]
        curves_by_states[curve_id] = len(states)
    
    curve_type = "parametric" if is_parametric else "pre-computed"
    for curve_id, num_states in curves_by_states.items():
        states = [col[1] for col in fragility_curves.columns if col[0] == curve_id]
        print(f"  {curve_id}: {num_states} damage states ({', '.join(states)})")
    
    # Create multi_curves dictionary for compatibility
    multi_curves = {curve_id: fragility_curves for curve_id in curves_found}
    
    if not multi_curves:
        raise ValueError(f"No valid fragility curves found for asset type {asset_type}")
    
    # Create maximum damage values dataframe
    asset_maxdam_dict = INFRASTRUCTURE_DAMAGE_VALUES[asset_type]
    maxdam_dict = {key: values[1] for key, values in asset_maxdam_dict.items()}
    maxdam = pd.DataFrame.from_dict(maxdam_dict, orient='index').reset_index()
    maxdam.columns = ['object_type', 'damage']
    
    return fragility_curves, multi_curves, maxdam

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
    
    # Apply geometry standardization if needed (reuse your existing functions)
    # For now, keep it simple
    
    # Drop any invalid geometries
    valid_mask = features['geometry'].is_valid
    if not valid_mask.all():
        print(f"Warning: Dropping {(~valid_mask).sum()} invalid geometries")
        features = features[valid_mask]
    
    return features

def get_country_bounds(country_iso3, nuts2_path=None):
    """
    Get country boundaries from NUTS2 data.
    """
    if nuts2_path:
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
        print("NUTS2 data not provided, using simplified approach for country bounds")
        return None, None

def interpolate_damage(rp_values, damage_values, target_rp):
    """
    Interpolate damage value for a specific return period.
    """
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
        ead_value = np.trapezoid(
            y=filtered_damages[::-1], 
            x=filtered_probs[::-1]
        )
        
        eae_value = np.trapezoid(
            y=filtered_lengths[::-1],
            x=filtered_probs[::-1]
        )
    elif len(filtered_damages) == 1:
        ead_value = filtered_probs[0] * filtered_damages[0]
        eae_value = filtered_probs[0] * filtered_lengths[0]
    else:
        ead_value = 0
        eae_value = 0
    
    return ead_value, eae_value

def process_coverage(row, cell_area_m2=None):
    """
    Process coverage based on geometry type.
    """
    coverage = row['coverage']
    geom_type = row['geom_type']
    
    if isinstance(coverage, (list, tuple, np.ndarray)):
        if geom_type in ['Polygon', 'MultiPolygon']:
            return sum(coverage) * cell_area_m2 if len(coverage) > 0 else 0
        else:
            return sum(coverage) if len(coverage) > 0 else 0
    elif isinstance(coverage, (int, float)):
        if geom_type in ['Polygon', 'MultiPolygon']:
            return float(coverage) * cell_area_m2
        else:
            return float(coverage)
    elif coverage is None or pd.isna(coverage):
        return 0
    else:
        try:
            return float(coverage)
        except (TypeError, ValueError):
            return 0

def calculate_damage_earthquake(hazard, features, fragility_curves, multi_curves, maxdam, asset_type):
    """
    Calculate earthquake damage for assets exposed to a hazard using fragility curves.
    """
    # Ensure rioxarray is properly imported
    try:
        import rioxarray
        if not hasattr(hazard, 'rio'):
            hazard = hazard.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            if 'crs' in hazard.attrs:
                hazard = hazard.rio.write_crs(hazard.attrs['crs'], inplace=False)
    except (ImportError, AttributeError):
        pass
    
    # Get exposure (overlay hazard with features)
    results = DamageScanner(
        hazard, 
        features, 
        curves=pd.DataFrame(),  # Empty DataFrame since we're not using vulnerability curves
        maxdam=maxdam
    ).exposure(asset_type=asset_type)
    
    # Now calculate damage using fragility curves for each curve ID
    curve_damages = {}
    
    for curve_id in multi_curves.keys():
        # Calculate damage using this specific fragility curve
        curve_results = results.copy()
        
        # Apply fragility-based damage calculation
        curve_results = _estimate_damage_fragility(
            curve_results, 
            fragility_curves, 
            curve_id, 
            _get_cell_area_m2(features, abs(hazard.rio.resolution()[0]))
        )
        
        # Store damage values for this curve
        curve_damages[curve_id] = curve_results['damage']
    
    # Add all curve damages to the results dataframe
    for curve_id, damages in curve_damages.items():
        results[curve_id] = damages
    
    return results

def format_risk_results_earthquake(largest_rp, all_risks, all_exposures, multi_curves, mean_damages, lengths):
    """
    Format earthquake risk results into the final output DataFrame.
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
    largest_rp['hazard_type'] = 'earthquake'
    
    # Add mean damage and length columns for each return period
    all_return_periods = sorted(list(set(list(mean_damages.keys()) + list(lengths.keys()))))
    
    for rp in all_return_periods:
        if rp in mean_damages:
            largest_rp[f'mean_damage_{rp}'] = largest_rp['osm_id'].map(mean_damages[rp])
        
        if rp in lengths:
            largest_rp[f'exposure_{rp}'] = largest_rp['osm_id'].map(lengths[rp])
    
    # Create final output DataFrame with essential columns
    risk_columns = ['osm_id', 'geometry', 'object_type', 'EAD', 'EAE', 'hazard_type']
    
    # Add length and damage columns
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

def process_return_period_wrapper_earthquake(common_data, args):
    """
    Wrapper that unpacks common data and specific return period data for earthquake assessment.
    """
    return_period, hazard_data = args
    features, fragility_curves, multi_curves, maxdam, asset_type = common_data
    
    # Calculate damage using fragility curves
    results = calculate_damage_earthquake(
        hazard_data, 
        features, 
        fragility_curves, 
        multi_curves, 
        maxdam, 
        asset_type
    )
    
    # Calculate mean damage across all curves
    mean_damage_col = f'mean_damage_{return_period}'
    curve_values = [str(k) for k in multi_curves.keys()]
    results[mean_damage_col] = results[curve_values].mean(axis=1)

    hazard_resolution = abs(hazard_data.rio.resolution()[0])
    
    # Get cell area
    if hasattr(hazard_data, 'rio') and hazard_data.rio.crs:
        import pyproj
        crs = pyproj.CRS.from_user_input(hazard_data.rio.crs)
        if crs.axis_info[0].unit_name == "metre":
            cell_area_m2 = abs(hazard_data.x[1].values - hazard_data.x[0].values) * \
                            abs(hazard_data.y[0].values - hazard_data.y[1].values)
        else:
            cell_area_m2 = _get_cell_area_m2(features, hazard_resolution)
    else:
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
    
    return return_period, results

def assess_earthquake_risk(country_iso3, asset_type, nuts2_path=None, 
                          fragility_path=None, standardize_geom=True):
    """
    Assess earthquake risk for a specific country and asset type.
    
    Args:
        country_iso3 (str): ISO3 code of the country
        asset_type (str): Type of infrastructure to assess
        nuts2_path (str, optional): Path to NUTS2 regions file
        fragility_path (str, optional): Path to fragility curves file
        standardize_geom (bool): Whether to standardize geometries for mixed-type assets
        
    Returns:
        geopandas.GeoDataFrame: Risk assessment results with infrastructure features
    """
    # Load infrastructure data
    features = load_infrastructure_data(country_iso3, asset_type, standardize_geom)
    
    print(f"Loaded {len(features)} infrastructure features")
    
    # If fragility path not provided, use default
    if fragility_path is None:
        fragility_path = "https://zenodo.org/records/10203846/files/Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx?download=1"
    
    # Prepare fragility curves
    print("Preparing earthquake fragility curves...")
    fragility_curves, multi_curves, maxdam = prepare_fragility_curves(asset_type, fragility_path)
    
    # Connect maxdam to features - add this right after loading features and curves
    maxdam_dict = dict(zip(maxdam['object_type'], maxdam['damage']))
    
    try:
        features["maximum_damage"] = features.apply(
            lambda x: maxdam_dict[x["object_type"]], axis=1
        )
    except KeyError:
        missing_object_types = [
            i for i in features.object_type.unique() if i not in maxdam_dict.keys()
        ]
        raise KeyError(
            f"Not all object types in the exposure are included in the maximum damage file: {missing_object_types}"
        )
    
    # Define return periods for earthquakes
    return_periods = [50, 101, 476, 976, 2500, 5000]
    
    # Get country bounds
    country_bounds, country_iso2 = get_country_bounds(country_iso3, nuts2_path)
    
    # Load hazard data for each return period
    print(f"Loading earthquake hazard data for {country_iso3}...")
    hazard_dict = {}
    
    data_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\Earthquakes")

    for return_period in tqdm(return_periods, desc="Loading hazard data"):
        try:
            hazard_file = data_path / f"PGA_1_{return_period}_vs30.tif"
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
        print("No hazard data could be loaded. Aborting earthquake risk assessment.")
        return gpd.GeoDataFrame()
    
    # Perform damage assessment for each return period
    print(f"Calculating damage for {len(hazard_dict)} return periods...")
    
    risk = {}  # Store damage results for each return period
    mean_damages = {}  # Store mean damage values by return period
    lengths = {}  # Store length values by return period
   
    # Package the common data that doesn't change per return period
    common_data = (features, fragility_curves, multi_curves, maxdam, asset_type)
    
    # Create a partial function with the common data already filled in
    process_fn = functools.partial(process_return_period_wrapper_earthquake, common_data)
    
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

    # Calculate risk
    RP_list = sorted(list(hazard_dict.keys()))
    df_risk = pd.concat(risk, axis=1)
    largest_rp = df_risk.loc[:, pd.IndexSlice[RP_list[-1], :]]
    
    # Create lookup for osm_id
    lookup_osm_id = dict(zip(features.index, features.osm_id))
    
    # Calculate risk and exposure for each curve
    collect_risks = {}
    collect_exposures = {}
    
    for curve in multi_curves.keys():
        # Get the risk dataframe for this curve
        subrisk = df_risk.loc[:, pd.IndexSlice[:, curve]].fillna(0)
        
        # Calculate risk and exposure for each asset
        risk_values = []
        exposure_values = []
        
        for idx, row in subrisk.iterrows():
            # For earthquakes, we don't have protection standards
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
    return format_risk_results_earthquake(
        largest_rp, 
        all_risks, 
        all_exposures, 
        multi_curves, 
        mean_damages,
        lengths
    )

# Example usage
if __name__ == "__main__":
    country_iso3 = sys.argv[1] if len(sys.argv) > 1 else 'ESP'  # Default to Cyprus
    asset_type = sys.argv[2] if len(sys.argv) > 2 else 'power'  # Default to power for testing
    
    # Use the correct path you provided
    fragility_path = Path(r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\12_repositories\AssetRisk_PanEU\EQ_fragility.xlsx")
    
    # # Fallback to your server path if the local path doesn't exist
    # if not fragility_path.exists():
    #     fragility_path = Path("/scistor/ivm/eks510/Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx")
    
    # Save results
    result_dir = Path(r"C:\MIRACA\risk")
    result_file = result_dir / f"{country_iso3}_{asset_type}_earthquake_risk.parquet"

    # Create directory if it doesn't exist
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)

    # Check if result file exists
    # if result_file.exists():
    #     print(f"Results already exist at {result_file}. Skipping risk assessment.")
    #     sys.exit(0)

    print(f"Running earthquake risk assessment for {country_iso3} {asset_type}...")
    print(f"Using fragility curves from: {fragility_path}")
    
    # Run earthquake risk assessment
    risk_results = assess_earthquake_risk(
        country_iso3=country_iso3,
        asset_type=asset_type,
        nuts2_path="NUTS_RG_20M_2024_3035.geojson",
        fragility_path=str(fragility_path),
        standardize_geom=True
    )
    
    # Save results to CSV
    risk_results.to_parquet(str(result_file))
    
    # Print summary statistics
    if len(risk_results) > 0:
        print("\nEarthquake risk summary statistics:")
        print(f"Mean risk (EAD): {risk_results['EAD'].mean():.2f}")
        print(f"Max risk (EAD): {risk_results['EAD'].max():.2f}")
        print(f"Total risk (EAD): {risk_results['EAD'].sum():.2f}")
        print(f"Mean exposure (EAE): {risk_results['EAE'].mean():.2f}")
        print(f"Total exposure (EAE): {risk_results['EAE'].sum():.2f}")
        print(f"Number of assets assessed: {len(risk_results)}")
    else:
        print("No risk results found for the given parameters.")