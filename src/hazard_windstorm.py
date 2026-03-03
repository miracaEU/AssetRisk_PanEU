"""
hazard_windstorm.py

Windstorm risk assessment module.

Takes a pre-loaded exposure GeoDataFrame and returns it enriched with:
  - EAD_windstorm, EAD_windstorm_min, EAD_windstorm_max
  - exposure_wind_100  (length / area / count at RP100)

Hazard data: local GeoTIFFs, one per return period.
Vulnerability: wind speed curves (W_Vuln_V10m_3sec sheet).
No protection standards for windstorm.
"""

import time
import warnings
import functools
import concurrent.futures
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

from constants import (
    DICT_CIS_VULNERABILITY_WIND,
    INFRASTRUCTURE_DAMAGE_VALUES,
)

from risk_integration import (
    compute_damage_per_rp,
    collect_ead_per_asset,
    compute_exposure_metric,
)
from hazard_river import filter_curve_results

def _worker_init():
    import sys
    sys.excepthook = lambda *args: None

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIND_RETURN_PERIODS    = [5, 10, 25, 50, 100, 250, 500]
WIND_HAZARD_COL        = "band_data"
WIND_EXPOSURE_RP       = 100
WIND_FILENAME_TEMPLATE = "{rp}yr_wisc_nao_0.59.tif"

# Object types to keep for power (wind only damages above-ground elements)
WIND_POWER_OBJECT_TYPES = {"line", "tower", "catenary_mast", "pole", "minor_line"}

def prepare_wind_curves(
    asset_type: str,
    vulnerability_path: Union[str, Path],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare wind vulnerability curves for an asset type.

    Args:
        asset_type:         Internal asset type name
        vulnerability_path: Path to vulnerability Excel file

    Returns:
        (damage_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max)
    """
    vul_df = pd.read_excel(
        vulnerability_path,
        sheet_name="W_Vuln_V10m_3sec"
    ).ffill()

    ci_system = DICT_CIS_VULNERABILITY_WIND.get(asset_type, {})
    if not ci_system:
        raise ValueError(
            f"No wind vulnerability curves defined for asset type '{asset_type}'. "
            f"Available: {list(DICT_CIS_VULNERABILITY_WIND.keys())}"
        )

    # Mean damage curves (first listed curve per object type)
    selected_curves = [curves[0] for curves in ci_system.values()]
    damage_curves = (
        vul_df[["ID number"] + selected_curves]
        .iloc[4:125]
        .set_index("ID number")
        .rename_axis("Speed")
        .astype(np.float32)
    )
    damage_curves.columns = list(ci_system.keys())
    damage_curves = damage_curves.ffill()

    # multi_curves: one entry per unique curve ID
    unique_curves = {c for curves in ci_system.values() for c in curves}
    multi_curves = {}
    for curve_id in unique_curves:
        curve_df = damage_curves.copy()
        curve_values = vul_df[curve_id].iloc[4:125].values
        for col in curve_df.columns:
            curve_df[col] = curve_values
        multi_curves[curve_id] = curve_df.astype(np.float32)

    # Min / mean / max maxdam
    asset_maxdam = INFRASTRUCTURE_DAMAGE_VALUES.get(asset_type, {})

    def _make_maxdam(idx: int) -> pd.DataFrame:
        d = {k: v[idx] for k, v in asset_maxdam.items() if k in ci_system}
        df = pd.DataFrame.from_dict(d, orient="index").reset_index()
        df.columns = ["object_type", "damage"]
        return df

    return damage_curves, multi_curves, _make_maxdam(1), _make_maxdam(0), _make_maxdam(2)


# ---------------------------------------------------------------------------
# Hazard data loading
# ---------------------------------------------------------------------------

def load_windstorm_hazard(
    hazard_dir: Union[str, Path],
    return_periods: list[int],
    country_bounds: Optional[tuple] = None,
    filename_template: str = WIND_FILENAME_TEMPLATE,
) -> dict[int, xr.Dataset]:
    """
    Load windstorm hazard rasters for all return periods.

    Args:
        hazard_dir:         Directory containing windstorm rasters
        return_periods:     List of return periods to load
        country_bounds:     Optional (minx, miny, maxx, maxy) in EPSG:4326 to clip
        filename_template:  Filename pattern with {rp} placeholder

    Returns:
        Dict mapping return_period → xarray Dataset
    """
    hazard_dir = Path(hazard_dir)
    hazard_dict = {}

    for rp in tqdm(return_periods, desc="Loading windstorm hazard rasters"):
        path = hazard_dir / filename_template.format(rp=rp)
        if not path.exists():
            print(f"  Warning: windstorm file not found for RP{rp}: {path}")
            continue
        try:
            ds = xr.open_dataset(path, engine="rasterio")
            if country_bounds is not None:
                minx, miny, maxx, maxy = country_bounds
                ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            hazard_dict[rp] = ds
        except Exception as e:
            print(f"  Warning: could not load RP{rp}: {e}")

    return hazard_dict


# ---------------------------------------------------------------------------
# Per-RP damage worker (used in parallel)
# ---------------------------------------------------------------------------

def _compute_wind_rp_damage(args, common):
    """Worker: compute wind damage for a single return period."""
    rp, hazard = args
    features, damage_curves, multi_curves, maxdam, asset_type, exclusions = common

    result = compute_damage_per_rp(
        features=features,
        hazard=hazard,
        curve_path=damage_curves,
        maxdam=maxdam,
        asset_type=asset_type,
        multi_curves=multi_curves,
        hazard_value_col=WIND_HAZARD_COL,
    )

    result = filter_curve_results(result, multi_curves, exclusions or {})

    curve_cols = [c for c in multi_curves.keys() if c in result.columns]
    result["damage_mean"] = result[curve_cols].mean(axis=1, skipna=True)
    result["damage_min"]  = result[curve_cols].min(axis=1, skipna=True)
    result["damage_max"]  = result[curve_cols].max(axis=1, skipna=True)

    return rp, result


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------

def assess_windstorm(
    features: gpd.GeoDataFrame,
    hazard_dir: Union[str, Path],
    vulnerability_path: Union[str, Path],
    asset_type: str,
    object_curve_exclusions: Optional[dict] = None,
    return_periods: list[int] = WIND_RETURN_PERIODS,
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Assess windstorm risk for a pre-loaded exposure GeoDataFrame.

    Args:
        features:                Exposure GeoDataFrame (EPSG:3035)
        hazard_dir:              Directory with windstorm rasters
        vulnerability_path:      Path to vulnerability Excel file
        asset_type:              Internal asset type (e.g. 'power', 'roads')
        object_curve_exclusions: {object_type: [curve_ids_to_exclude]}
        return_periods:          Return periods to assess
        n_workers:               Number of parallel workers (None = all CPUs)

    Returns:
        Input GeoDataFrame enriched with:
          EAD_windstorm, EAD_windstorm_min, EAD_windstorm_max
          exposure_wind_100
    """
    t0 = time.time()

    # For power assets, windstorm only affects above-ground linear/point elements
    features_wind = features.copy()
    if asset_type == "power":
        features_wind = features_wind[
            features_wind["object_type"].isin(WIND_POWER_OBJECT_TYPES)
        ]
        print(f"[windstorm] Power asset filter: {len(features_wind)}/{len(features)} features retained")

    print(f"[windstorm] Starting assessment for {asset_type} "
          f"({len(features_wind)} features, {len(return_periods)} return periods)")

    # --- 1. Vulnerability curves ---
    try:
        damage_curves, multi_curves, maxdam_mean, _, _ = prepare_wind_curves(
            asset_type, vulnerability_path
        )
    except ValueError as e:
        print(f"[windstorm] {e} — skipping windstorm for this asset type.")
        features["EAD_windstorm"]     = np.nan
        features["EAD_windstorm_min"] = np.nan
        features["EAD_windstorm_max"] = np.nan
        features["exposure_wind_100"] = np.nan
        return features

    # --- 2. Hazard data ---
    from hazard_river import get_country_bounds_4326
    country_bounds = get_country_bounds_4326(features_wind)
    hazard_dict = load_windstorm_hazard(hazard_dir, return_periods, country_bounds)

    if not hazard_dict:
        print("[windstorm] No hazard data found. Returning features unchanged.")
        features["EAD_windstorm"]     = np.nan
        features["EAD_windstorm_min"] = np.nan
        features["EAD_windstorm_max"] = np.nan
        features["exposure_wind_100"] = np.nan
        return features

    available_rps = sorted(hazard_dict.keys())

    # --- 3. Parallel damage calculation per return period ---
    common = (features_wind, damage_curves, multi_curves, maxdam_mean,
              asset_type, object_curve_exclusions)
    work_items = [(rp, hazard_dict[rp]) for rp in available_rps]
    worker_fn = functools.partial(_compute_wind_rp_damage, common=common)

    print(f"[windstorm] Running damage calculation across {len(work_items)} return periods...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers,
                                                initializer=_worker_init) as executor:
        raw_results = list(tqdm(
            executor.map(worker_fn, work_items),
            total=len(work_items),
            desc="Windstorm RPs",
        ))

    rp_results = {rp: df for rp, df in raw_results}

    # --- 4. Integrate EAD (no protection standards for wind) ---
    print("[windstorm] Integrating EAD...")
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features_wind,
        protection_standards=None,
    )

    # Write results back onto the full features GeoDataFrame
    features = features.copy()
    features["EAD_windstorm"]     = 0.0
    features["EAD_windstorm_min"] = 0.0
    features["EAD_windstorm_max"] = 0.0

    features.loc[features_wind.index, "EAD_windstorm"]     = ead_df["EAD"].values
    features.loc[features_wind.index, "EAD_windstorm_min"] = ead_df["EAD_min"].values
    features.loc[features_wind.index, "EAD_windstorm_max"] = ead_df["EAD_max"].values

    # --- 5. Exposure metric at RP100 ---
    if WIND_EXPOSURE_RP in hazard_dict:
        print(f"[windstorm] Computing exposure metric at RP{WIND_EXPOSURE_RP}...")
        exposure = compute_exposure_metric(
            features=features_wind,
            hazard=hazard_dict[WIND_EXPOSURE_RP],
            reference_rp=WIND_EXPOSURE_RP,
            hazard_value_col=WIND_HAZARD_COL,
            pga_threshold=0.0,
        )
        features["exposure_wind_100"] = 0.0
        features.loc[features_wind.index, "exposure_wind_100"] = exposure.values
    else:
        print(f"[windstorm] RP{WIND_EXPOSURE_RP} not available, skipping exposure metric.")
        features["exposure_wind_100"] = np.nan

    elapsed = time.time() - t0
    print(f"[windstorm] Done in {elapsed:.1f}s. "
          f"Mean EAD_windstorm: {features['EAD_windstorm'].mean():.2f}, "
          f"Total: {features['EAD_windstorm'].sum():.2e}")

    return features
