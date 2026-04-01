"""
hazard_river.py

River flood risk assessment module.

Takes a pre-loaded exposure GeoDataFrame and returns it enriched with:
  - EAD_river, EAD_river_min, EAD_river_max
  - EAD_river_1.5C, EAD_river_1.5C_min, EAD_river_1.5C_max  (+ 2.0C, 3.0C, 4.0C)
  - exposure_river_100  (length / area / count at RP100)

Depends on:
  - risk_integration.py  (EAD integration, exposure metrics, climate adjustment)
  - damagescanner        (VectorScanner via risk_integration.compute_damage_per_rp)
"""

import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import functools
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

from damagescanner.core import VectorExposure
from constants import (
    DICT_CIS_VULNERABILITY_FLOOD,
    INFRASTRUCTURE_DAMAGE_VALUES,
)

from risk_integration import (
    compute_damage_per_rp,
    collect_ead_per_asset,
    compute_exposure_metric,
    collect_ead_climate_scenarios,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Return periods and hazard band name
# ---------------------------------------------------------------------------

RIVER_RETURN_PERIODS = [10, 20, 30, 40, 50, 75, 100, 200, 500]
RIVER_HAZARD_COL = "band_data"
RIVER_EXPOSURE_RP = 100  # reference return period for exposure metric

# Temperature scenarios for future river — mapped to time periods + SSP
# 1.5°C → RCP4.5/2050 → SSP245, 2.0°C → RCP8.5/2050 → SSP585
# 3.0°C → RCP4.5/2100 → SSP245, 4.0°C → RCP8.5/2100 → SSP585
TEMP_CODES = ("15", "20", "30", "40")
TEMP_LABELS = ("2050_SSP245", "2050_SSP585", "2100_SSP245", "2100_SSP585")


def _worker_init():
    import sys

    sys.excepthook = lambda *args: None


# ---------------------------------------------------------------------------
# Vulnerability curve preparation
# ---------------------------------------------------------------------------


def prepare_flood_curves(
    asset_type: str,
    vulnerability_path: Union[str, Path],
    vulnerability_dict: Optional[dict] = None,
    damage_values: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare flood vulnerability curves for an asset type.

    Produces three sets of maxdam (min / mean / max) so that the caller
    can compute uncertainty bounds without running VectorScanner three times.
    The uncertainty from *curves* is captured via multi_curves (one entry
    per unique curve ID); uncertainty from *damage values* is captured via
    the three maxdam DataFrames.

    Args:
        asset_type:         Internal asset type name (e.g. 'power', 'roads')
        vulnerability_path: Path or URL to the Excel vulnerability file
        vulnerability_dict: Override for DICT_CIS_VULNERABILITY_FLOOD
                            (pass DICT_CIS_VULNERABILITY_WIND for windstorm)
        damage_values:      Override for INFRASTRUCTURE_DAMAGE_VALUES
                            (dict mapping object_type → [min, mean, max])

    Returns:
        Tuple of:
          - damage_curves   : DataFrame indexed by depth, columns = object types (mean curve)
          - multi_curves    : Dict {curve_id: DataFrame} for all unique curves
          - maxdam_mean     : DataFrame with mean maximum damage per object type
          - maxdam_min      : DataFrame with minimum maximum damage per object type
          - maxdam_max      : DataFrame with maximum maximum damage per object type
    """
    if vulnerability_dict is None:
        vulnerability_dict = DICT_CIS_VULNERABILITY_FLOOD

    if damage_values is None:
        damage_values = INFRASTRUCTURE_DAMAGE_VALUES

    # Read vulnerability Excel
    vul_df = pd.read_excel(vulnerability_path, sheet_name="F_Vuln_Depth").ffill()

    ci_system = vulnerability_dict[asset_type]

    # Build mean damage curves (one column per object type, using first listed curve)
    selected_curves = [curves[0] for curves in ci_system.values()]
    damage_curves = (
        vul_df[["ID number"] + selected_curves]
        .iloc[4:125]
        .set_index("ID number")
        .rename_axis("Depth")
        .astype(np.float32)
    )
    damage_curves.columns = list(ci_system.keys())
    damage_curves = damage_curves.ffill()

    # Build multi_curves: one entry per unique curve ID, all object columns set to that curve
    unique_curves = {c for curves in ci_system.values() for c in curves}
    multi_curves = {}
    for curve_id in unique_curves:
        curve_df = damage_curves.copy()
        curve_values = vul_df[curve_id].iloc[4:125].values
        for col in curve_df.columns:
            curve_df[col] = curve_values
        multi_curves[curve_id] = curve_df.astype(np.float32)

    # Build min / mean / max maxdam DataFrames
    asset_maxdam = damage_values.get(asset_type, {})

    def _make_maxdam(idx: int) -> pd.DataFrame:
        d = {k: v[idx] for k, v in asset_maxdam.items() if k in ci_system}
        df = pd.DataFrame.from_dict(d, orient="index").reset_index()
        df.columns = ["object_type", "damage"]
        return df

    maxdam_min = _make_maxdam(0)
    maxdam_mean = _make_maxdam(1)
    maxdam_max = _make_maxdam(2)

    return damage_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max


# ---------------------------------------------------------------------------
# Hazard data loading
# ---------------------------------------------------------------------------


def load_river_hazard(
    hazard_dir: Union[str, Path],
    return_periods: list[int],
    country_bounds: Optional[tuple] = None,
    filename_template: str = "Europe_RP{rp}_filled_depth.tif",
) -> dict[int, xr.Dataset]:
    """
    Load river flood hazard rasters for all return periods.

    Args:
        hazard_dir:         Directory containing the flood raster files
        return_periods:     List of return periods to load
        country_bounds:     Optional (minx, miny, maxx, maxy) in EPSG:4326 to clip
        filename_template:  Filename pattern with {rp} placeholder

    Returns:
        Dict mapping return_period → xarray Dataset (clipped to country if bounds given)
    """
    hazard_dir = Path(hazard_dir)
    hazard_dict = {}

    for rp in tqdm(return_periods, desc="Loading river hazard rasters"):
        path = hazard_dir / filename_template.format(rp=rp)
        if not path.exists():
            print(f"  Warning: hazard file not found for RP{rp}: {path}")
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


def get_country_bounds_4326(
    features: gpd.GeoDataFrame,
    buffer_deg: float = 0.1,
) -> tuple[float, float, float, float]:
    """
    Derive country bounding box in EPSG:4326 from the features GeoDataFrame.

    Args:
        features:   Exposure GeoDataFrame (any CRS)
        buffer_deg: Buffer in degrees to add around the bounds

    Returns:
        (minx, miny, maxx, maxy) in EPSG:4326
    """
    bounds = features.to_crs(4326).total_bounds  # [minx, miny, maxx, maxy]
    return (
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg,
    )


# ---------------------------------------------------------------------------
# Protection standards
# ---------------------------------------------------------------------------


def load_protection_standards(
    features: gpd.GeoDataFrame,
    protection_standard_path: Union[str, Path],
) -> pd.Series:
    """
    Extract flood protection standard (design return period) for each asset
    by overlaying with the protection standard raster.

    Args:
        features:                  Exposure GeoDataFrame (any CRS)
        protection_standard_path:  Path to protection standard GeoTIFF (EPSG:3035)

    Returns:
        Series mapping feature index → protection standard RP (0 if unprotected)
    """
    print("Loading flood protection standards...")

    prot_map = xr.open_dataset(protection_standard_path, engine="rasterio")

    # Coarsen to reduce memory and speed up overlay
    prot_map = prot_map.coarsen(x=10, y=10, boundary="trim").mean()
    prot_map.rio.write_crs("EPSG:3035", inplace=True)

    # Clip to country extent
    bounds = features.to_crs(3035).total_bounds
    prot_map = prot_map.rio.clip_box(
        minx=bounds[0],
        miny=bounds[1],
        maxx=bounds[2],
        maxy=bounds[3],
    )

    # Use centroid points for overlay (protection standard is a coarse raster)
    features_prot = features.to_crs(3035).copy()
    features_prot["geometry"] = features_prot.centroid

    exposed, _, _, _ = VectorExposure(
        hazard_file=prot_map,
        feature_file=features_prot,
        hazard_value_col="band_data",
        disable_progress=True,
    )

    # Take the maximum value overlapping each asset as its protection standard
    design_standards = exposed["values"].apply(
        lambda v: float(np.max(v)) if hasattr(v, "__len__") and len(v) > 0 else 0.0
    )
    design_standards.index = features.index
    design_standards = design_standards.fillna(0).clip(lower=0)

    print(
        f"  Protection standards loaded. Mean: {design_standards.mean():.0f} yr, "
        f"Max: {design_standards.max():.0f} yr"
    )
    return design_standards


# ---------------------------------------------------------------------------
# Curve filtering (inappropriate curve exclusions)
# ---------------------------------------------------------------------------


def filter_curve_results(
    result_df: gpd.GeoDataFrame,
    multi_curves: dict,
    object_curve_exclusions: dict,
) -> gpd.GeoDataFrame:
    """
    Set damage values to NaN for curves that are inappropriate for
    specific object types (e.g. don't apply underground cable curves to towers).

    Args:
        result_df:               VectorScanner output with one column per curve
        multi_curves:            Dict of curve_id → DataFrame
        object_curve_exclusions: {object_type: [curve_ids_to_exclude]}

    Returns:
        Filtered DataFrame
    """
    if not object_curve_exclusions:
        return result_df

    result_df = result_df.copy()
    curve_cols = [c for c in multi_curves.keys() if c in result_df.columns]

    for obj_type, excluded in object_curve_exclusions.items():
        mask = result_df["object_type"] == obj_type
        if not mask.any():
            continue
        for curve in excluded:
            if curve in curve_cols:
                result_df.loc[mask, curve] = np.nan

    return result_df


# ---------------------------------------------------------------------------
# Per-RP damage worker (used in parallel)
# ---------------------------------------------------------------------------


def _compute_rp_damage(args, common):
    """
    Worker function: compute damage for a single return period.

    Args:
        args:   (return_period, hazard_dataset)
        common: (features, damage_curves, multi_curves, maxdam, asset_type,
                 object_curve_exclusions)

    Returns:
        (return_period, result_df_with_mean_min_max_columns)
    """
    rp, hazard = args
    features, damage_curves, multi_curves, maxdam, asset_type, exclusions = common

    # Run VectorScanner — get one damage column per curve
    result = compute_damage_per_rp(
        features=features,
        hazard=hazard,
        curve_path=damage_curves,
        maxdam=maxdam,
        asset_type=asset_type,
        multi_curves=multi_curves,
        hazard_value_col=RIVER_HAZARD_COL,
    )

    # Filter inappropriate curves
    result = filter_curve_results(result, multi_curves, exclusions or {})

    # Summarise across curves → mean / min / max damage per asset
    curve_cols = [c for c in multi_curves.keys() if c in result.columns]
    result["damage_mean"] = result[curve_cols].mean(axis=1, skipna=True)
    result["damage_min"] = result[curve_cols].min(axis=1, skipna=True)
    result["damage_max"] = result[curve_cols].max(axis=1, skipna=True)

    return rp, result


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def assign_basin_ids(
    features: gpd.GeoDataFrame,
    basin_data: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Spatially assign each feature to a HydroBASIN polygon via centroid join.

    Uses a spatial index for efficiency — same pattern as protection standards.

    Args:
        features:   Exposure GeoDataFrame (any CRS — reprojected internally)
        basin_data: GeoDataFrame with basin polygons indexed by HYBAS_ID

    Returns:
        Series mapping feature index → HYBAS_ID (NaN where no basin found)
    """
    # Reproject features centroids to match basin CRS
    basin_crs = "EPSG:3035"
    centroids = features.geometry.to_crs(basin_crs).centroid
    centroids_gdf = gpd.GeoDataFrame(
        geometry=centroids, index=features.index, crs=basin_crs
    )

    basin_reset = basin_data[["geometry"]].reset_index()
    # The former index is now the first non-geometry column
    id_col = [c for c in basin_reset.columns if c != "geometry"][0]

    joined = gpd.sjoin(
        centroids_gdf,
        basin_reset,
        how="left",
        predicate="within",
    )

    joined = joined[~joined.index.duplicated(keep="first")]
    result = joined[id_col].reindex(features.index)

    return result


def assess_river(
    features: gpd.GeoDataFrame,
    hazard_dir: Union[str, Path],
    vulnerability_path: Union[str, Path],
    asset_type: str,
    protection_standard_path: Optional[Union[str, Path]] = None,
    basin_data: Optional[gpd.GeoDataFrame] = None,
    object_curve_exclusions: Optional[dict] = None,
    return_periods: list[int] = RIVER_RETURN_PERIODS,
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Assess river flood risk for a pre-loaded exposure GeoDataFrame.

    Enriches the input GeoDataFrame with risk columns and returns it.
    Climate scenario columns are only added if basin_data is provided.

    Args:
        features:                  Exposure GeoDataFrame (EPSG:3035)
        hazard_dir:                Directory with river flood rasters
        vulnerability_path:        Path to vulnerability Excel file
        asset_type:                Internal asset type (e.g. 'power', 'roads')
        protection_standard_path:  Optional path to protection standard raster
        basin_data:                Optional GeoDataFrame with basin polygons and
                                   climate RP shift columns (index = HYBAS_ID).
                                   Feature centroids are spatially joined to basins.
        object_curve_exclusions:   {object_type: [curve_ids_to_exclude]}
        return_periods:            Return periods to assess
        n_workers:                 Number of parallel workers (None = all CPUs)

    Returns:
        Input GeoDataFrame enriched with:
          EAD_river, EAD_river_min, EAD_river_max
          EAD_river_1.5C, EAD_river_1.5C_min, EAD_river_1.5C_max (if basin_data given)
          ... (2.0C, 3.0C, 4.0C)
          exposure_river_100
    """
    t0 = time.time()
    print(
        f"[river] Starting assessment for {asset_type} "
        f"({len(features)} features, {len(return_periods)} return periods)"
    )

    # --- 1. Vulnerability curves ---
    damage_curves, multi_curves, maxdam_mean, _, _ = prepare_flood_curves(
        asset_type, vulnerability_path
    )

    # --- 2. Hazard data ---
    country_bounds = get_country_bounds_4326(features)
    hazard_dict = load_river_hazard(hazard_dir, return_periods, country_bounds)

    if not hazard_dict:
        print("[river] No hazard data found. Returning features unchanged.")
        return features

    available_rps = sorted(hazard_dict.keys())

    # --- 3. Protection standards ---
    protection_standards = None
    if protection_standard_path is not None:
        protection_standards = load_protection_standards(
            features, protection_standard_path
        )

    # --- 4. Parallel damage calculation per return period ---
    common = (
        features,
        damage_curves,
        multi_curves,
        maxdam_mean,
        asset_type,
        object_curve_exclusions,
    )
    work_items = [(rp, hazard_dict[rp]) for rp in available_rps]
    worker_fn = functools.partial(_compute_rp_damage, common=common)

    print(
        f"[river] Running damage calculation across {len(work_items)} return periods..."
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init
    ) as executor:
        raw_results = list(
            tqdm(
                executor.map(worker_fn, work_items),
                total=len(work_items),
                desc="River flood RPs",
            )
        )

    # rp_results: {rp: GeoDataFrame with damage_mean/min/max columns}
    rp_results = {rp: df for rp, df in raw_results}

    # --- 5. Integrate EAD (base / current climate) ---
    print("[river] Integrating EAD...")
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features,
        protection_standards=protection_standards,
    )
    features = features.copy()
    features["EAD_mid_river_current"] = ead_df["EAD_mid"].values
    features["EAD_min_river_current"] = ead_df["EAD_min"].values
    features["EAD_max_river_current"] = ead_df["EAD_max"].values

    # --- 6. Exposure metric at RP100 ---
    if RIVER_EXPOSURE_RP in hazard_dict:
        print(f"[river] Computing exposure metric at RP{RIVER_EXPOSURE_RP}...")
        features["exposure_abs_river_current"] = compute_exposure_metric(
            features=features,
            hazard=hazard_dict[RIVER_EXPOSURE_RP],
            reference_rp=RIVER_EXPOSURE_RP,
            hazard_value_col=RIVER_HAZARD_COL,
            pga_threshold=0.0,
        ).values
    else:
        print(f"[river] RP{RIVER_EXPOSURE_RP} not available, skipping exposure metric.")
        features["exposure_abs_river_current"] = np.nan

    # --- 7. Future climate scenarios ---
    if basin_data is not None:
        print("[river] Computing future climate scenario EADs...")
        basin_ids = assign_basin_ids(features, basin_data)
        climate_df = collect_ead_climate_scenarios(
            rp_results=rp_results,
            features=features,
            basin_data=basin_data,
            basin_ids=basin_ids,
            protection_standards=protection_standards,
            temp_scenarios=TEMP_CODES,
            temp_labels=TEMP_LABELS,
        )
        for col in climate_df.columns:
            features[col] = climate_df[col].values

        # Future exposure: compute exposure at RP100 for each future period
        # (uses same hazard data — future exposure assumes same hazard intensity)
        for period_label in TEMP_LABELS:
            features[f"exposure_abs_river_{period_label}"] = features[
                "exposure_abs_river_current"
            ].copy()
    else:
        print("[river] No basin data provided, skipping future climate scenarios.")

    elapsed = time.time() - t0
    print(
        f"[river] Done in {elapsed:.1f}s. "
        f"Mean EAD_mid_river_current: {features['EAD_mid_river_current'].mean():.2f}, "
        f"Total: {features['EAD_mid_river_current'].sum():.2e}"
    )

    return features
