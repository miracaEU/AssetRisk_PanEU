"""
hazard_wrappers_tent.py

TENT-specific hazard assessment functions.

Uses compute_damage_per_rp_tent (return_full=True) so that per-RP
exposed length is computed from actual VectorScanner coverage data
(sum of cell-level intersected lengths) rather than full geometry length.

Per-RP columns added per hazard:
  - vuln_ratio_{hazard}_RP{rp}       : mean damage ratio (0-1)
  - exposure_length_{hazard}_RP{rp}  : actual intersected length in metres
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

from risk_integration import (
    collect_ead_per_asset,
    collect_ead_climate_scenarios,
)
from risk_integration_tent import (
    compute_damage_per_rp_tent,
    compute_exposure_metric_tent,
    attach_per_rp_columns,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Port-specific curve loading (curves_flooding/TC/earthquake.xlsx)
# ---------------------------------------------------------------------------

_PORT_CURVE_FILES = {
    "flood":       "curves_flooding.xlsx",
    "windstorm":   "curves_TC.xlsx",       # TC = tropical cyclone / windstorm
    "earthquake":  "curves_earthquake.xlsx",
}

_PORT_CURVE_COLS = [
    "General Cargo", "Container", "RoRo", "Liquid", "Dry Bulk",
    "Raw", "Refinery", "Industry", "Warehouse", "Crane", "Storage", "Other",
]


def _prepare_port_curves(
    hazard_type: str,
    curves_dir: Union[str, Path],
) -> tuple:
    """
    Load port-specific damage curves from the new xlsx files.

    Returns (damage_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max)
    matching the format expected by compute_damage_per_rp_tent.

    Args:
        hazard_type:  'flood', 'windstorm', or 'earthquake'
        curves_dir:   Directory containing curves_flooding/TC/earthquake.xlsx
    """
    from constants import PORT_TENT_MAXDAM

    fname = _PORT_CURVE_FILES[hazard_type]
    df = pd.read_excel(Path(curves_dir) / fname)

    # First column = intensity (depth / max_wind / PGA), rest = land_use curves
    intensity_col = df.columns[0]
    df = df.set_index(intensity_col)
    df.index.name = intensity_col

    # Keep only port land_use columns, clip ratios to [0, 1]
    port_cols = [c for c in _PORT_CURVE_COLS if c in df.columns]
    damage_curves = df[port_cols].astype(np.float32).clip(0, 1)

    # Single curve variant (no uncertainty from curves) — min = max = mean
    multi_curves = {"port_curves": damage_curves}

    def _maxdam_df(idx):  # idx: 0=min, 1=mean, 2=max
        rows = [
            {"object_type": lu, "damage": float(PORT_TENT_MAXDAM.get(lu, [60, 120, 200])[idx])}
            for lu in port_cols
        ]
        return pd.DataFrame(rows)

    return damage_curves, multi_curves, _maxdam_df(1), _maxdam_df(0), _maxdam_df(2)


# ---------------------------------------------------------------------------
# TENT-specific per-RP worker functions
# ---------------------------------------------------------------------------


def _worker_init():
    import sys
    sys.excepthook = lambda *args: None


def _compute_eq_ports_rp_damage_tent(args, common):
    """
    Port earthquake worker using VectorScanner (same as flood/wind).
    Port EQ curves are direct damage ratios (PGA vs ratio), not fragility.
    """
    from hazard_earthquake import EQ_HAZARD_COL
    from hazard_river import filter_curve_results

    rp, hazard = args
    features, damage_curves, multi_curves, maxdam, asset_type, _ = common

    result = compute_damage_per_rp_tent(
        features=features,
        hazard=hazard,
        curve_path=damage_curves,
        maxdam=maxdam,
        asset_type=asset_type,
        multi_curves=multi_curves,
        hazard_value_col=EQ_HAZARD_COL,
    )

    result = filter_curve_results(result, multi_curves, {})

    curve_cols = [c for c in multi_curves.keys() if c in result.columns]
    result["damage_mean"] = result[curve_cols].mean(axis=1, skipna=True)
    result["damage_min"]  = result[curve_cols].min(axis=1, skipna=True)
    result["damage_max"]  = result[curve_cols].max(axis=1, skipna=True)

    if "area" in result.columns:
        first_geom = result.geometry.iloc[0] if len(result) > 0 else None
        if first_geom is not None and first_geom.geom_type in ("Point", "MultiPoint"):
            area_vals = pd.to_numeric(result["area"], errors="coerce").fillna(0)
            for col in ["damage_mean", "damage_min", "damage_max"]:
                result[col] = result[col] * area_vals

    return rp, result


def _compute_rp_damage_tent(args, common):
    """Worker: compute damage for a single RP using TENT version (retains coverage)."""
    from hazard_river import filter_curve_results

    rp, hazard = args
    features, damage_curves, multi_curves, maxdam, asset_type, exclusions = common

    result = compute_damage_per_rp_tent(
        features=features,
        hazard=hazard,
        curve_path=damage_curves,
        maxdam=maxdam,
        asset_type=asset_type,
        multi_curves=multi_curves,
        hazard_value_col="band_data",
    )

    result = filter_curve_results(result, multi_curves, exclusions or {})

    curve_cols = [c for c in multi_curves.keys() if c in result.columns]
    result["damage_mean"] = result[curve_cols].mean(axis=1, skipna=True)
    result["damage_min"] = result[curve_cols].min(axis=1, skipna=True)
    result["damage_max"] = result[curve_cols].max(axis=1, skipna=True)

    # For point ports only: scale damage by area attribute (VectorScanner gives per-unit damage)
    # Polygon ports already have area incorporated via VectorScanner coverage — do not scale
    # Check per-feature, not just first, because dataset may mix Point and Polygon geometries
    if asset_type in ("ports", "iww") and "area" in result.columns:
        is_point = result.geometry.geom_type.isin(["Point", "MultiPoint"])
        if is_point.any():
            area_vals = pd.to_numeric(result["area"], errors="coerce").fillna(0)
            for col in ["damage_mean", "damage_min", "damage_max"]:
                result.loc[is_point, col] = result.loc[is_point, col] * area_vals[is_point]

    return rp, result


def _compute_wind_rp_damage_tent(args, common):
    """Worker: compute wind damage for a single RP using TENT version."""
    from hazard_river import filter_curve_results
    from hazard_windstorm import WIND_HAZARD_COL

    rp, hazard = args
    features, damage_curves, multi_curves, maxdam, asset_type, exclusions = common

    result = compute_damage_per_rp_tent(
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
    result["damage_min"] = result[curve_cols].min(axis=1, skipna=True)
    result["damage_max"] = result[curve_cols].max(axis=1, skipna=True)

    if asset_type in ("ports", "iww") and "area" in result.columns:
        is_point = result.geometry.geom_type.isin(["Point", "MultiPoint"])
        if is_point.any():
            area_vals = pd.to_numeric(result["area"], errors="coerce").fillna(0)
            for col in ["damage_mean", "damage_min", "damage_max"]:
                result.loc[is_point, col] = result.loc[is_point, col] * area_vals[is_point]

    return rp, result


def _compute_eq_rp_damage_tent(args, common):
    """
    Worker: compute earthquake damage for a single RP using TENT version.

    Reuses the vectorised EDR logic from hazard_earthquake._compute_eq_rp_damage
    but also computes exposed_length from VectorExposure coverage data,
    consistent with the other TENT hazard wrappers.
    """
    from hazard_earthquake import (
        EQ_HAZARD_COL,
        _build_edr_lookup,
        load_earthquake_hazard,
        DAMAGE_RATIOS,
    )
    from damagescanner.vector import VectorExposure
    from constants import DICT_CIS_VULNERABILITY_EARTHQUAKE

    rp, hazard = args
    features, fragility_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max, asset_type = common

    # --- 1. Extract PGA values via VectorExposure (retains coverage + values) ---
    exposed, _, crs, cell_area = VectorExposure(
        hazard_file=hazard,
        feature_file=features,
        hazard_value_col=EQ_HAZARD_COL,
        disable_progress=True,
    )

    if exposed is None or exposed.empty:
        return rp, pd.DataFrame()

    if cell_area is None:
        try:
            cell_area = float(
                abs(hazard.x[1].values - hazard.x[0].values)
                * abs(hazard.y[0].values - hazard.y[1].values)
            )
        except Exception:
            cell_area = 1.0

    # Reproject to EPSG:3035 so geometry lengths are in metres
    exposed = exposed.to_crs(3035)

    # --- 2. Maxdam lookups ---
    maxdam_lookup_mean = dict(zip(maxdam_mean["object_type"], maxdam_mean["damage"]))
    maxdam_lookup_min  = dict(zip(maxdam_min["object_type"],  maxdam_min["damage"]))
    maxdam_lookup_max  = dict(zip(maxdam_max["object_type"],  maxdam_max["damage"]))

    ci_system = DICT_CIS_VULNERABILITY_EARTHQUAKE.get(asset_type, {})
    pga_index = fragility_curves.index.to_numpy(dtype=float)

    # --- 3. Pre-compute EDR lookup tables once per curve ---
    edr_tables = {cid: _build_edr_lookup(fragility_curves, cid) for cid in multi_curves}

    # --- 4. Vectorised damage per object_type group ---
    damage_mean = np.zeros(len(exposed))
    damage_min  = np.zeros(len(exposed))
    damage_max  = np.zeros(len(exposed))

    for obj_type, group_idx in exposed.groupby("object_type").groups.items():
        curve_ids = ci_system.get(obj_type, [])
        if not curve_ids:
            continue

        group = exposed.loc[group_idx]
        pos = [exposed.index.get_loc(i) for i in group_idx]

        md_mean = maxdam_lookup_mean.get(obj_type, 0.0)
        md_min  = maxdam_lookup_min.get(obj_type, 0.0)
        md_max  = maxdam_lookup_max.get(obj_type, 0.0)

        values_list   = [np.asarray(v if v is not None else [0], dtype=float) for v in group["values"].tolist()]
        coverage_list = [np.asarray(c if c is not None else [0], dtype=float) for c in group["coverage"].tolist()]

        n_assets = len(group)
        max_len  = max(len(v) for v in values_list) if values_list else 1

        pga_mat = np.zeros((n_assets, max_len))
        cov_mat = np.zeros((n_assets, max_len))
        mask    = np.zeros((n_assets, max_len), dtype=bool)

        for k, (pga_vals, cov_vals) in enumerate(zip(values_list, coverage_list)):
            n = len(pga_vals)
            pga_mat[k, :n] = pga_vals
            cov_mat[k, :n] = cov_vals
            mask[k, :n]    = True

        geom_types   = group.geometry.geom_type.to_numpy()
        is_poly_mask  = np.isin(geom_types, ["Polygon", "MultiPolygon"])
        is_point_mask = ~np.isin(geom_types, ["Polygon", "MultiPolygon", "LineString", "MultiLineString"])
        cov_mat_scaled = cov_mat.copy()
        cov_mat_scaled[is_poly_mask]  *= cell_area
        cov_mat_scaled[is_point_mask]  = 1.0

        curve_damages = []
        for cid in curve_ids:
            if cid not in edr_tables:
                continue
            edr_flat = np.interp(pga_mat.ravel(), pga_index, edr_tables[cid])
            edr_mat  = edr_flat.reshape(pga_mat.shape)
            asset_damages = np.sum(edr_mat * cov_mat_scaled * mask, axis=1) * md_mean
            curve_damages.append(asset_damages)

        if not curve_damages:
            continue

        curve_mat  = np.vstack(curve_damages)
        scale_min  = md_min / md_mean if md_mean > 0 else 0.0
        scale_max  = md_max / md_mean if md_mean > 0 else 0.0

        for k, p in enumerate(pos):
            damage_mean[p] = float(np.mean(curve_mat[:, k]))
            damage_min[p]  = float(np.min(curve_mat[:, k]))  * scale_min
            damage_max[p]  = float(np.max(curve_mat[:, k]))  * scale_max

    exposed["damage_mean"] = damage_mean
    exposed["damage_min"]  = damage_min
    exposed["damage_max"]  = damage_max

    # --- 5. Exposed length/count from coverage ---
    def _exposed_length(row):
        vals = row.get("values", [])
        covs = row.get("coverage", [])
        vals = np.asarray(vals if vals is not None else [], dtype=float)
        covs = np.asarray(covs if covs is not None else [], dtype=float)
        if len(vals) == 0 or len(covs) == 0:
            return 0.0
        geom = row.geometry if hasattr(row, "geometry") else None
        if geom is not None and geom.geom_type in ("Point", "MultiPoint"):
            return 1.0 if bool(np.any(vals > 0)) else 0.0
        exposed_len = float(np.sum(covs[vals > 0]))
        geom_len = geom.length if geom is not None else np.inf
        return min(exposed_len, geom_len)

    exposed["exposed_length"] = exposed.apply(_exposed_length, axis=1)
    exposed = exposed.drop(columns=["values", "coverage"], errors="ignore")

    return rp, exposed


# ---------------------------------------------------------------------------
# River flood -- single-pass TENT version
# ---------------------------------------------------------------------------


def assess_river_tent(
    features: gpd.GeoDataFrame,
    hazard_dir: Union[str, Path],
    vulnerability_path: Union[str, Path],
    asset_type: str,
    protection_standard_path: Optional[Union[str, Path]] = None,
    basin_data: Optional[gpd.GeoDataFrame] = None,
    object_curve_exclusions: Optional[dict] = None,
    return_periods: Optional[list[int]] = None,
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    River flood assessment with per-RP ratio and exposure length columns.
    Uses actual VectorScanner coverage for exposed length.
    """
    from hazard_river import (
        prepare_flood_curves,
        load_river_hazard,
        get_country_bounds_4326,
        load_protection_standards,
        assign_basin_ids,
        RIVER_RETURN_PERIODS,
        RIVER_HAZARD_COL,
        RIVER_EXPOSURE_RP,
        TEMP_CODES,
        TEMP_LABELS,
    )

    if return_periods is None:
        return_periods = RIVER_RETURN_PERIODS

    t0 = time.time()
    print(
        f"[river-TENT] Starting assessment for {asset_type} "
        f"({len(features)} features, {len(return_periods)} return periods)"
    )

    # --- 1. Vulnerability curves ---
    try:
        if asset_type in ("ports", "iww"):
            damage_curves, multi_curves, maxdam_mean, _, _ = _prepare_port_curves(
                "flood", Path(vulnerability_path).parent
            )
        else:
            damage_curves, multi_curves, maxdam_mean, _, _ = prepare_flood_curves(
                asset_type, vulnerability_path
            )
    except (ValueError, KeyError) as e:
        print(f"[river-TENT] {e} -- skipping.")
        for col in ["EAD_mid_river_current", "EAD_min_river_current",
                     "EAD_max_river_current", "exposure_abs_river_current"]:
            features[col] = np.nan
        return features

    # --- 2. Hazard data ---
    country_bounds = get_country_bounds_4326(features)
    hazard_dict = load_river_hazard(hazard_dir, return_periods, country_bounds)

    if not hazard_dict:
        print("[river-TENT] No hazard data found.")
        for col in ["EAD_mid_river_current", "EAD_min_river_current",
                     "EAD_max_river_current", "exposure_abs_river_current"]:
            features[col] = np.nan
        return features

    available_rps = sorted(hazard_dict.keys())

    # --- 3. Protection standards ---
    protection_standards = None
    if protection_standard_path is not None:
        protection_standards = load_protection_standards(
            features, protection_standard_path
        )

    print(features.crs)

    # --- 4. Parallel damage calculation per RP (TENT version) ---
    common = (
        features, damage_curves, multi_curves, maxdam_mean,
        asset_type, object_curve_exclusions,
    )
    work_items = [(rp, hazard_dict[rp]) for rp in available_rps]
    worker_fn = functools.partial(_compute_rp_damage_tent, common=common)

    print(f"[river-TENT] Running damage calculation across {len(work_items)} RPs...")
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

    rp_results = {rp: df for rp, df in raw_results}

    # --- 5. Integrate EAD (standard) ---
    print("[river-TENT] Integrating EAD...")
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features,
        protection_standards=protection_standards,
    )
    features = features.copy()
    features["EAD_mid_river_current"] = ead_df["EAD_mid"].values
    features["EAD_min_river_current"] = ead_df["EAD_min"].values
    features["EAD_max_river_current"] = ead_df["EAD_max"].values
    features["protection_standard_river"] = (
        features.index.map(protection_standards) if protection_standards is not None else 0
    ).fillna(0).astype(float)

    print(features.crs)


    # --- 6. Exposure metric at reference RP ---
    if RIVER_EXPOSURE_RP in hazard_dict:
        features["exposure_abs_river_current"] = compute_exposure_metric_tent(
            features=features,
            hazard=hazard_dict[RIVER_EXPOSURE_RP],
            reference_rp=RIVER_EXPOSURE_RP,
            hazard_value_col=RIVER_HAZARD_COL,
            pga_threshold=0.0,
        ).values
    else:
        features["exposure_abs_river_current"] = np.nan

    # --- 7. Future climate scenarios ---
    if basin_data is not None:
        print("[river-TENT] Computing future climate scenario EADs...")
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

        for period_label in TEMP_LABELS:
            features[f"exposure_abs_river_{period_label}"] = features[
                "exposure_abs_river_current"
            ].copy()
    else:
        print("[river-TENT] No basin data, skipping future scenarios.")

    # --- 8. TENT-specific: per-RP vulnerability ratio + exposure length ---
    print("[river-TENT] Computing per-RP vulnerability ratios and exposure lengths...")
    features = attach_per_rp_columns(
        features=features,
        rp_results=rp_results,
        maxdam=maxdam_mean,
        hazard_name="river",
    )

    elapsed = time.time() - t0
    print(
        f"[river-TENT] Done in {elapsed:.1f}s. "
        f"Mean EAD: {features['EAD_mid_river_current'].mean():.2f}, "
        f"Total: {features['EAD_mid_river_current'].sum():.2e}"
    )
    return features


# ---------------------------------------------------------------------------
# Windstorm -- single-pass TENT version
# ---------------------------------------------------------------------------


def assess_windstorm_tent(
    features: gpd.GeoDataFrame,
    hazard_dir: Union[str, Path],
    vulnerability_path: Union[str, Path],
    asset_type: str,
    object_curve_exclusions: Optional[dict] = None,
    return_periods: Optional[list[int]] = None,
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Windstorm assessment with per-RP ratio and exposure length."""
    from hazard_windstorm import (
        prepare_wind_curves,
        load_windstorm_hazard,
        WIND_RETURN_PERIODS,
        WIND_HAZARD_COL,
        WIND_EXPOSURE_RP,
        WIND_POWER_OBJECT_TYPES,
    )
    from constants import WIND_PROTECTION_STANDARD_RP
    from hazard_river import get_country_bounds_4326

    if return_periods is None:
        return_periods = WIND_RETURN_PERIODS

    t0 = time.time()

    # Substations have no wind vulnerability model — return early with zeros
    if asset_type == "energy_buses":
        features = features.copy()
        for col in ["EAD_mid_windstorm_current", "EAD_min_windstorm_current",
                    "EAD_max_windstorm_current", "exposure_abs_windstorm_current"]:
            features[col] = 0.0
        print(f"[windstorm-TENT] energy_buses: substations not wind-sensitive, skipping.")
        return features

    features_wind = features.copy()
    if asset_type in ("power", "energy_lines"):
        features_wind = features_wind[
            features_wind["object_type"].isin(WIND_POWER_OBJECT_TYPES)
        ]
        print(f"[windstorm-TENT] Power filter: {len(features_wind)}/{len(features)} retained")

    print(
        f"[windstorm-TENT] Starting assessment for {asset_type} "
        f"({len(features_wind)} features, {len(return_periods)} RPs)"
    )

    # --- 1. Vulnerability curves ---
    # Note: port TC curves use max_wind (km/h); verify wind raster units match.
    try:
        if asset_type in ("ports", "iww"):
            damage_curves, multi_curves, maxdam_mean, _, _ = _prepare_port_curves(
                "windstorm", Path(vulnerability_path).parent
            )
        else:
            damage_curves, multi_curves, maxdam_mean, _, _ = prepare_wind_curves(
                asset_type, vulnerability_path
            )
    except (ValueError, KeyError) as e:
        print(f"[windstorm-TENT] {e} -- skipping.")
        for col in ["EAD_mid_windstorm_current", "EAD_min_windstorm_current",
                     "EAD_max_windstorm_current", "exposure_abs_windstorm_current"]:
            features[col] = np.nan
        return features

    # --- 2. Hazard data ---
    country_bounds = get_country_bounds_4326(features_wind)
    hazard_dict = load_windstorm_hazard(hazard_dir, return_periods, country_bounds)

    if not hazard_dict:
        print("[windstorm-TENT] No hazard data found.")
        for col in ["EAD_mid_windstorm_current", "EAD_min_windstorm_current",
                     "EAD_max_windstorm_current", "exposure_abs_windstorm_current"]:
            features[col] = np.nan
        return features

    available_rps = sorted(hazard_dict.keys())

    # --- 3. Parallel damage calculation (TENT version) ---
    common = (
        features_wind, damage_curves, multi_curves, maxdam_mean,
        asset_type, object_curve_exclusions,
    )
    work_items = [(rp, hazard_dict[rp]) for rp in available_rps]
    worker_fn = functools.partial(_compute_wind_rp_damage_tent, common=common)

    print(f"[windstorm-TENT] Running damage across {len(work_items)} RPs...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init
    ) as executor:
        raw_results = list(
            tqdm(
                executor.map(worker_fn, work_items),
                total=len(work_items),
                desc="Windstorm RPs",
            )
        )

    rp_results = {rp: df for rp, df in raw_results}

    # --- 4. Integrate EAD (RP50 design standard per IEC 60826 lower bound) ---
    print("[windstorm-TENT] Integrating EAD...")
    wind_protection = pd.Series(
        WIND_PROTECTION_STANDARD_RP, index=features_wind.index, dtype=float
    )
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features_wind,
        protection_standards=wind_protection,
    )

    features = features.copy()
    features["EAD_mid_windstorm_current"] = 0.0
    features["EAD_min_windstorm_current"] = 0.0
    features["EAD_max_windstorm_current"] = 0.0

    features.loc[features_wind.index, "EAD_mid_windstorm_current"] = ead_df["EAD_mid"].values
    features.loc[features_wind.index, "EAD_min_windstorm_current"] = ead_df["EAD_min"].values
    features.loc[features_wind.index, "EAD_max_windstorm_current"] = ead_df["EAD_max"].values

    # --- 5. Exposure metric ---
    if WIND_EXPOSURE_RP in hazard_dict:
        exposure = compute_exposure_metric_tent(
            features=features_wind,
            hazard=hazard_dict[WIND_EXPOSURE_RP],
            reference_rp=WIND_EXPOSURE_RP,
            hazard_value_col=WIND_HAZARD_COL,
            pga_threshold=0.0,
        )
        features["exposure_abs_windstorm_current"] = 0.0
        features.loc[features_wind.index, "exposure_abs_windstorm_current"] = exposure.values
    else:
        features["exposure_abs_windstorm_current"] = np.nan

    # --- 6. Per-RP columns ---
    print("[windstorm-TENT] Computing per-RP vulnerability ratios and exposure lengths...")
    features_wind = attach_per_rp_columns(
        features=features_wind,
        rp_results=rp_results,
        maxdam=maxdam_mean,
        hazard_name="windstorm",
    )
    per_rp_cols = [c for c in features_wind.columns
                   if c.startswith("vuln_ratio_windstorm_")
                   or c.startswith("exposure_length_windstorm_")
                   or c.startswith("exposure_area_windstorm_")
                   or c.startswith("exposure_count_windstorm_")]
    for col in per_rp_cols:
        features[col] = 0.0
        features.loc[features_wind.index, col] = features_wind[col].values

    elapsed = time.time() - t0
    print(
        f"[windstorm-TENT] Done in {elapsed:.1f}s. "
        f"Mean EAD: {features['EAD_mid_windstorm_current'].mean():.2f}"
    )
    return features


# ---------------------------------------------------------------------------
# Earthquake -- single-pass TENT version
# ---------------------------------------------------------------------------


def assess_earthquake_tent(
    features: gpd.GeoDataFrame,
    hazard_dir: Union[str, Path],
    fragility_path: Union[str, Path],
    asset_type: str,
    return_periods: Optional[list[int]] = None,
    pga_threshold: float = 0.1,
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Earthquake assessment with per-RP ratio and exposure length."""
    from hazard_earthquake import (
        prepare_earthquake_fragility,
        load_earthquake_hazard,
        EQ_RETURN_PERIODS,
        EQ_HAZARD_COL,
        EQ_EXPOSURE_RP,
    )
    from hazard_river import get_country_bounds_4326

    if return_periods is None:
        return_periods = EQ_RETURN_PERIODS

    t0 = time.time()
    print(
        f"[earthquake-TENT] Starting assessment for {asset_type} "
        f"({len(features)} features, {len(return_periods)} RPs)"
    )

    # --- 1. Fragility / vulnerability curves ---
    try:
        if asset_type in ("ports", "iww"):
            fragility_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max = (
                _prepare_port_curves("earthquake", Path(fragility_path).parent)
            )
        else:
            fragility_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max = (
                prepare_earthquake_fragility(asset_type, fragility_path)
            )

        print(fragility_curves.head())
        print(multi_curves.keys())

    except ValueError as e:
        print(f"[earthquake-TENT] {e} -- skipping.")
        for col in ["EAD_mid_earthquake_current", "EAD_min_earthquake_current",
                     "EAD_max_earthquake_current", "exposure_abs_earthquake_current"]:
            features[col] = np.nan
        return features

    # --- 2. Hazard data ---
    country_bounds = get_country_bounds_4326(features)
    hazard_dict = load_earthquake_hazard(hazard_dir, return_periods, country_bounds)

    for rp, ds in hazard_dict.items():
        print(f"[eq debug] RP{rp}: max PGA = {float(ds['band_data'].max()):.4f}g")

    if not hazard_dict:
        print("[earthquake-TENT] No hazard data found.")
        for col in ["EAD_mid_earthquake_current", "EAD_min_earthquake_current",
                     "EAD_max_earthquake_current", "exposure_abs_earthquake_current"]:
            features[col] = np.nan
        return features

    available_rps = sorted(hazard_dict.keys())

    # --- 3. Parallel damage calculation ---
    # Ports use VectorScanner (direct damage ratio curves); others use EDR fragility path
    if asset_type in ("ports", "iww"):
        # Use same VectorScanner worker as river/wind — direct damage ratio curves
        common = (
            features, fragility_curves, multi_curves,
            maxdam_mean, asset_type, None,
        )
        worker_fn = functools.partial(_compute_rp_damage_tent, common=common)
    else:
        common = (
            features, fragility_curves, multi_curves,
            maxdam_mean, maxdam_min, maxdam_max, asset_type,
        )
        worker_fn = functools.partial(_compute_eq_rp_damage_tent, common=common)
    work_items = [(rp, hazard_dict[rp]) for rp in available_rps]

    print(f"[earthquake-TENT] Running damage across {len(work_items)} RPs...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init
    ) as executor:
        raw_results = list(
            tqdm(
                executor.map(worker_fn, work_items),
                total=len(work_items),
                desc="Earthquake RPs",
            )
        )

    rp_results = {rp: df for rp, df in raw_results if not df.empty}

    if not rp_results:
        print("[earthquake-TENT] No damage computed.")
        for col in ["EAD_mid_earthquake_current", "EAD_min_earthquake_current",
                     "EAD_max_earthquake_current", "exposure_abs_earthquake_current"]:
            features[col] = 0.0
        return features

    # --- DEBUG: check raw damage values before EAD integration ---
    for rp, df in sorted(rp_results.items()):
        if len(df) > 0 and "damage_mean" in df.columns:
            print(f"[eq-debug] RP{rp}: n_rows={len(df)}, "
                  f"max_damage_mean={df['damage_mean'].max():.3e}, "
                  f"sum_damage_mean={df['damage_mean'].sum():.3e}")

    # --- 4. Integrate EAD ---
    print("[earthquake-TENT] Integrating EAD...")
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features,
        protection_standards=None,
    )

    features = features.copy()
    features["EAD_mid_earthquake_current"] = ead_df["EAD_mid"].values
    features["EAD_min_earthquake_current"] = ead_df["EAD_min"].values
    features["EAD_max_earthquake_current"] = ead_df["EAD_max"].values

    # --- 5. Exposure metric ---
    if EQ_EXPOSURE_RP in hazard_dict:
        features["exposure_abs_earthquake_current"] = compute_exposure_metric_tent(
            features=features,
            hazard=hazard_dict[EQ_EXPOSURE_RP],
            reference_rp=EQ_EXPOSURE_RP,
            hazard_value_col=EQ_HAZARD_COL,
            pga_threshold=pga_threshold,
        ).values
    else:
        features["exposure_abs_earthquake_current"] = np.nan

    # --- 6. Per-RP columns ---
    print("[earthquake-TENT] Computing per-RP vulnerability ratios and exposure lengths...")
    features = attach_per_rp_columns(
        features=features,
        rp_results=rp_results,
        maxdam=maxdam_mean,
        hazard_name="earthquake",
    )

    elapsed = time.time() - t0
    print(
        f"[earthquake-TENT] Done in {elapsed:.1f}s. "
        f"Mean EAD: {features['EAD_mid_earthquake_current'].mean():.2f}"
    )
    return features


# ---------------------------------------------------------------------------
# Coastal -- TENT version
# ---------------------------------------------------------------------------


def assess_coastal_tent(
    features: gpd.GeoDataFrame,
    vulnerability_path: Union[str, Path],
    asset_type: str,
    stac_catalog_url: str = "https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json",
    object_curve_exclusions: Optional[dict] = None,
    scenarios: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """
    Coastal flood assessment for TENT data.

    Maps TENT 'id' column to 'osm_id' for coastal aggregation.
    Per-RP ratio/exposure columns for coastal require modifying the
    tile-streaming internals and can be added later if needed.
    """
    from hazard_coastal import assess_coastal
    from hazard_river import prepare_flood_curves

    features = features.copy()
    if "osm_id" not in features.columns and "id" in features.columns:
        print("[coastal-TENT] Mapping 'id' -> 'osm_id' for coastal aggregation")
        features["osm_id"] = features["id"]
    elif "osm_id" not in features.columns:
        print("[coastal-TENT] No id/osm_id column — using integer index as osm_id")
        features["osm_id"] = features.index.astype(str)

    if "LAU" not in features.columns:
        features["LAU"] = np.nan

    pre_loaded = None
    if asset_type in ("ports", "iww"):
        _dc, _mc, _md, _, _ = _prepare_port_curves(
            "flood", Path(vulnerability_path).parent
        )
        pre_loaded = (_dc, _mc, _md)

    result, all_rp_results = assess_coastal(
        features=features,
        vulnerability_path=vulnerability_path,
        asset_type=asset_type,
        stac_catalog_url=stac_catalog_url,
        object_curve_exclusions=object_curve_exclusions,
        scenarios=scenarios,
        return_rp_results=True,
        pre_loaded_curves=pre_loaded,
    )

    # Attach per-RP vuln_ratio + exposure columns for all scenarios.
    # all_rp_results: {scenario_key: rp_results} e.g. {"current": {...}, "2050_SSP245": {...}}
    # Baseline uses hazard_name="coastal" (no suffix) for backward compat.
    # Future scenarios use "coastal_2050_SSP245" etc.
    if any(v for v in all_rp_results.values()):
        print("[coastal-TENT] Computing per-RP vulnerability ratios and exposure areas...")
        if asset_type in ("ports", "iww"):
            _, _, maxdam_mean, _, _ = _prepare_port_curves("flood", Path(vulnerability_path).parent)
        else:
            _, _, maxdam_mean, _, _ = prepare_flood_curves(asset_type, vulnerability_path)
        for scenario_key, rp_results in all_rp_results.items():
            if not rp_results:
                continue
            hazard_name = "coastal" if scenario_key == "current" else f"coastal_{scenario_key}"
            print(f"[coastal-TENT] Attaching per-RP columns for {hazard_name}...")
            result = attach_per_rp_columns(
                features=result,
                rp_results=rp_results,
                maxdam=maxdam_mean,
                hazard_name=hazard_name,
            )
    else:
        print("[coastal-TENT] No coastal damage found — skipping per-RP columns.")

    return result
