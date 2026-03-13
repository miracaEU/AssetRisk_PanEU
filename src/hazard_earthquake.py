"""
hazard_earthquake.py

Earthquake risk assessment module.

Takes a pre-loaded exposure GeoDataFrame and returns it enriched with:
  - EAD_earthquake, EAD_earthquake_min, EAD_earthquake_max
  - exposure_eq_475  (length / area / count at RP475 where PGA > 0.1g)

Uses PGA-based fragility curves (lognormal CDF parameterised by median + beta).
Damage states (minor / moderate / extensive / severe / complete) are combined
into an expected damage ratio using probability weighting.

No protection standards for earthquake.
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
from scipy.stats import norm
from tqdm import tqdm
from typing import Optional, Union

from damagescanner.core import VectorExposure

from risk_integration import (
    collect_ead_per_asset,
    compute_exposure_metric,
)

from constants import (
    DICT_CIS_VULNERABILITY_EARTHQUAKE,
    INFRASTRUCTURE_DAMAGE_VALUES,
)


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _worker_init():
    import sys

    sys.excepthook = lambda *args: None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EQ_RETURN_PERIODS = [50, 101, 476, 976, 2500, 5000]  # from actual hazard files
EQ_FILENAME_TEMPLATE = "PGA_1_{rp}_vs30.tif"
EQ_HAZARD_COL = "band_data"
EQ_EXPOSURE_RP = 476  # closest available RP to standard 475yr
EQ_PGA_THRESHOLD = 0.1  # g — minimum PGA to count as exposed

# PGA intensity measure range (g) — matches your fragility curve definition
EQ_PGA_RANGE = np.arange(0.0, 3.35, 0.05)

# Damage state → mean loss ratio mapping
DAMAGE_RATIOS = {
    "minor": 0.05,
    "moderate": 0.20,
    "extensive": 0.70,
    "severe": 0.85,
    "complete": 1.00,
    "collapse": 1.00,
}

# Damage state name standardisation
DAMAGE_STATE_MAP = {
    "Slight": "minor",
    "Minor": "minor",
    "DS1": "minor",
    "Moderate": "moderate",
    "DS2": "moderate",
    "Extensive": "extensive",
    "DS3": "extensive",
    "Severe": "severe",
    "DS4": "severe",
    "Complete": "complete",
    "DS5": "complete",
    "Collapse": "collapse",
}

# ---------------------------------------------------------------------------
# Fragility curve preparation
# ---------------------------------------------------------------------------


def _standardise_damage_states(fragility_curves: pd.DataFrame) -> pd.DataFrame:
    """Rename damage state columns to standardised names."""
    new_cols = [
        (curve_id, DAMAGE_STATE_MAP.get(ds, ds.lower()))
        for curve_id, ds in fragility_curves.columns
    ]
    fragility_curves.columns = pd.MultiIndex.from_tuples(new_cols)
    return fragility_curves


def _is_curve_parametric(fragility_df: pd.DataFrame, curve_id: str) -> bool:
    """Return True if a curve uses parametric (median/beta) format."""
    col = next((c for c in fragility_df.columns if c[0] == curve_id), None)
    if col is None:
        return False
    vals = [str(v).lower() for v in fragility_df[col].dropna()]
    return any("median" in v for v in vals) and any("beta" in v for v in vals)


def prepare_earthquake_fragility(
    asset_type: str,
    fragility_path: Union[str, Path],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare earthquake fragility curves for an asset type.

    Handles both:
      - Pre-computed curves (PGA as index, damage state probabilities as columns)
      - Parametric curves (median + beta per damage state → generated via lognormal CDF)
      - Mixed: some curves parametric, some pre-computed (e.g. healthcare E21.67-C)

    Args:
        asset_type:     Internal asset type name
        fragility_path: Path to fragility Excel file (sheet: E_Frag_PGA)

    Returns:
        (fragility_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max)
    """
    fragility_df = pd.read_excel(fragility_path, sheet_name="E_Frag_PGA", header=[0, 1])

    ci_system = DICT_CIS_VULNERABILITY_EARTHQUAKE.get(asset_type, {})
    if not ci_system:
        raise ValueError(
            f"No earthquake fragility curves defined for asset type '{asset_type}'. "
            f"Available: {list(DICT_CIS_VULNERABILITY_EARTHQUAKE.keys())}"
        )

    unique_curves = {c for curves in ci_system.values() for c in curves}

    # Validate at least some curves exist in the Excel
    available_level0 = set(fragility_df.columns.get_level_values(0))
    found_curves = unique_curves & available_level0
    if not found_curves:
        raise ValueError(
            f"No fragility curves found for {asset_type}. "
            f"Required: {unique_curves}, Available: {list(available_level0)}"
        )

    # Split curves into parametric vs pre-computed
    parametric_curves = {
        c for c in unique_curves if _is_curve_parametric(fragility_df, c)
    }
    precomputed_curves = unique_curves - parametric_curves

    # Load each group and concatenate
    frames = []
    if parametric_curves:
        frames.append(_build_curves_from_parameters(fragility_df, parametric_curves))
    if precomputed_curves:
        frames.append(_load_precomputed_curves(fragility_df, precomputed_curves))

    fragility_curves = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]

    # Standardise damage state names
    fragility_curves = _standardise_damage_states(fragility_curves)

    # Build multi_curves dict
    multi_curves = {
        curve_id: fragility_curves
        for curve_id in unique_curves
        if any(col[0] == curve_id for col in fragility_curves.columns)
    }

    # Max damage values (min / mean / max)
    asset_maxdam = INFRASTRUCTURE_DAMAGE_VALUES.get(asset_type, {})

    def _make_maxdam(idx: int) -> pd.DataFrame:
        d = {k: v[idx] for k, v in asset_maxdam.items() if k in ci_system}
        df = pd.DataFrame.from_dict(d, orient="index").reset_index()
        df.columns = ["object_type", "damage"]
        return df

    return (
        fragility_curves,
        multi_curves,
        _make_maxdam(1),
        _make_maxdam(0),
        _make_maxdam(2),
    )


def _build_curves_from_parameters(
    param_df: pd.DataFrame,
    unique_curves: set,
) -> pd.DataFrame:
    """Generate fragility curves from median/beta parameters using lognormal CDF."""
    result = pd.DataFrame(index=EQ_PGA_RANGE)

    for col in param_df.columns:
        curve_id, damage_state = col
        if curve_id not in unique_curves:
            continue

        column_data = param_df[col]
        str_vals = [str(v).lower() for v in column_data if not pd.isna(v)]

        # Skip columns that don't contain median/beta parameters
        if not (
            any("median" in v for v in str_vals) and any("beta" in v for v in str_vals)
        ):
            continue

        # Extract median and beta values
        median_val = beta_val = None
        for i, cell in enumerate(column_data):
            if pd.isna(cell):
                continue
            if i > 0:
                prev = str(column_data.iloc[i - 1]).lower()
                if "median" in prev and median_val is None:
                    median_val = cell
                elif "beta" in prev and beta_val is None:
                    beta_val = cell

        if median_val is None or beta_val is None:
            continue
        try:
            median_val = float(str(median_val).replace(",", "."))
            beta_val = float(str(beta_val).replace(",", "."))
        except (ValueError, TypeError):
            continue

        if median_val <= 0 or beta_val <= 0:
            continue

        ln_pga = np.log(np.maximum(EQ_PGA_RANGE, 1e-6))
        ln_median = np.log(median_val)
        probs = norm.cdf((ln_pga - ln_median) / beta_val)
        result[(curve_id, damage_state)] = np.clip(probs, 0, 1)

    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


def _load_precomputed_curves(
    fragility_df: pd.DataFrame,
    unique_curves: set,
) -> pd.DataFrame:
    """Load pre-computed (PGA-indexed) fragility curves."""
    pga_values = pd.to_numeric(fragility_df.iloc[:, 0], errors="coerce")
    valid = ~pga_values.isna()
    pga_values = pga_values[valid]

    data_cols = fragility_df.iloc[:, 1:]
    data_vals = pd.DataFrame(data_cols.values[valid], columns=data_cols.columns).apply(
        pd.to_numeric, errors="coerce"
    )

    keep_cols = [col for col in data_vals.columns if col[0] in unique_curves]
    data_vals = data_vals[keep_cols]
    data_vals.index = pga_values

    return data_vals.ffill().fillna(0)


# ---------------------------------------------------------------------------
# Hazard data loading
# ---------------------------------------------------------------------------


def load_earthquake_hazard(
    hazard_dir: Union[str, Path],
    return_periods: list[int],
    country_bounds: Optional[tuple] = None,
    filename_template: str = EQ_FILENAME_TEMPLATE,
) -> dict[int, xr.Dataset]:
    """
    Load earthquake PGA hazard rasters for all return periods.

    Args:
        hazard_dir:         Directory containing earthquake hazard rasters
        return_periods:     List of return periods to load
        country_bounds:     Optional (minx, miny, maxx, maxy) in EPSG:4326 to clip
        filename_template:  Filename pattern with {rp} placeholder

    Returns:
        Dict mapping return_period → xarray Dataset
    """
    hazard_dir = Path(hazard_dir)
    hazard_dict = {}

    for rp in tqdm(return_periods, desc="Loading earthquake hazard rasters"):
        path = hazard_dir / filename_template.format(rp=rp)
        if not path.exists():
            print(f"  Warning: EQ hazard file not found for RP{rp}: {path}")
            continue
        try:
            ds = xr.open_dataset(path, engine="rasterio")
            if country_bounds is not None:
                minx, miny, maxx, maxy = country_bounds
                ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            hazard_dict[rp] = ds
        except Exception as e:
            print(f"  Warning: could not load EQ RP{rp}: {e}")

    return hazard_dict


# ---------------------------------------------------------------------------
# Per-asset fragility-based damage calculation — vectorised
# ---------------------------------------------------------------------------


def _build_edr_lookup(fragility_curves: pd.DataFrame, curve_id: str) -> np.ndarray:
    """
    Pre-compute an Expected Damage Ratio (EDR) lookup table for one curve.

    Returns a 1-D array of EDR values over the PGA index of fragility_curves,
    so downstream code can use np.interp(pga_values, pga_index, edr_table)
    instead of calling the scalar version per PGA cell.

    This is computed once per curve per RP worker and reused across all assets.
    """
    states = sorted(
        {col[1] for col in fragility_curves.columns if col[0] == curve_id},
        key=lambda s: DAMAGE_RATIOS.get(s, 0.5),
    )
    if not states:
        return np.zeros(len(fragility_curves))

    pga_index = fragility_curves.index.to_numpy(dtype=float)
    n_pga = len(pga_index)

    # Exceedance matrix: shape (n_pga, n_states)
    exceed = np.zeros((n_pga, len(states)))
    for j, state in enumerate(states):
        col = (curve_id, state)
        if col in fragility_curves.columns:
            exceed[:, j] = fragility_curves[col].to_numpy(dtype=float)

    # Individual state probabilities: P(state) = P(exceed state) - P(exceed next state)
    # Shape: (n_pga, n_states + 1)  — first col = P(no damage)
    n_states = len(states)
    individual = np.zeros((n_pga, n_states + 1))
    individual[:, 0] = np.maximum(0.0, 1.0 - exceed[:, 0])  # no damage
    for j in range(n_states - 1):
        individual[:, j + 1] = np.maximum(0.0, exceed[:, j] - exceed[:, j + 1])
    individual[:, n_states] = exceed[:, -1]  # complete/collapse

    # Normalise rows
    row_sums = individual.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    individual /= row_sums

    # Damage ratio weights: [0 for no_damage, then DAMAGE_RATIOS per state]
    weights = np.array([0.0] + [DAMAGE_RATIOS.get(s, 0.0) for s in states])

    # EDR = dot(individual, weights), shape (n_pga,)
    edr = np.clip(individual @ weights, 0.0, 1.0)
    return edr


def _compute_eq_rp_damage(args, common) -> tuple:
    """
    Worker: compute earthquake damage for a single return period.

    Vectorised implementation:
      1. Extract PGA values for all features at once via VectorExposure
      2. Pre-compute EDR lookup table per curve (once per curve, not per asset)
      3. For each object_type group, apply np.interp over all assets simultaneously
      4. Aggregate min/mean/max across curves per asset

    Returns (rp, DataFrame with damage_mean/min/max columns)
    """
    rp, hazard = args
    (
        features,
        fragility_curves,
        multi_curves,
        maxdam_mean,
        maxdam_min,
        maxdam_max,
        asset_type,
    ) = common

    # --- Extract PGA values ---
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

    # --- Maxdam lookups ---
    maxdam_lookup_mean = dict(zip(maxdam_mean["object_type"], maxdam_mean["damage"]))
    maxdam_lookup_min = dict(zip(maxdam_min["object_type"], maxdam_min["damage"]))
    maxdam_lookup_max = dict(zip(maxdam_max["object_type"], maxdam_max["damage"]))

    ci_system = DICT_CIS_VULNERABILITY_EARTHQUAKE.get(asset_type, {})
    pga_index = fragility_curves.index.to_numpy(dtype=float)

    # --- Pre-compute EDR lookup tables once per curve ---
    edr_tables: dict[str, np.ndarray] = {
        cid: _build_edr_lookup(fragility_curves, cid) for cid in multi_curves
    }

    # --- Process each object_type group ---
    # Grouping avoids repeated curve_id lookups; within each group all assets
    # share the same curve list so we can batch the np.interp calls.
    damage_mean = np.zeros(len(exposed))
    damage_min = np.zeros(len(exposed))
    damage_max = np.zeros(len(exposed))

    for obj_type, group_idx in exposed.groupby("object_type").groups.items():
        curve_ids = ci_system.get(obj_type, [])
        if not curve_ids:
            continue

        group = exposed.loc[group_idx]
        pos = [exposed.index.get_loc(i) for i in group_idx]

        md_mean = maxdam_lookup_mean.get(obj_type, 0.0)
        md_min = maxdam_lookup_min.get(obj_type, 0.0)
        md_max = maxdam_lookup_max.get(obj_type, 0.0)

        # --- Build padded matrices once per group (shared across all curves) ---
        values_list = [
            np.asarray(v if v is not None else [0], dtype=float)
            for v in group["values"].tolist()
        ]
        coverage_list = [
            np.asarray(c if c is not None else [0], dtype=float)
            for c in group["coverage"].tolist()
        ]

        n_assets = len(group)
        max_len = max(len(v) for v in values_list) if values_list else 1

        pga_mat = np.zeros((n_assets, max_len))
        cov_mat = np.zeros((n_assets, max_len))
        mask = np.zeros((n_assets, max_len), dtype=bool)

        for k, (pga_vals, cov_vals) in enumerate(zip(values_list, coverage_list)):
            n = len(pga_vals)
            pga_mat[k, :n] = pga_vals
            cov_mat[k, :n] = cov_vals
            mask[k, :n] = True

        # Geometry-specific coverage scaling -- done once per group
        geom_types_group = group.geometry.geom_type.to_numpy()
        is_poly_mask = np.isin(geom_types_group, ["Polygon", "MultiPolygon"])
        is_point_mask = ~np.isin(
            geom_types_group,
            ["Polygon", "MultiPolygon", "LineString", "MultiLineString"],
        )
        cov_mat_scaled = cov_mat.copy()
        cov_mat_scaled[is_poly_mask] *= cell_area
        cov_mat_scaled[is_point_mask] = 1.0

        # Accumulate damage across all curves for this object type
        curve_damages = []

        for cid in curve_ids:
            if cid not in edr_tables:
                continue

            edr_table = edr_tables[cid]

            # Vectorised EDR interpolation over all assets x all cells at once
            edr_flat = np.interp(pga_mat.ravel(), pga_index, edr_table)
            edr_mat = edr_flat.reshape(pga_mat.shape)

            # Sum over cells per asset (masked to ignore padding)
            asset_damages = np.sum(edr_mat * cov_mat_scaled * mask, axis=1) * md_mean
            curve_damages.append(asset_damages)

        if not curve_damages:
            continue

        curve_mat = np.vstack(curve_damages)  # (n_curves, n_assets_in_group)

        # Scale min/max variants by maxdam ratio (avoids re-running the loop)
        scale_min = md_min / md_mean if md_mean > 0 else 0.0
        scale_max = md_max / md_mean if md_mean > 0 else 0.0

        for k, p in enumerate(pos):
            damage_mean[p] = float(np.mean(curve_mat[:, k]))
            damage_min[p] = float(np.min(curve_mat[:, k])) * scale_min
            damage_max[p] = float(np.max(curve_mat[:, k])) * scale_max

    exposed["damage_mean"] = damage_mean
    exposed["damage_min"] = damage_min
    exposed["damage_max"] = damage_max

    return rp, exposed


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def assess_earthquake(
    features: gpd.GeoDataFrame,
    hazard_dir: Union[str, Path],
    fragility_path: Union[str, Path],
    asset_type: str,
    return_periods: list[int] = EQ_RETURN_PERIODS,
    pga_threshold: float = EQ_PGA_THRESHOLD,
    n_workers: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Assess earthquake risk for a pre-loaded exposure GeoDataFrame.

    Args:
        features:        Exposure GeoDataFrame (EPSG:3035)
        hazard_dir:      Directory with PGA hazard rasters
        fragility_path:  Path to fragility Excel file
        asset_type:      Internal asset type (e.g. 'power', 'roads')
        return_periods:  Return periods to assess (default: 475, 975, 2475)
        pga_threshold:   Minimum PGA (g) to count as exposed (default: 0.1g)
        n_workers:       Number of parallel workers (None = all CPUs)

    Returns:
        Input GeoDataFrame enriched with:
          EAD_earthquake, EAD_earthquake_min, EAD_earthquake_max
          exposure_eq_475
    """
    t0 = time.time()
    print(
        f"[earthquake] Starting assessment for {asset_type} "
        f"({len(features)} features, {len(return_periods)} return periods)"
    )

    # --- 1. Fragility curves ---
    try:
        fragility_curves, multi_curves, maxdam_mean, maxdam_min, maxdam_max = (
            prepare_earthquake_fragility(asset_type, fragility_path)
        )
    except ValueError as e:
        print(f"[earthquake] {e} — skipping earthquake for this asset type.")
        features["EAD_earthquake"] = np.nan
        features["EAD_earthquake_min"] = np.nan
        features["EAD_earthquake_max"] = np.nan
        features["exposure_eq_475"] = np.nan
        return features

    # --- 2. Hazard data ---
    from hazard_river import get_country_bounds_4326

    country_bounds = get_country_bounds_4326(features)
    hazard_dict = load_earthquake_hazard(hazard_dir, return_periods, country_bounds)

    if not hazard_dict:
        print("[earthquake] No hazard data found. Returning features unchanged.")
        features["EAD_earthquake"] = np.nan
        features["EAD_earthquake_min"] = np.nan
        features["EAD_earthquake_max"] = np.nan
        features["exposure_eq_475"] = np.nan
        return features

    available_rps = sorted(hazard_dict.keys())

    # --- 3. Parallel damage calculation per return period ---
    common = (
        features,
        fragility_curves,
        multi_curves,
        maxdam_mean,
        maxdam_min,
        maxdam_max,
        asset_type,
    )
    work_items = [(rp, hazard_dict[rp]) for rp in available_rps]
    worker_fn = functools.partial(_compute_eq_rp_damage, common=common)

    print(
        f"[earthquake] Running damage calculation across {len(work_items)} return periods..."
    )
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
        print("[earthquake] No damage computed.")
        features["EAD_earthquake"] = 0.0
        features["EAD_earthquake_min"] = 0.0
        features["EAD_earthquake_max"] = 0.0
        features["exposure_eq_475"] = 0.0
        return features

    # --- 4. Integrate EAD ---
    print("[earthquake] Integrating EAD...")
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features,
        protection_standards=None,
    )

    features = features.copy()
    features["EAD_earthquake"] = ead_df["EAD"].values
    features["EAD_earthquake_min"] = ead_df["EAD_min"].values
    features["EAD_earthquake_max"] = ead_df["EAD_max"].values

    # --- 5. Exposure metric at RP475 (count/length/area where PGA > threshold) ---
    if EQ_EXPOSURE_RP in hazard_dict:
        print(
            f"[earthquake] Computing exposure metric at RP{EQ_EXPOSURE_RP} "
            f"(PGA > {pga_threshold}g)..."
        )
        features["exposure_eq_475"] = compute_exposure_metric(
            features=features,
            hazard=hazard_dict[EQ_EXPOSURE_RP],
            reference_rp=EQ_EXPOSURE_RP,
            hazard_value_col=EQ_HAZARD_COL,
            pga_threshold=pga_threshold,
        ).values
    else:
        print(
            f"[earthquake] RP{EQ_EXPOSURE_RP} not available, skipping exposure metric."
        )
        features["exposure_eq_475"] = np.nan

    elapsed = time.time() - t0
    print(
        f"[earthquake] Done in {elapsed:.1f}s. "
        f"Mean EAD_earthquake: {features['EAD_earthquake'].mean():.2f}, "
        f"Total: {features['EAD_earthquake'].sum():.2e}"
    )

    return features
