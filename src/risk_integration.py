"""
risk_integration.py

Shared core for all hazard risk assessments.

Responsibilities:
  - Run VectorScanner per return period and collect damage arrays
  - Integrate damage curves to EAD (mean / min / max) via trapezoid rule
  - Compute geometry-aware exposure metrics at a reference return period
  - Adjust return periods for climate scenarios (future river)

All hazard modules (hazard_river, hazard_coastal, etc.) call into this module.
They are responsible for:
  - Loading the correct hazard rasters per return period
  - Providing the right vulnerability curves / maxdam
  - Calling the functions here with the assembled data
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional
from scipy.interpolate import interp1d

from damagescanner.core import VectorScanner, VectorExposure


# ---------------------------------------------------------------------------
# Damage calculation per return period
# ---------------------------------------------------------------------------


def compute_damage_per_rp(
    features: gpd.GeoDataFrame,
    hazard,
    curve_path: pd.DataFrame,
    maxdam: pd.DataFrame,
    asset_type: str,
    multi_curves: dict,
    object_col: str = "object_type",
    hazard_value_col: str = "band_data",
) -> pd.DataFrame:
    """
    Run VectorScanner for a single return period hazard map.

    Uses multi_curves to compute damage for all vulnerability curves
    (min / mean / max) in a single pass.

    Args:
        features:        Exposure GeoDataFrame
        hazard:          Hazard raster (xarray Dataset, DataArray, or path)
        curve_path:      Vulnerability curves DataFrame
        maxdam:          Maximum damage values DataFrame
        asset_type:      Internal asset type name (e.g. 'power')
        multi_curves:    Dict mapping curve_id → curve DataFrame (for uncertainty)
        object_col:      Column name for object type in features
        hazard_value_col: Band name in hazard dataset

    Returns:
        GeoDataFrame with damage columns per curve, indexed by osm_id
    """

    features.geometry.iloc[0].geom_type

    result = VectorScanner(
        hazard_file=hazard,
        feature_file=features,
        curve_path=curve_path,
        maxdam_path=maxdam,
        asset_type=asset_type,
        multi_curves=multi_curves,
        object_col=object_col,
        hazard_value_col=hazard_value_col,
        disable_progress=True,
        return_full=False,
    )

    return result


# ---------------------------------------------------------------------------
# EAD integration
# ---------------------------------------------------------------------------


def integrate_ead(
    damages_by_rp: dict[int, dict],
    protection_standard: float = 0,
) -> tuple[float, float, float]:
    """
    Integrate damage-probability curve to Expected Annual Damage (EAD).

    Args:
        damages_by_rp:      {return_period: {'mean': val, 'min': val, 'max': val}}
        protection_standard: Return period of flood protection (0 = no protection)

    Returns:
        Tuple of (ead_mean, ead_min, ead_max)
    """
    if not damages_by_rp:
        return 0.0, 0.0, 0.0

    sorted_rps = sorted(damages_by_rp.keys())

    results = {}
    for stat in ("mean", "min", "max"):
        sorted_damages = [damages_by_rp[rp].get(stat, 0.0) for rp in sorted_rps]

        rps, damages = _apply_protection_standard(
            sorted_rps, sorted_damages, protection_standard
        )

        if len(damages) >= 2:
            probs = [1.0 / rp for rp in rps]
            ead = float(np.trapezoid(y=damages[::-1], x=probs[::-1]))
        elif len(damages) == 1:
            ead = (1.0 / rps[0]) * damages[0]
        else:
            ead = 0.0

        results[stat] = max(ead, 0.0)

    return results["mean"], results["min"], results["max"]


def _apply_protection_standard(
    sorted_rps: list,
    sorted_damages: list,
    protection_standard: float,
) -> tuple[list, list]:
    """
    Filter/interpolate damage curve to account for flood protection standard.

    Events with return period < protection_standard are considered protected
    (damage = 0 up to that threshold, then linearly interpolated).
    """
    if protection_standard <= 0:
        return sorted_rps, sorted_damages

    # Interpolate damage at the protection standard RP if not already present
    if protection_standard not in sorted_rps:
        interp_damage = _interpolate_damage(
            sorted_rps, sorted_damages, protection_standard
        )
        idx = int(np.searchsorted(sorted_rps, protection_standard))
        sorted_rps = sorted_rps[:idx] + [protection_standard] + sorted_rps[idx:]
        sorted_damages = sorted_damages[:idx] + [interp_damage] + sorted_damages[idx:]

    # Keep only RPs >= protection standard
    filtered = [
        (rp, d)
        for rp, d in zip(sorted_rps, sorted_damages)
        if rp >= protection_standard
    ]
    if not filtered:
        return [], []

    rps, damages = zip(*filtered)
    return list(rps), list(damages)


def _interpolate_damage(
    rp_values: list,
    damage_values: list,
    target_rp: float,
) -> float:
    """Linear interpolation of damage at a target return period."""
    idx = int(np.searchsorted(rp_values, target_rp))
    if idx == 0:
        return damage_values[0]
    if idx >= len(rp_values):
        return damage_values[-1]
    rp_low, rp_high = rp_values[idx - 1], rp_values[idx]
    d_low, d_high = damage_values[idx - 1], damage_values[idx]
    return d_low + (target_rp - rp_low) * (d_high - d_low) / (rp_high - rp_low)


# ---------------------------------------------------------------------------
# Collect per-asset EAD across all return periods
# ---------------------------------------------------------------------------


def collect_ead_per_asset(
    rp_results: dict[int, gpd.GeoDataFrame],
    features: gpd.GeoDataFrame,
    protection_standards: Optional[pd.Series] = None,
    damage_col_mean: str = "damage_mean",
    damage_col_min: str = "damage_min",
    damage_col_max: str = "damage_max",
) -> pd.DataFrame:
    """
    Collect per-return-period damage arrays and integrate to EAD per asset.

    Args:
        rp_results:          Dict of {return_period: GeoDataFrame with damage columns}
        features:            Original exposure GeoDataFrame (for index / osm_id)
        protection_standards: Series mapping asset index → protection RP (0 if none)
        damage_col_mean/min/max: Column names for damage stats in rp_results

    Returns:
        DataFrame with columns [EAD, EAD_min, EAD_max] indexed like features
    """
    sorted_rps = sorted(rp_results.keys())
    n = len(features)

    # Pre-extract all damage values into (n_features × n_rps) numpy matrices.
    # One vectorised .reindex per RP instead of n_features × n_rps .loc calls.
    mean_mat = np.zeros((n, len(sorted_rps)))
    min_mat = np.zeros((n, len(sorted_rps)))
    max_mat = np.zeros((n, len(sorted_rps)))

    for j, rp in enumerate(sorted_rps):
        aligned = rp_results[rp].reindex(features.index)
        mean_mat[:, j] = aligned[damage_col_mean].fillna(0).to_numpy()
        min_mat[:, j] = aligned[damage_col_min].fillna(0).to_numpy()
        max_mat[:, j] = aligned[damage_col_max].fillna(0).to_numpy()

    ead_mean = np.zeros(n)
    ead_min = np.zeros(n)
    ead_max = np.zeros(n)

    for i, idx in enumerate(features.index):
        prot = (
            float(protection_standards.get(idx, 0))
            if protection_standards is not None
            else 0.0
        )
        damages_by_rp = {
            rp: {
                "mean": mean_mat[i, j],
                "min": min_mat[i, j],
                "max": max_mat[i, j],
            }
            for j, rp in enumerate(sorted_rps)
        }
        m, lo, hi = integrate_ead(damages_by_rp, protection_standard=prot)
        ead_mean[i] = m
        ead_min[i] = lo
        ead_max[i] = hi

    return pd.DataFrame(
        {"EAD": ead_mean, "EAD_min": ead_min, "EAD_max": ead_max},
        index=features.index,
    )


# ---------------------------------------------------------------------------
# Exposure metrics (geometry-aware)
# ---------------------------------------------------------------------------


def compute_exposure_metric(
    features: gpd.GeoDataFrame,
    hazard,
    reference_rp: int,
    hazard_value_col: str = "band_data",
    pga_threshold: float = 0.0,
) -> pd.Series:
    """
    Compute geometry-aware exposure at a reference return period.

    Returns:
      - LineStrings:  total exposed length in metres
      - Polygons:     total exposed area in m²
      - Points:       count of exposed assets

    For earthquake (pga_threshold > 0), only counts assets where
    mean PGA exceeds the threshold.

    Args:
        features:       Exposure GeoDataFrame (EPSG:3035)
        hazard:         Hazard raster at the reference return period
        reference_rp:   Return period label (for logging only)
        hazard_value_col: Band name in hazard dataset
        pga_threshold:  Minimum hazard value to count as exposed (default 0 = any inundation)

    Returns:
        Series of exposure values indexed like features (0 where not exposed)
    """
    # Extract hazard values onto features
    exposed_features, _, crs, cell_area = VectorExposure(
        hazard_file=hazard,
        feature_file=features,
        hazard_value_col=hazard_value_col,
        disable_progress=True,
    )

    # Filter to exposed assets only (hazard value > threshold)
    if "values" not in exposed_features.columns:
        return pd.Series(0.0, index=features.index)

    # Mean hazard value per asset
    exposed_features["_mean_hazard"] = exposed_features["values"].apply(
        lambda v: (
            float(np.mean(v)) if hasattr(v, "__len__") and len(v) > 0 else float(v or 0)
        )
    )

    exposed = exposed_features[exposed_features["_mean_hazard"] > pga_threshold].copy()

    if len(exposed) == 0:
        return pd.Series(0.0, index=features.index)

    # Reproject to metric CRS for correct area/length
    exposed_metric = exposed.to_crs(epsg=3035)

    # Vectorised geometry metric — no row-wise apply
    geom_types = exposed_metric.geometry.geom_type
    exposure_values = pd.Series(0.0, index=exposed_metric.index)
    is_line = geom_types.isin(["LineString", "MultiLineString"])
    is_poly = geom_types.isin(["Polygon", "MultiPolygon"])
    is_point = ~is_line & ~is_poly
    if is_line.any():
        exposure_values[is_line] = exposed_metric.geometry[is_line].length
    if is_poly.any():
        exposure_values[is_poly] = exposed_metric.geometry[is_poly].area
    if is_point.any():
        exposure_values[is_point] = 1.0

    # Reindex to full feature set (0 for unexposed)
    return exposure_values.reindex(features.index, fill_value=0.0)


# ---------------------------------------------------------------------------
# Climate return period adjustment (future river)
# ---------------------------------------------------------------------------


def adjust_return_periods_climate(
    original_rps: list[int],
    protection_standard: float,
    basin_changes: pd.Series,
    temp_scenario: str,
) -> tuple[list[float], float]:
    """
    Adjust return periods for a given temperature scenario using
    basin-level absolute RP shift data.

    Anchor points: RP10, RP100, RP500 → new RPs from basin data.
    All other RPs are linearly interpolated between anchors.

    Args:
        original_rps:       Original return periods (e.g. [10,20,50,100,200,500])
        protection_standard: Original protection standard RP (0 = none)
        basin_changes:      Row from basin dataframe with columns like
                            '10_rp_change_{temp}', '100_rp_change_{temp}', '500_rp_change_{temp}'
        temp_scenario:      Temperature string used in column names (e.g. '15','20','30','40')

    Returns:
        Tuple of (adjusted_rps, adjusted_protection_standard)
    """
    # Read anchor point new RPs, fall back to original if NaN
    new_rp_10 = _safe_rp(
        basin_changes.get(f"10_rp_change_{temp_scenario}"), 10, 1.0, 99.0
    )
    new_rp_100 = _safe_rp(
        basin_changes.get(f"100_rp_change_{temp_scenario}"), 100, 1.0, 499.0
    )
    new_rp_500 = _safe_rp(
        basin_changes.get(f"500_rp_change_{temp_scenario}"), 500, 1.0, 1000.0
    )

    interp_func = interp1d(
        [10, 100, 500],
        [new_rp_10, new_rp_100, new_rp_500],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    adjusted_rps = [max(float(interp_func(rp)), 1.0) for rp in original_rps]

    adjusted_protection = protection_standard
    if protection_standard > 0:
        adjusted_protection = max(float(interp_func(protection_standard)), 1.0)

    return adjusted_rps, adjusted_protection


def _safe_rp(value, default: float, lo: float, hi: float) -> float:
    """Return value clamped to [lo, hi], falling back to default if NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(np.clip(value, lo, hi))


def collect_ead_climate_scenarios(
    rp_results: dict[int, gpd.GeoDataFrame],
    features: gpd.GeoDataFrame,
    basin_data: pd.DataFrame,
    basin_ids: pd.Series,
    protection_standards: Optional[pd.Series] = None,
    temp_scenarios: tuple[str, ...] = ("15", "20", "30", "40"),
    temp_labels: tuple[str, ...] = ("1.5C", "2.0C", "3.0C", "4.0C"),
    damage_col_mean: str = "damage_mean",
    damage_col_min: str = "damage_min",
    damage_col_max: str = "damage_max",
) -> pd.DataFrame:
    """
    Compute climate-adjusted EAD for all temperature scenarios.

    Uses the damage arrays already computed for the base river assessment
    (rp_results) — no new VectorScanner calls needed.

    Damage matrices are pre-extracted once and shared across all scenarios
    to avoid repeated pandas .loc calls.

    Args:
        rp_results:          Dict of {return_period: GeoDataFrame} from base river run
        features:            Exposure GeoDataFrame
        basin_data:          GeoDataFrame with basin-level RP shift data (index = HYBAS_ID)
        basin_ids:           Series mapping feature index → HYBAS_ID
        protection_standards: Series mapping asset index → protection RP
        temp_scenarios:      Short labels used in basin_data columns ('15','20','30','40')
        temp_labels:         Output column labels ('1.5C','2.0C','3.0C','4.0C')

    Returns:
        DataFrame with columns [EAD_river_{label}, EAD_river_{label}_min, EAD_river_{label}_max]
        for each temperature scenario, indexed like features
    """
    sorted_rps = sorted(rp_results.keys())
    n = len(features)

    # Pre-extract damage matrices once — shared across all temperature scenarios
    mean_mat = np.zeros((n, len(sorted_rps)))
    min_mat = np.zeros((n, len(sorted_rps)))
    max_mat = np.zeros((n, len(sorted_rps)))

    for j, rp in enumerate(sorted_rps):
        aligned = rp_results[rp].reindex(features.index)
        mean_mat[:, j] = aligned[damage_col_mean].fillna(0).to_numpy()
        min_mat[:, j] = aligned[damage_col_min].fillna(0).to_numpy()
        max_mat[:, j] = aligned[damage_col_max].fillna(0).to_numpy()

    output = {}

    for temp_code, temp_label in zip(temp_scenarios, temp_labels):
        ead_mean = np.zeros(n)
        ead_min = np.zeros(n)
        ead_max = np.zeros(n)

        for i, idx in enumerate(features.index):
            prot = (
                float(protection_standards.get(idx, 0))
                if protection_standards is not None
                else 0.0
            )

            basin_id = basin_ids.get(idx)
            if (
                basin_id is not None
                and not pd.isna(basin_id)
                and basin_id in basin_data.index
            ):
                basin_row = basin_data.loc[basin_id]
            else:
                basin_row = pd.Series(dtype=float)

            adjusted_rps, adjusted_prot = adjust_return_periods_climate(
                sorted_rps, prot, basin_row, temp_code
            )

            damages_by_rp = {
                new_rp: {
                    "mean": mean_mat[i, j],
                    "min": min_mat[i, j],
                    "max": max_mat[i, j],
                }
                for j, (orig_rp, new_rp) in enumerate(zip(sorted_rps, adjusted_rps))
            }

            m, lo, hi = integrate_ead(damages_by_rp, protection_standard=adjusted_prot)
            ead_mean[i] = m
            ead_min[i] = lo
            ead_max[i] = hi

        output[f"EAD_river_{temp_label}"] = ead_mean
        output[f"EAD_river_{temp_label}_min"] = ead_min
        output[f"EAD_river_{temp_label}_max"] = ead_max

    return pd.DataFrame(output, index=features.index)
