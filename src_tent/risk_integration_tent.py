"""
risk_integration_tent.py

Extended risk integration for the TENT pipeline.

Adds on top of the standard risk_integration.py:
  - Per-RP vulnerability ratio (damage fraction with maxdam=1)
  - Per-RP exposure length computed from VectorScanner coverage data
    (actual intersected length, not full geometry length)
  - TENT-specific compute_damage_per_rp that retains coverage/values
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional

from damagescanner.vector import VectorScanner, VectorExposure

from risk_integration import (
    collect_ead_per_asset,
    integrate_ead,
    adjust_return_periods_climate,
    collect_ead_climate_scenarios,
)

# Re-export everything from risk_integration so hazard modules work unchanged
__all__ = [
    "compute_damage_per_rp_tent",
    "compute_exposure_metric_tent",
    "collect_ead_per_asset",
    "compute_exposure_metric",
    "integrate_ead",
    "adjust_return_periods_climate",
    "collect_ead_climate_scenarios",
    "compute_ratio_and_exposure_per_rp",
    "attach_per_rp_columns",
]


# ---------------------------------------------------------------------------
# TENT-specific damage computation (retains coverage for exposed length)
# ---------------------------------------------------------------------------


def compute_damage_per_rp_tent(
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

    Like compute_damage_per_rp but uses return_full=False to retain
    the 'coverage' and 'values' columns, then computes 'exposed_length'
    as sum(coverage where hazard > 0) for each feature.

    Returns:
        GeoDataFrame with damage columns per curve + 'exposed_length' column
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
        return_full=False,  # Keep coverage and values
    )

    result = result.to_crs(3035)  # Ensure consistent CRS for length calculations

    # Compute exposed_length from coverage and values
    # Lines: per-cell lengths (m); Polygons: per-cell areas (m²); Points: binary × area attr
    def _exposed_length(row):
        vals = row.get("values", [])
        covs = row.get("coverage", [])
        vals = np.array(vals) if not isinstance(vals, np.ndarray) else vals
        covs = np.array(covs) if not isinstance(covs, np.ndarray) else covs
        if len(vals) == 0 or len(covs) == 0:
            return 0.0
        geom = row.geometry if hasattr(row, 'geometry') else None
        if geom is not None and geom.geom_type in ("Polygon", "MultiPolygon"):
            if np.any(vals > 0):
                return float(geom.area)
            return 0.0
        if geom is not None and geom.geom_type in ("Point", "MultiPoint"):
            # Binary: 1 if any cell exposed, scaled by area attribute if available
            is_exposed = bool(np.any(vals > 0))
            if not is_exposed:
                return 0.0
            area_attr = row.get("area", None)
            if area_attr is not None and pd.notna(area_attr) and float(area_attr) > 0:
                return float(area_attr)
            return 1.0
        # LineString fallback
        exposed = float(np.sum(covs[vals > 0]))
        geom_size = geom.length if geom is not None else np.inf
        return min(exposed, geom_size)

    if "values" in result.columns and "coverage" in result.columns:
        result["exposed_length"] = result.apply(_exposed_length, axis=1)
    else:
        result["exposed_length"] = 0.0


    # Drop the heavy list columns to save memory
    result = result.drop(columns=["values", "coverage"], errors="ignore")

    return result


# ---------------------------------------------------------------------------
# TENT-specific exposure metric (uses coverage, not geometry length)
# ---------------------------------------------------------------------------


def compute_exposure_metric_tent(
    features: gpd.GeoDataFrame,
    hazard,
    reference_rp: int,
    hazard_value_col: str = "band_data",
    pga_threshold: float = 0.0,
) -> pd.Series:
    """
    Compute exposure using actual intersected length from VectorExposure coverage.

    Instead of returning the full geometry length for any exposed feature,
    this returns sum(coverage[i] where values[i] > threshold) — the actual
    length (in metres) of each edge that intersects the hazard raster.

    Args:
        features:       Exposure GeoDataFrame (EPSG:3035)
        hazard:         Hazard raster at the reference return period
        reference_rp:   Return period label (for logging only)
        hazard_value_col: Band name in hazard dataset
        pga_threshold:  Minimum hazard value to count as exposed (default 0)

    Returns:
        Series of exposure values indexed like features (0 where not exposed)
    """
    exposed_features, _, crs, cell_area = VectorExposure(
        hazard_file=hazard,
        feature_file=features,
        hazard_value_col=hazard_value_col,
        disable_progress=True,
    )

    exposed_features = exposed_features.to_crs(3035)

    if "values" not in exposed_features.columns or "coverage" not in exposed_features.columns:
        return pd.Series(0.0, index=features.index)

    def _exposed_length(row):
        vals = row.get("values", [])
        covs = row.get("coverage", [])
        vals = np.array(vals) if not isinstance(vals, np.ndarray) else vals
        covs = np.array(covs) if not isinstance(covs, np.ndarray) else covs
        if len(vals) == 0 or len(covs) == 0:
            return 0.0
        geom = row.geometry if hasattr(row, 'geometry') else None
        if geom is not None and geom.geom_type in ("Polygon", "MultiPolygon"):
            # Any cell exposed → return full polygon area (consistent with compute_exposure_metric)
            if np.any(vals > pga_threshold):
                return float(geom.area)
            return 0.0
        if geom is not None and geom.geom_type in ("Point", "MultiPoint"):
            is_exposed = bool(np.any(vals > pga_threshold))
            if not is_exposed:
                return 0.0
            area_attr = row.get("area", None)
            if area_attr is not None and pd.notna(area_attr) and float(area_attr) > 0:
                return float(area_attr)
            return 1.0
        exposed = float(np.sum(covs[vals > pga_threshold]))
        geom_size = geom.length if geom is not None else np.inf
        return min(exposed, geom_size)

    exposure_values = exposed_features.apply(_exposed_length, axis=1)

    return exposure_values.reindex(features.index, fill_value=0.0)


# ---------------------------------------------------------------------------
# Per-RP vulnerability ratio and exposure length
# ---------------------------------------------------------------------------


def _compute_feature_size(features: gpd.GeoDataFrame) -> pd.Series:
    """
    Per-feature size in the same unit basis as `maxdam` (length in metres
    for lines, area in m² for polygons, area attribute for area-bearing
    points, else 1 for point assets with no area attribute).

    Matches the geometry-type branching used elsewhere (e.g. asset_size
    in run_pipeline_tent.py, exp_col_prefix logic in attach_per_rp_columns).
    """
    geom_types = features.geometry.geom_type
    is_line = geom_types.isin(["LineString", "MultiLineString"])
    is_poly = geom_types.isin(["Polygon", "MultiPolygon"])
    is_point = ~is_line & ~is_poly

    size = pd.Series(1.0, index=features.index)
    if is_line.any():
        size[is_line] = features.geometry[is_line].length
    if is_poly.any():
        size[is_poly] = features.geometry[is_poly].area
    if is_point.any() and "area" in features.columns:
        area_vals = pd.to_numeric(features.loc[is_point, "area"], errors="coerce").fillna(0)
        has_area = area_vals > 0
        size.loc[is_point] = area_vals.where(has_area, 1.0)
    return size


def compute_ratio_and_exposure_per_rp(
    rp_result: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    maxdam: pd.DataFrame,
    object_col: str = "object_type",
) -> tuple[pd.Series, pd.Series]:
    """
    Compute vulnerability ratio and exposure length from an existing
    per-RP damage result.

    Vulnerability ratio:
        VectorScanner computes:
            damage = sum(interp(hazard, curve) * coverage) * maxdam_per_unit
        i.e. damage is already a *total* euro amount over the whole exposed
        length/area, while `maxdam` is a *per-unit* cost (€/m for lines,
        €/m² for polygons). Dividing damage straight by maxdam_per_unit
        therefore returns an exposed-length/area number (in metres/m²),
        not a 0-1 fraction — it saturates past 1 for any feature bigger
        than ~1 unit. To get the true curve-fraction ratio we must also
        divide out the feature's own size:
            ratio = damage / (maxdam_per_unit * feature_size)
        For assets with maxdam=0, size=0, or no damage, ratio=0.

    Exposure length:
        Uses the 'exposed_length' column from compute_damage_per_rp_tent,
        which is sum(coverage[i] where hazard_value[i] > 0) — the actual
        intersected length in metres, not the full geometry length.

    Args:
        rp_result:   GeoDataFrame from compute_damage_per_rp_tent + filter/summarise.
                     Must have 'damage_mean' and 'exposed_length' columns.
        features:    Original exposure GeoDataFrame (for full index)
        maxdam:      Maximum damage DataFrame with columns ['object_type', 'damage']
        object_col:  Column name for object type

    Returns:
        Tuple of (vuln_ratio, exposure_length), both Series indexed like features.
    """
    # Build a lookup: object_type -> maxdam value
    maxdam_lookup = dict(zip(maxdam["object_type"], maxdam["damage"]))

    # Align rp_result to features index
    aligned = rp_result.reindex(features.index)

    # Get per-unit maxdam per feature based on object_type
    feature_maxdam = features[object_col].map(maxdam_lookup).fillna(0).astype(float)

    # Total max damage for the *whole* feature = per-unit cost x feature size
    feature_size = _compute_feature_size(features)
    total_maxdam = feature_maxdam * feature_size

    # Vulnerability ratio = damage_mean / total_maxdam (where total_maxdam > 0)
    damage_mean = aligned["damage_mean"].fillna(0).astype(float)
    vuln_ratio = pd.Series(0.0, index=features.index)
    valid = total_maxdam > 0
    vuln_ratio[valid] = damage_mean[valid] / total_maxdam[valid]
    # Clip to [0, 1] -- ratio should not exceed 1
    vuln_ratio = vuln_ratio.clip(0, 1)

    # Exposure length from VectorScanner coverage (actual intersected length)
    if "exposed_length" in aligned.columns:
        exposure_length = aligned["exposed_length"].fillna(0).astype(float)
        exposure_length = exposure_length.reindex(features.index, fill_value=0.0)
    else:
        # Fallback: geometry size where damage > 0 (used e.g. for coastal tile-aggregated results)
        exposed_mask = damage_mean > 0
        exposure_length = pd.Series(0.0, index=features.index)
        if exposed_mask.any():
            geoms = features.geometry[exposed_mask]
            is_poly = geoms.geom_type.isin(["Polygon", "MultiPolygon"])
            is_point = ~geoms.geom_type.isin(
                ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
            )
            is_line = ~is_poly & ~is_point
            sizes = np.where(is_poly, geoms.area, np.where(is_line, geoms.length, 1.0))
            exposure_length[exposed_mask] = sizes

    return vuln_ratio, exposure_length


def attach_per_rp_columns(
    features: gpd.GeoDataFrame,
    rp_results: dict[int, gpd.GeoDataFrame],
    maxdam: pd.DataFrame,
    hazard_name: str,
    object_col: str = "object_type",
) -> gpd.GeoDataFrame:
    """
    Attach per-RP vulnerability ratio and exposure length columns to features.

    Creates columns:
        vuln_ratio_{hazard_name}_RP{rp}     -- mean vulnerability ratio (0-1)
        exposure_length_{hazard_name}_RP{rp} -- exposed length in metres
                                               (actual intersected length from coverage)

    Args:
        features:     Exposure GeoDataFrame (modified in-place)
        rp_results:   Dict of {return_period: GeoDataFrame} from damage calculation
        maxdam:       Maximum damage DataFrame
        hazard_name:  Hazard identifier for column naming (e.g. 'river', 'windstorm')
        object_col:   Column name for object type

    Returns:
        features GeoDataFrame with new columns added
    """

    features = features.to_crs(3035)  # Avoid modifying original in-place

    # Determine exposure column prefix from geometry type and area attribute
    first_geom = features.geometry.iloc[0] if len(features) > 0 else None
    is_poly_asset = (
        first_geom is not None
        and first_geom.geom_type in ("Polygon", "MultiPolygon")
    )
    is_point_with_area = (
        first_geom is not None
        and first_geom.geom_type in ("Point", "MultiPoint")
        and "area" in features.columns
        and (pd.to_numeric(features["area"], errors="coerce") > 0).any()
    )
    is_point_no_area = (
        first_geom is not None
        and first_geom.geom_type in ("Point", "MultiPoint")
        and not is_point_with_area
    )
    if is_poly_asset or is_point_with_area:
        exp_col_prefix = "exposure_area_"
    elif is_point_no_area:
        exp_col_prefix = "exposure_count_"  # binary: 1=exposed node, 0=not
    else:
        exp_col_prefix = "exposure_length_"

    for rp, rp_result in sorted(rp_results.items()):
        ratio, exp_len = compute_ratio_and_exposure_per_rp(
            rp_result=rp_result,
            features=features,
            maxdam=maxdam,
            object_col=object_col,
        )
        features[f"vuln_ratio_{hazard_name}_RP{rp}"] = ratio.values
        features[f"{exp_col_prefix}{hazard_name}_RP{rp}"] = exp_len.values

    return features
