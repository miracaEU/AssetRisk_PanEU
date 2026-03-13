"""
exposure_heat.py

Heat exposure assessment module.

Processes flat NetCDF files:
  06_hot_days-{type}-monthly-{threshold}deg-{scenario}-{model}-...-v1.0.nc

Key efficiency improvements over original heat_exposure_analysis.py:
  - Opens each NetCDF ONCE per model, loops all windows inside
  - Averages across years FIRST → single 2D grid per month, then samples
    features (vs original: sample per individual timestep, average later)
  - Vectorised coordinate extraction: all centroid lat/lon passed to
    xr.sel() in one batch call using advanced indexing — no iterrows loop
  - groupby().agg() for multi-model aggregation (vs wide pivot merge)
  - Spatial clip applied once per file before any window slicing

Output columns (yearly totals = sum of warm-season months):
  heat_{threshold}_{scenario}_{window}_mean
  heat_{threshold}_{scenario}_{window}_min / _max   (multi-model future)
  heat_{threshold}_{scenario}_rel_change_recent_to_{window}_mean / _min / _max
  heat_{threshold}_{scenario}_abs_change_recent_to_{window}_mean / _min / _max
"""

import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_WINDOWS = {
    "recent":         (1990, 2016),
    "near_future":    (2021, 2040),
    "mid_future":     (2041, 2060),
    "far_future":     (2061, 2080),
    "distant_future": (2081, 2100),
}

WARM_MONTHS = [4, 5, 6, 7, 8, 9, 10]

MONTH_NAMES = {
    4: "April", 5: "May", 6: "June", 7: "July",
    8: "August", 9: "September", 10: "October",
    0: "Yearly",
}

def _detect_heat_data_var(ds: xr.Dataset) -> str:
    """Detect heat data variable — projections use tasAdjust_NON_CDM, reanalysis uses t2m."""
    for candidate in ("tasAdjust_NON_CDM", "t2m"):
        if candidate in ds.data_vars:
            return candidate
    # Fall back to first non-coordinate variable
    skip = {"time", "lat", "lon", "height", "realization"}
    candidates = [v for v in ds.data_vars if v not in skip]
    if not candidates:
        raise ValueError(f"No data variable found. Available: {list(ds.data_vars)}")
    return candidates[0]

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _parse_heat_filename(stem: str) -> dict | None:
    threshold_match = re.search(r"-(\d+)deg-", stem)
    if not threshold_match:
        return None
    threshold = f"{threshold_match.group(1)}C"

    if "reanalysis" in stem:
        return {"threshold": threshold, "file_type": "reanalysis",
                "scenario": "historical", "model": "reanalysis"}
    elif "projections" in stem:
        scenario_match = re.search(r"-(rcp_[\d_]+)-", stem)
        if not scenario_match:
            return None
        scenario = scenario_match.group(1)
        model_match = re.search(
            rf"-{re.escape(scenario)}-(.+?)-(?:r\d+i\d+p\d+|grid)", stem
        )
        model = model_match.group(1) if model_match else "unknown"
        return {"threshold": threshold, "file_type": "projections",
                "scenario": scenario, "model": model}
    return None


def discover_heat_files(
    heat_dir: Union[str, Path],
) -> dict[str, dict[str, list[tuple[str, Path]]]]:
    """Returns: {threshold: {scenario: [(model, path), ...]}}"""
    heat_dir = Path(heat_dir)
    result: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for nc_file in sorted(heat_dir.glob("*.nc")):
        meta = _parse_heat_filename(nc_file.stem)
        if meta is None:
            continue
        result[meta["threshold"]][meta["scenario"]].append((meta["model"], nc_file))
    return {k: dict(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Vectorised sampling helper
# ---------------------------------------------------------------------------

def _sample_features_vectorised(
    monthly_mean: xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Sample a 2D (lat, lon) DataArray at N feature centroid locations
    using vectorised xr.sel with advanced indexing.

    Returns array of shape (N,) with float values (NaN → 0.0).
    """
    lat_da = xr.DataArray(lats, dims="points")
    lon_da = xr.DataArray(lons, dims="points")
    vals = monthly_mean.sel(lat=lat_da, lon=lon_da, method="nearest").values
    vals = np.where(np.isnan(vals), 0.0, vals)
    return vals.astype(float)


# ---------------------------------------------------------------------------
# Core: open file once, compute all windows
# ---------------------------------------------------------------------------

def _assess_all_windows(
    nc_path: Path,
    features_4326: gpd.GeoDataFrame,
    applicable_windows: dict[str, tuple[int, int]],
    months: list[int],
    data_var: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Open one NetCDF file and compute monthly-mean exposure for all applicable
    time windows. File is opened and spatially clipped once.

    Returns: {window_name: DataFrame(osm_id, month, avg_days_{window_name})}
    """
    ds = xr.open_dataset(nc_path)
    data_var = _detect_heat_data_var(ds)

    # Spatial clip once for this file (reduces all subsequent ops)
    bounds = features_4326.total_bounds   # minx, miny, maxx, maxy
    ds_clip = ds.sel(
        lat=slice(bounds[1] - 0.5, bounds[3] + 0.5),
        lon=slice(bounds[0] - 0.5, bounds[2] + 0.5),
    )

    # Pre-extract centroid coordinates as numpy arrays — done once per file
    # Compute centroids in EPSG:3035 (metres) then reproject to WGS84 for sampling
    geoms_3035 = features_4326.to_crs(3035).geometry
    centroids_3035 = geoms_3035.where(geoms_3035.geom_type == "Point", geoms_3035.centroid)
    centroids_4326 = centroids_3035.to_crs(4326)
    lats = centroids_4326.y.values
    lons = centroids_4326.x.values
    osm_ids = features_4326["osm_id"].values

    years_in_file = pd.DatetimeIndex(ds.time.values).year
    results: dict[str, pd.DataFrame] = {}

    for window_name, (yr_start, yr_end) in applicable_windows.items():
        if yr_start > years_in_file.max() or yr_end < years_in_file.min():
            continue

        ds_window = ds_clip.sel(
            time=slice(f"{yr_start}-01-01", f"{yr_end}-12-31")
        )
        if ds_window.sizes["time"] == 0:
            continue

        # Build (n_features × n_months) result matrix
        n = len(osm_ids)
        month_vals = np.zeros((n, len(months)), dtype=float)

        for j, month in enumerate(months):
            month_mask = ds_window.time.dt.month == month
            month_data = ds_window[data_var].sel(time=month_mask)
            if month_data.sizes["time"] == 0:
                continue
            # Average across years → single 2D grid, then sample all features at once
            monthly_mean = month_data.mean(dim="time").compute()
            month_vals[:, j] = _sample_features_vectorised(monthly_mean, lats, lons)

        # Build long-form DataFrame
        records = []
        for j, month in enumerate(months):
            df_month = pd.DataFrame({
                "osm_id": osm_ids,
                "month": month,
                f"avg_days_{window_name}": month_vals[:, j],
            })
            records.append(df_month)

        results[window_name] = pd.concat(records, ignore_index=True)

    ds.close()
    return results


# ---------------------------------------------------------------------------
# Aggregation across models
# ---------------------------------------------------------------------------

def _aggregate_models(
    window_dfs: list[pd.DataFrame],
    window_name: str,
) -> pd.DataFrame:
    """
    Aggregate monthly exposure across climate models.
    Single model → mean only.  Multiple → mean/min/max.
    Appends Yearly totals row per osm_id.
    """
    col = f"avg_days_{window_name}"

    if len(window_dfs) == 1:
        df = window_dfs[0].copy().rename(columns={col: f"{col}_mean"})
        stat_cols = [f"{col}_mean"]
    else:
        combined = pd.concat(window_dfs, ignore_index=True)
        agg = (
            combined.groupby(["osm_id", "month"])[col]
            .agg(["mean", "min", "max"])
            .reset_index()
        )
        agg.columns = ["osm_id", "month", f"{col}_mean", f"{col}_min", f"{col}_max"]
        df = agg
        stat_cols = [f"{col}_mean", f"{col}_min", f"{col}_max"]

    df["month_name"] = df["month"].map(MONTH_NAMES)
    yearly = df.groupby("osm_id")[stat_cols].sum().reset_index()
    yearly["month"] = 0
    yearly["month_name"] = "Yearly"
    return pd.concat([df, yearly], ignore_index=True)


# ---------------------------------------------------------------------------
# Relative change calculation
# ---------------------------------------------------------------------------

def _calculate_relative_changes(
    recent_df: pd.DataFrame,
    future_df: pd.DataFrame,
    future_window: str,
) -> pd.DataFrame:
    """
    Compute abs and rel change between recent baseline and a future window.
    Matches original calculate_relative_change_safe logic exactly.
    """
    merged = pd.merge(
        recent_df, future_df,
        on=["osm_id", "month", "month_name"],
        suffixes=("_baseline", "_future"),
    )

    stats_to_process = ["mean"]
    if f"avg_days_{future_window}_min" in merged.columns:
        stats_to_process += ["min", "max"]

    for stat in stats_to_process:
        base_col = "avg_days_recent_mean"
        fut_col  = f"avg_days_{future_window}_{stat}"
        if base_col not in merged.columns or fut_col not in merged.columns:
            continue

        abs_col = f"abs_change_recent_to_{future_window}_{stat}"
        rel_col = f"rel_change_recent_to_{future_window}_{stat}"

        merged[abs_col] = merged[fut_col] - merged[base_col]
        merged[rel_col] = np.nan

        normal   = merged[base_col] > 0
        zero_pos = (merged[base_col] == 0) & (merged[fut_col] > 0)
        zero_zer = (merged[base_col] == 0) & (merged[fut_col] == 0)

        merged.loc[normal,   rel_col] = merged.loc[normal, abs_col] / merged.loc[normal, base_col]
        merged.loc[zero_pos, rel_col] = 10.0
        merged.loc[zero_zer, rel_col] = 0.0

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assess_heat(
    features: gpd.GeoDataFrame,
    heat_dir: Union[str, Path],
    asset_type: str,
    thresholds: list[str] | None = None,
    time_windows: dict[str, tuple[int, int]] | None = None,
    warm_months: list[int] | None = None,
) -> dict[str, pd.Series]:
    """
    Assess heat exposure for a pre-loaded infrastructure GeoDataFrame.

    Args:
        features:      Infrastructure GeoDataFrame (any CRS)
        heat_dir:      Directory containing heat NetCDF files
        asset_type:    Asset type name (for logging)
        thresholds:    Threshold keys e.g. ['30C','35C'] (None = all found)
        time_windows:  Override default year ranges
        warm_months:   Override default warm-season months

    Returns:
        Dict: output column name → pd.Series indexed by osm_id (yearly totals)
        Includes both exposure values and relative/absolute changes.
    """
    t0 = time.time()
    print(f"[heat] Starting for {asset_type} ({len(features)} features)")

    if time_windows is None:
        time_windows = TIME_WINDOWS
    if warm_months is None:
        warm_months = WARM_MONTHS

    features_4326 = features.to_crs(4326)

    all_files = discover_heat_files(heat_dir)
    if not all_files:
        print(f"[heat] No files found in {heat_dir}")
        return {}
    if thresholds:
        all_files = {k: v for k, v in all_files.items() if k in thresholds}

    print(f"[heat] Thresholds: {list(all_files.keys())}")
    all_columns: dict[str, pd.Series] = {}

    for threshold, scenario_files in all_files.items():
        threshold_clean = threshold.replace("°", "")
        hist_recent: pd.DataFrame | None = None

        # --- Pass 1: exposure values per scenario/window ---
        scenario_aggregated: dict[str, dict[str, pd.DataFrame]] = {}

        for scenario, model_paths in scenario_files.items():
            scenario_clean = scenario.lower().replace("-", "_")
            print(f"[heat]  {threshold_clean} | {scenario_clean} ({len(model_paths)} model(s))")

            if scenario == "historical":
                applicable = {"recent": time_windows["recent"]}
            else:
                applicable = {k: v for k, v in time_windows.items() if k != "recent"}

            window_model_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)

            for model, nc_path in model_paths:
                print(f"[heat]    {model}")
                try:
                    model_results = _assess_all_windows(
                        nc_path, features_4326, applicable, warm_months
                    )
                    for window_name, df in model_results.items():
                        window_model_dfs[window_name].append(df)
                except Exception as e:
                    print(f"[heat]    WARNING {model}: {e}")

            scenario_aggregated[scenario_clean] = {}
            for window_name, window_dfs in window_model_dfs.items():
                if not window_dfs:
                    continue
                agg = _aggregate_models(window_dfs, window_name)
                scenario_aggregated[scenario_clean][window_name] = agg

                yearly = agg[agg["month_name"] == "Yearly"].set_index("osm_id")
                for col in yearly.columns:
                    if not col.startswith("avg_days"):
                        continue
                    stat = col.split("_")[-1]
                    out_col = f"heat_{threshold_clean}_{scenario_clean}_{window_name}_{stat}"
                    all_columns[out_col] = yearly[col].rename(out_col)

            if scenario == "historical" and "recent" in scenario_aggregated.get("historical", {}):
                hist_recent = scenario_aggregated["historical"]["recent"]

        # --- Pass 2: relative changes ---
        if hist_recent is None:
            print(f"[heat] No historical/recent data for {threshold_clean} — skipping relative changes")
            continue

        for scenario_clean, windows in scenario_aggregated.items():
            if scenario_clean == "historical":
                continue
            for future_window, future_agg in windows.items():
                changes = _calculate_relative_changes(hist_recent, future_agg, future_window)
                yearly_ch = changes[changes["month_name"] == "Yearly"].set_index("osm_id")
                for col in yearly_ch.columns:
                    if not (col.startswith("abs_change") or col.startswith("rel_change")):
                        continue
                    out_col = f"heat_{threshold_clean}_{scenario_clean}_{col}"
                    all_columns[out_col] = yearly_ch[col].rename(out_col)

    elapsed = time.time() - t0
    print(f"[heat] Done in {elapsed:.1f}s — {len(all_columns)} columns")
    return all_columns
