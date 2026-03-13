"""
exposure_wildfire.py

Wildfire fire-danger exposure assessment module.

Processes flat NetCDF files:
  22_days_with_high_fire_danger-{type}-monthly-{scenario}-{model}-...-v1.0.nc

Key efficiency improvements over original wildfire_exposure.py:
  - Opens each NetCDF ONCE per model, loops all windows inside
  - Averages across years FIRST → single 2D grid per month, then samples
    features (vs original: sample per individual timestep, average later)
  - Vectorised coordinate extraction: all centroid lat/lon passed to
    xr.sel() in one batch call using advanced indexing — no iterrows loop
  - groupby().agg() for multi-model aggregation (vs wide pivot merge)
  - Spatial clip applied once per file before any window slicing

Output columns (yearly totals = sum of fire-season months):
  wildfire_{scenario}_{window}_mean
  wildfire_{scenario}_{window}_min / _max   (multi-model future)
  wildfire_{scenario}_rel_change_recent_to_{window}_mean / _min / _max
  wildfire_{scenario}_abs_change_recent_to_{window}_mean / _min / _max
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
    "recent":      (1990, 2016),
    "near_future": (2021, 2040),
    "mid_future":  (2041, 2060),
    "far_future":  (2061, 2080),
}

FIRE_MONTHS = [5, 6, 7, 8, 9]

MONTH_NAMES = {
    5: "May", 6: "June", 7: "July", 8: "August", 9: "September",
    0: "Yearly",
}

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _parse_wildfire_filename(stem: str) -> dict | None:
    if "reanalysis" in stem:
        return {"file_type": "reanalysis", "scenario": "historical", "model": "reanalysis"}
    elif "projections" in stem:
        scenario_match = re.search(r"-(rcp_[\d_]+)-", stem)
        if not scenario_match:
            return None
        scenario = scenario_match.group(1)
        model_match = re.search(rf"-{re.escape(scenario)}-(.+?)-grid-", stem)
        model = model_match.group(1) if model_match else "unknown"
        return {"file_type": "projections", "scenario": scenario, "model": model}
    return None


def discover_wildfire_files(
    wildfire_dir: Union[str, Path],
) -> dict[str, list[tuple[str, Path]]]:
    """Returns: {scenario: [(model, path), ...]}"""
    wildfire_dir = Path(wildfire_dir)
    result: dict[str, list] = defaultdict(list)
    for nc_file in sorted(wildfire_dir.glob("*.nc")):
        meta = _parse_wildfire_filename(nc_file.stem)
        if meta is None:
            continue
        result[meta["scenario"]].append((meta["model"], nc_file))
    return dict(result)


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

def _detect_data_var(ds: xr.Dataset) -> str:
    """Detect wildfire data variable (fwi, data, or first available)."""
    if "fwi" in ds.data_vars:
        return "fwi"
    if "data" in ds.data_vars:
        return "data"
    candidates = [v for v in ds.data_vars if v not in ("time", "lat", "lon", "height")]
    if not candidates:
        raise ValueError("No data variable found in wildfire NetCDF")
    return candidates[0]


def _assess_all_windows(
    nc_path: Path,
    features_4326: gpd.GeoDataFrame,
    applicable_windows: dict[str, tuple[int, int]],
    months: list[int],
) -> dict[str, pd.DataFrame]:
    """
    Open one NetCDF file and compute monthly-mean exposure for all applicable
    time windows. File is opened and spatially clipped once.

    Returns: {window_name: DataFrame(osm_id, month, avg_days_{window_name})}
    """
    ds = xr.open_dataset(nc_path)
    data_var = _detect_data_var(ds)

    # Spatial clip once
    bounds = features_4326.total_bounds
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

        records = []
        for j, month in enumerate(months):
            records.append(pd.DataFrame({
                "osm_id": osm_ids,
                "month": month,
                f"avg_days_{window_name}": month_vals[:, j],
            }))

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

def assess_wildfire(
    features: gpd.GeoDataFrame,
    wildfire_dir: Union[str, Path],
    asset_type: str,
    time_windows: dict[str, tuple[int, int]] | None = None,
    fire_months: list[int] | None = None,
) -> dict[str, pd.Series]:
    """
    Assess wildfire fire-danger exposure for a pre-loaded infrastructure GeoDataFrame.

    Args:
        features:      Infrastructure GeoDataFrame (any CRS)
        wildfire_dir:  Directory containing wildfire NetCDF files
        asset_type:    Asset type name (for logging)
        time_windows:  Override default year ranges
        fire_months:   Override default fire-season months

    Returns:
        Dict: output column name → pd.Series indexed by osm_id (yearly totals)
        Includes both exposure values and relative/absolute changes.
    """
    t0 = time.time()
    print(f"[wildfire] Starting for {asset_type} ({len(features)} features)")

    if time_windows is None:
        time_windows = TIME_WINDOWS
    if fire_months is None:
        fire_months = FIRE_MONTHS

    features_4326 = features.to_crs(4326)

    all_files = discover_wildfire_files(wildfire_dir)
    if not all_files:
        print(f"[wildfire] No files found in {wildfire_dir}")
        return {}

    print(f"[wildfire] Scenarios: {list(all_files.keys())}")
    all_columns: dict[str, pd.Series] = {}
    scenario_aggregated: dict[str, dict[str, pd.DataFrame]] = {}
    hist_recent: pd.DataFrame | None = None

    # --- Pass 1: exposure values ---
    for scenario, model_paths in all_files.items():
        scenario_clean = scenario.lower().replace("-", "_")
        print(f"[wildfire]   {scenario_clean} ({len(model_paths)} model(s))")

        if scenario == "historical":
            applicable = {"recent": time_windows["recent"]}
        else:
            applicable = {k: v for k, v in time_windows.items() if k != "recent"}

        window_model_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)

        for model, nc_path in model_paths:
            print(f"[wildfire]     {model}")
            try:
                model_results = _assess_all_windows(
                    nc_path, features_4326, applicable, fire_months
                )
                for window_name, df in model_results.items():
                    window_model_dfs[window_name].append(df)
            except Exception as e:
                print(f"[wildfire]     WARNING {model}: {e}")

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
                out_col = f"wildfire_{scenario_clean}_{window_name}_{stat}"
                all_columns[out_col] = yearly[col].rename(out_col)

        if scenario == "historical" and "recent" in scenario_aggregated.get("historical", {}):
            hist_recent = scenario_aggregated["historical"]["recent"]

    # --- Pass 2: relative changes ---
    if hist_recent is None:
        print("[wildfire] No historical/recent data — skipping relative changes")
    else:
        for scenario_clean, windows in scenario_aggregated.items():
            if scenario_clean == "historical":
                continue
            for future_window, future_agg in windows.items():
                changes = _calculate_relative_changes(hist_recent, future_agg, future_window)
                yearly_ch = changes[changes["month_name"] == "Yearly"].set_index("osm_id")
                for col in yearly_ch.columns:
                    if not (col.startswith("abs_change") or col.startswith("rel_change")):
                        continue
                    out_col = f"wildfire_{scenario_clean}_{col}"
                    all_columns[out_col] = yearly_ch[col].rename(out_col)

    elapsed = time.time() - t0
    print(f"[wildfire] Done in {elapsed:.1f}s — {len(all_columns)} columns")
    return all_columns
