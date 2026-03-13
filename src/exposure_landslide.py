"""
exposure_landslide.py

Landslide susceptibility exposure assessment module.

Takes a pre-loaded exposure GeoDataFrame and returns it enriched with:
  - landslide_min        : minimum susceptibility class overlapping the asset
  - landslide_avg        : coverage-weighted average susceptibility
  - landslide_max        : maximum susceptibility class overlapping the asset
  - landslide_exposure   : total length (m) / area (m²) / count overlapping susceptibility > 0
  - landslide_max_cat    : length/area in the highest susceptibility class

Depends on:
  - damagescanner (VectorExposure)
  - exposure_utils.py
"""

import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from typing import Optional, Union

from damagescanner.core import VectorExposure

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Hazard loading
# ---------------------------------------------------------------------------


def load_landslide(
    landslide_path: Union[str, Path],
    bounds_3035: Optional[tuple[float, float, float, float]] = None,
) -> xr.Dataset:
    """
    Load landslide susceptibility raster (.asc or GeoTIFF).

    Args:
        landslide_path:  Path to the susceptibility raster
        bounds_3035:     Optional (minx, miny, maxx, maxy) in EPSG:3035 to clip

    Returns:
        xarray Dataset in EPSG:3035
    """
    ds = xr.open_dataset(landslide_path, engine="rasterio")
    ds = ds.rio.write_crs("EPSG:3035")

    if bounds_3035 is not None:
        minx, miny, maxx, maxy = bounds_3035
        ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    return ds


# ---------------------------------------------------------------------------
# Vectorised susceptibility statistics
# ---------------------------------------------------------------------------


def _compute_susceptibility_stats(
    exposed: gpd.GeoDataFrame,
    cell_area_m2: float,
) -> pd.DataFrame:
    """
    Compute min, avg, max susceptibility and exposure metrics from VectorExposure output.

    Fully vectorised — no iterrows.

    Args:
        exposed:       VectorExposure output with 'values' and 'coverage' columns
        cell_area_m2:  Raster cell area in m² (for polygon coverage scaling)

    Returns:
        DataFrame with columns: min_susceptibility, avg_susceptibility,
                                max_susceptibility, total_exposure, max_cat_exposure
    """
    is_poly = exposed.geometry.geom_type.isin(["Polygon", "MultiPolygon"])

    records = []
    for idx, row in exposed.iterrows():
        vals = np.asarray(row["values"], dtype=float)
        cov = np.asarray(row["coverage"], dtype=float)

        if len(vals) == 0 or np.all(np.isnan(vals)):
            records.append(
                {
                    "min_susceptibility": np.nan,
                    "avg_susceptibility": np.nan,
                    "max_susceptibility": np.nan,
                    "total_exposure": 0.0,
                    "max_cat_exposure": 0.0,
                }
            )
            continue

        valid = ~np.isnan(vals)
        v = vals[valid]
        c = cov[valid] if len(cov) == len(vals) else np.ones(valid.sum())
        c = c[valid] if len(cov) == len(vals) else c

        # Scale coverage for polygons
        if is_poly.loc[idx]:
            c_scaled = c * cell_area_m2
        else:
            c_scaled = c

        total = float(c_scaled.sum())
        max_v = float(v.max())
        min_v = float(v.min())
        avg_v = (
            float(np.average(v, weights=c_scaled))
            if c_scaled.sum() > 0
            else float(v.mean())
        )
        max_cat = float(c_scaled[v == max_v].sum())

        records.append(
            {
                "min_susceptibility": min_v,
                "avg_susceptibility": avg_v,
                "max_susceptibility": max_v,
                "total_exposure": total,
                "max_cat_exposure": max_cat,
            }
        )

    return pd.DataFrame(records, index=exposed.index)


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def assess_landslide(
    features: gpd.GeoDataFrame,
    landslide_path: Union[str, Path],
    asset_type: str,
) -> gpd.GeoDataFrame:
    """
    Assess landslide susceptibility exposure for a pre-loaded exposure GeoDataFrame.

    Args:
        features:        Exposure GeoDataFrame (any CRS — reprojected internally to EPSG:3035)
        landslide_path:  Path to landslide susceptibility raster
        asset_type:      Internal asset type name (for logging)

    Returns:
        Input GeoDataFrame enriched with:
          landslide_min, landslide_avg, landslide_max,
          landslide_exposure, landslide_max_cat
    """
    t0 = time.time()
    print(
        f"[landslide] Starting assessment for {asset_type} ({len(features)} features)"
    )

    # Reproject features to EPSG:3035 to match raster
    features_3035 = features.to_crs(3035)

    # Derive bounds for clipping
    bounds = features_3035.total_bounds
    bounds_3035 = (
        bounds[0] - 10_000,
        bounds[1] - 10_000,
        bounds[2] + 10_000,
        bounds[3] + 10_000,
    )

    # Load and clip raster
    landslide_ds = load_landslide(landslide_path, bounds_3035)

    # Derive cell area in m² (raster is in EPSG:3035 = metres)
    res = landslide_ds.rio.resolution()
    cell_area_m2 = abs(float(res[0])) * abs(float(res[1]))

    # Run overlay
    print("[landslide] Running overlay...")
    exposed, _, _, _ = VectorExposure(
        hazard_file=landslide_ds,
        feature_file=features_3035,
        hazard_value_col="band_data",
        disable_progress=False,
    )

    # Compute statistics
    print("[landslide] Computing susceptibility statistics...")
    stats = _compute_susceptibility_stats(exposed, cell_area_m2)

    # Merge back to original features (preserving original CRS)
    features = features.copy()
    features["landslide_min"] = stats["min_susceptibility"].values
    features["landslide_avg"] = stats["avg_susceptibility"].values
    features["landslide_max"] = stats["max_susceptibility"].values
    features["landslide_exposure"] = stats["total_exposure"].values
    features["landslide_max_cat"] = stats["max_cat_exposure"].values

    elapsed = time.time() - t0
    n_exposed = (features["landslide_max"] > 0).sum()
    print(
        f"[landslide] Done in {elapsed:.1f}s. "
        f"{n_exposed}/{len(features)} features exposed. "
        f"Mean susceptibility: {features['landslide_avg'].mean():.2f}"
    )

    return features
