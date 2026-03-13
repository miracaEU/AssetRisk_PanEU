"""
hazard_coastal.py

Coastal flood risk assessment module.

Key design decisions:
  - Tiles are streamed one at a time from STAC and discarded immediately after
    damage extraction to avoid out-of-memory errors.
  - Damage is aggregated by (osm_id, LAU) — not osm_id alone — because the
    same osm_id can span multiple LAUs representing physically distinct segments.
  - All 5 scenarios (2010 baseline + 4 future) are processed in one pass.
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import shapely
from pathlib import Path
from typing import Optional, Union, Generator

import pystac_client
from pystac.extensions.projection import ProjectionExtension

from risk_integration import (
    collect_ead_per_asset,
    compute_exposure_metric,
    compute_damage_per_rp,
)
from hazard_river import (
    prepare_flood_curves,
    filter_curve_results,
    RIVER_HAZARD_COL,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def _worker_init():
    import sys

    sys.excepthook = lambda *args: None


try:
    from pystac_client.warnings import NoConformsTo, FallbackToPystac

    warnings.filterwarnings("ignore", category=NoConformsTo)
    warnings.filterwarnings("ignore", category=FallbackToPystac)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COASTAL_STAC_URL = "https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json"
COASTAL_COLLECTION = "cfhp_all"
COASTAL_RETURN_PERIODS = [10, 25, 50, 100, 250, 500, 1000]
COASTAL_EXPOSURE_RP = 100

# (time_horizon, climate_scenario) → output column label
COASTAL_SCENARIOS = {
    ("2010", "None"): "EAD_coastal",
    ("2050", "SSP245"): "EAD_coastal_2050_SSP245",
    ("2050", "SSP585"): "EAD_coastal_2050_SSP585",
    ("2100", "SSP245"): "EAD_coastal_2100_SSP245",
    ("2100", "SSP585"): "EAD_coastal_2100_SSP585",
}

# Unique key for aggregating damage — osm_id alone is insufficient
# because the same OSM way can cross LAU boundaries
_AGG_KEY = ["osm_id", "LAU"]


# ---------------------------------------------------------------------------
# Tile reading
# ---------------------------------------------------------------------------


def _read_tile(href: str) -> Optional[xr.Dataset]:
    """Open a single GeoTIFF tile as an xarray Dataset. Returns None on failure."""
    import rasterio
    import rioxarray  # noqa — registers .rio accessor

    try:
        with rasterio.open(href) as src:
            data = src.read(1).astype(np.float32)
            xs = np.linspace(
                src.bounds.left, src.bounds.right, src.width, endpoint=False
            )
            ys = np.linspace(
                src.bounds.top, src.bounds.bottom, src.height, endpoint=False
            )
            crs = src.crs.to_string() if src.crs else "EPSG:4326"

        da = xr.DataArray(
            data[np.newaxis],
            dims=("band", "y", "x"),
            coords={"band": [1], "x": xs, "y": ys, "spatial_ref": 0},
        )
        ds = xr.Dataset({"band_data": da})
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        ds = ds.rio.write_crs(crs, inplace=False)
        return ds
    except Exception as e:
        print(f"  [coastal] Warning: could not read {href}: {e}")
        return None


# ---------------------------------------------------------------------------
# STAC tile streaming — yields (rp, tile) one at a time
# ---------------------------------------------------------------------------


def stream_coastal_tiles(
    features: gpd.GeoDataFrame,
    time_horizon: str,
    climate_scenario: str,
    stac_catalog_url: str = COASTAL_STAC_URL,
    collection_id: str = COASTAL_COLLECTION,
) -> Generator[tuple[int, xr.Dataset], None, None]:
    """
    Generator that yields (return_period, tile_dataset) one tile at a time.

    Tiles are only loaded when the generator is iterated — caller should
    process and discard each tile before requesting the next.

    Args:
        features:         Exposure GeoDataFrame (any CRS)
        time_horizon:     '2010', '2050', or '2100'
        climate_scenario: 'None', 'SSP245', or 'SSP585'
    """
    try:
        catalog = pystac_client.Client.open(stac_catalog_url)
        collection = catalog.get_child(id=collection_id)
    except Exception as e:
        print(f"[coastal] Cannot connect to STAC: {e}")
        return

    features_3035 = features.to_crs(3035)
    feature_tree = shapely.STRtree(features_3035.geometry)

    for item in collection.get_items():
        name = "_".join(item.id.split("\\")).split(".")[0]

        # Filter by horizon / scenario / defence type
        if "static" in name:
            continue
        if time_horizon not in name:
            continue
        if climate_scenario not in name:
            continue
        if "LOW_DEFENDED" not in name:
            continue

        # Extract return period
        rp = None
        for part in name.split("_")[:-1]:
            try:
                rp = int(part)
                break
            except ValueError:
                continue
        if rp is None:
            continue

        for i, asset_key in enumerate(item.assets):
            if i == 0:
                continue  # skip metadata

            asset = item.assets[asset_key]
            try:
                proj = ProjectionExtension.ext(asset)
                [ring] = proj.geometry["coordinates"]
                tile_geom = shapely.Polygon(ring)
            except Exception:
                continue

            if len(feature_tree.query(tile_geom)) == 0:
                continue  # tile doesn't intersect any features

            tile = _read_tile(asset.href)
            if tile is None:
                continue
            if float(tile.band_data.max()) == 0:
                del tile
                continue

            yield rp, tile
            del tile  # discard raster immediately after caller processes it
            gc.collect()


# ---------------------------------------------------------------------------
# Damage extraction from a single tile
# ---------------------------------------------------------------------------


def _damage_from_tile(
    tile: xr.Dataset,
    features: gpd.GeoDataFrame,
    damage_curves: pd.DataFrame,
    multi_curves: dict,
    maxdam: pd.DataFrame,
    exclusions: dict,
    feature_bounds_3035: tuple,
) -> Optional[pd.DataFrame]:
    """
    Extract damage from one tile. Returns a small DataFrame with columns:
      osm_id, LAU, damage_mean, damage_min, damage_max
    or None if no damage.
    """
    # Clip tile to feature bounds to minimise data volume
    try:
        tile_clipped = tile.rio.clip_box(*feature_bounds_3035)
    except Exception:
        tile_clipped = tile

    if float(tile_clipped.band_data.max()) == 0:
        return None

    try:
        result = compute_damage_per_rp(
            features=features,
            hazard=tile_clipped,
            curve_path=damage_curves,
            maxdam=maxdam,
            asset_type=None,
            multi_curves=multi_curves,
            hazard_value_col=RIVER_HAZARD_COL,
        )
    except Exception as e:
        print(f"  [coastal] Warning: damage calc failed: {e}")
        return None
    finally:
        del tile_clipped

    if result is None or result.empty:
        return None

    result = filter_curve_results(result, multi_curves, exclusions)

    curve_cols = [c for c in multi_curves.keys() if c in result.columns]
    if not curve_cols:
        return None

    out = pd.DataFrame(index=result.index)
    out["osm_id"] = result.get("osm_id", result.index)
    out["LAU"] = result.get("LAU", np.nan)
    out["damage_mean"] = result[curve_cols].mean(axis=1, skipna=True).fillna(0)
    out["damage_min"] = result[curve_cols].min(axis=1, skipna=True).fillna(0)
    out["damage_max"] = result[curve_cols].max(axis=1, skipna=True).fillna(0)

    # Drop rows with no damage
    return out[out["damage_mean"] > 0] if len(out) else None


# ---------------------------------------------------------------------------
# Per-scenario EAD computation
# ---------------------------------------------------------------------------


def _run_coastal_scenario(
    features: gpd.GeoDataFrame,
    damage_curves: pd.DataFrame,
    multi_curves: dict,
    maxdam: pd.DataFrame,
    exclusions: dict,
    feature_bounds_3035: tuple,
    time_horizon: str,
    climate_scenario: str,
    stac_catalog_url: str,
    col_label: str,
    compute_exposure: bool = False,
) -> tuple[gpd.GeoDataFrame, Optional[pd.Series]]:
    """
    Run coastal assessment for one (time_horizon, climate_scenario) pair.

    Streams tiles one at a time, accumulates only small damage DataFrames,
    aggregates by (osm_id, LAU), then integrates to EAD.

    Args:
        compute_exposure: If True, also compute exposure_coastal_100 from RP100 tiles.

    Returns:
        (enriched features, exposure Series or None)
    """
    print(f"\n[coastal]  → {time_horizon}/{climate_scenario}  →  {col_label}")

    # Accumulate damage DataFrames per RP — only numbers, no rasters
    rp_damage: dict[int, list[pd.DataFrame]] = {}

    # Build (osm_id, LAU) → index lookup once — used for exposure accumulation
    key_to_idx = {
        (row.get("osm_id"), row.get("LAU")): idx for idx, row in features.iterrows()
    }
    exposure_accumulator: dict = {}  # (osm_id, LAU) → running sum across tiles

    tile_count = 0
    for rp, tile in stream_coastal_tiles(
        features=features,
        time_horizon=time_horizon,
        climate_scenario=climate_scenario,
        stac_catalog_url=stac_catalog_url,
    ):
        tile_count += 1

        # Accumulate exposure across all RP100 tiles keyed by (osm_id, LAU)
        # A feature clipped by two adjacent tiles gets partial values from each — sum them
        if compute_exposure and rp == COASTAL_EXPOSURE_RP:
            try:
                exp = compute_exposure_metric(
                    features=features,
                    hazard=tile,
                    reference_rp=COASTAL_EXPOSURE_RP,
                    hazard_value_col=RIVER_HAZARD_COL,
                    pga_threshold=0.0,
                )
                for idx, val in exp.items():
                    if val > 0:
                        row = features.loc[idx]
                        key = (row.get("osm_id"), row.get("LAU"))
                        exposure_accumulator[key] = (
                            exposure_accumulator.get(key, 0.0) + val
                        )
            except Exception as e:
                print(f"  [coastal] Warning: exposure metric failed: {e}")

        dmg = _damage_from_tile(
            tile=tile,
            features=features,
            damage_curves=damage_curves,
            multi_curves=multi_curves,
            maxdam=maxdam,
            exclusions=exclusions,
            feature_bounds_3035=feature_bounds_3035,
        )
        # tile is deleted by the generator after yield — dmg is just numbers
        if dmg is not None and len(dmg) > 0:
            rp_damage.setdefault(rp, []).append(dmg)

    print(
        f"  [coastal] Processed {tile_count} tiles, "
        f"damage found for RPs: {sorted(rp_damage.keys())}"
    )

    # Convert (osm_id, LAU) accumulator → Series indexed like features
    exposure_series: Optional[pd.Series] = None
    if exposure_accumulator:
        exposure_series = pd.Series(
            {
                key_to_idx[k]: v
                for k, v in exposure_accumulator.items()
                if k in key_to_idx
            }
        ).reindex(features.index, fill_value=0.0)

    if not rp_damage:
        features = features.copy()
        features[col_label] = 0.0
        features[f"{col_label}_min"] = 0.0
        features[f"{col_label}_max"] = 0.0
        return features, exposure_series

    # --- Aggregate tile damages per RP by (osm_id, LAU) ---
    # Same osm_id can appear in multiple tiles (it crosses tile boundaries)
    # and the same osm_id can exist in multiple LAUs (physically distinct segments)
    rp_results: dict[int, gpd.GeoDataFrame] = {}

    for rp, frames in rp_damage.items():
        combined = pd.concat(frames, ignore_index=True)

        # Sum across tiles for same (osm_id, LAU) segment
        agg = (
            combined.groupby(_AGG_KEY, dropna=False)[
                ["damage_mean", "damage_min", "damage_max"]
            ]
            .sum()
            .reset_index()
        )

        # Map back to original feature index
        agg["_idx"] = agg.apply(
            lambda r: key_to_idx.get((r["osm_id"], r["LAU"])), axis=1
        )
        agg = agg.dropna(subset=["_idx"])
        agg["_idx"] = agg["_idx"].astype(int)
        agg = agg.set_index("_idx")

        # Build a GeoDataFrame aligned to features index
        rp_gdf = gpd.GeoDataFrame(index=features.index)
        rp_gdf["damage_mean"] = agg["damage_mean"].reindex(features.index).fillna(0)
        rp_gdf["damage_min"] = agg["damage_min"].reindex(features.index).fillna(0)
        rp_gdf["damage_max"] = agg["damage_max"].reindex(features.index).fillna(0)
        rp_results[rp] = rp_gdf

    # --- Integrate to EAD ---
    ead_df = collect_ead_per_asset(
        rp_results=rp_results,
        features=features,
        protection_standards=None,
    )

    features = features.copy()
    features[col_label] = ead_df["EAD"].values
    features[f"{col_label}_min"] = ead_df["EAD_min"].values
    features[f"{col_label}_max"] = ead_df["EAD_max"].values

    print(f"  [coastal] Total {col_label}: {features[col_label].sum():.3e}")
    return features, exposure_series


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def assess_coastal(
    features: gpd.GeoDataFrame,
    vulnerability_path: Union[str, Path],
    asset_type: str,
    stac_catalog_url: str = COASTAL_STAC_URL,
    object_curve_exclusions: Optional[dict] = None,
    scenarios: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """
    Assess coastal flood risk for ALL scenarios in one pass.

    Streams STAC tiles one at a time (memory efficient) and aggregates
    damage by (osm_id, LAU) to correctly handle cross-LAU OSM features.

    Output columns:
      Baseline:  EAD_coastal, EAD_coastal_min, EAD_coastal_max, exposure_coastal_100
      Future:    EAD_coastal_2050_SSP245, ..._min, ..._max  (x4 scenarios)

    Args:
        features:                Exposure GeoDataFrame (EPSG:3035)
        vulnerability_path:      Path to vulnerability Excel file
        asset_type:              Internal asset type (e.g. 'rail', 'roads')
        stac_catalog_url:        CoCLiCo STAC catalog URL
        object_curve_exclusions: {object_type: [curve_ids_to_exclude]}
        scenarios:               Override COASTAL_SCENARIOS if only a subset needed
    """
    t0 = time.time()
    active_scenarios = scenarios or COASTAL_SCENARIOS

    print(
        f"\n[coastal] Starting assessment for {asset_type} "
        f"({len(features)} features, {len(active_scenarios)} scenarios)"
    )

    features = features.to_crs(3035)

    # Validate that LAU column exists
    if "LAU" not in features.columns:
        print("[coastal] Warning: 'LAU' column not found — falling back to osm_id only")
        features = features.copy()
        features["LAU"] = np.nan

    # Vulnerability curves — loaded once, shared across all scenarios
    damage_curves, multi_curves, maxdam_mean, _, _ = prepare_flood_curves(
        asset_type, vulnerability_path
    )

    bounds_3035 = tuple(features.total_bounds)  # (minx, miny, maxx, maxy)
    exclusions = object_curve_exclusions or {}

    # --- Run each scenario ---
    for (time_horizon, climate_scenario), col_label in active_scenarios.items():
        is_baseline = time_horizon == "2010" and climate_scenario == "None"

        features, exposure = _run_coastal_scenario(
            features=features,
            damage_curves=damage_curves,
            multi_curves=multi_curves,
            maxdam=maxdam_mean,
            exclusions=exclusions,
            feature_bounds_3035=bounds_3035,
            time_horizon=time_horizon,
            climate_scenario=climate_scenario,
            stac_catalog_url=stac_catalog_url,
            col_label=col_label,
            compute_exposure=is_baseline,
        )

        # Attach exposure metric from baseline scenario
        if is_baseline and exposure is not None:
            if "exposure_coastal_100" not in features.columns:
                features["exposure_coastal_100"] = exposure.reindex(
                    features.index
                ).values

        gc.collect()

    # Fallback if exposure metric was never set (e.g. no baseline tiles found)
    if "exposure_coastal_100" not in features.columns:
        features["exposure_coastal_100"] = np.nan

    elapsed = time.time() - t0
    print(f"\n[coastal] All scenarios completed in {elapsed / 60:.1f} min")
    return features
