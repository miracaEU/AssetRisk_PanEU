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
# Return periods are taken from the CoCliCo STAC items themselves (1, 100, 1000).
COASTAL_EXPOSURE_RP = 100
COASTAL_PROTECTION_SHEET = "COASTPROS-EU"
COASTAL_PROTECTION_RP_COL = "MODELLED RETURN PERIOD"

# (time_horizon, climate_scenario) → output column label prefix
# The full columns will be: {label}_mid, {label}_min, {label}_max
COASTAL_SCENARIOS = {
    ("2010", "None"): "EAD_mid_coastal_current",
    ("2050", "SSP245"): "EAD_mid_coastal_2050_SSP245",
    ("2050", "SSP585"): "EAD_mid_coastal_2050_SSP585",
    ("2100", "SSP245"): "EAD_mid_coastal_2100_SSP245",
    ("2100", "SSP585"): "EAD_mid_coastal_2100_SSP585",
}

# Unique key for aggregating damage — osm_id alone is insufficient
# because the same OSM way can cross LAU boundaries
_AGG_KEY = ["osm_id", "LAU"]


# ---------------------------------------------------------------------------
# Protection standards
# ---------------------------------------------------------------------------


def load_coastal_protection_standards(
    features: gpd.GeoDataFrame,
    coastpros_path: Union[str, Path],
    nuts2_path: Union[str, Path],
) -> pd.Series:
    """
    Assign coastal flood protection standard (design RP) to each asset via
    NUTS2 spatial join against COASTPROS-EU.

    Uses modelled RP values (mod_rp column). NUTS2 regions with no modelled
    value fall back to the country-level mean; countries with no data at all
    get 0 (unprotected).

    Args:
        features:       Exposure GeoDataFrame (any CRS)
        coastpros_path: Path to COASTPROS-EU.xlsx
        nuts2_path:     Path to NUTS2 geometry file (parquet or GeoJSON, EPSG:3035)

    Returns:
        Series mapping feature index → protection standard RP (0 if unknown)
    """
    print("[coastal] Loading coastal protection standards from COASTPROS-EU...")

    # --- 1. Read and parse COASTPROS-EU ---
    df = pd.read_excel(coastpros_path, sheet_name=COASTAL_PROTECTION_SHEET)

    # Normalise column names: strip whitespace
    df.columns = df.columns.str.strip()

    nuts2_id_col = "NUTS2 ID"
    cntr_col = "CNTR_CODE"
    rp_col = COASTAL_PROTECTION_RP_COL

    # Parse mod_rp to numeric (NA, empty, text → NaN)
    df[rp_col] = pd.to_numeric(df[rp_col], errors="coerce")

    # Country-level fallback: mean of available modelled RPs per country
    country_mean = (
        df.dropna(subset=[rp_col])
        .groupby(cntr_col)[rp_col]
        .mean()
    )

    # Fill NaN with country mean, then 0 for countries with no data at all
    def _fill(row):
        if pd.notna(row[rp_col]):
            return row[rp_col]
        return country_mean.get(row[cntr_col], 0.0)

    df["_rp_filled"] = df.apply(_fill, axis=1)
    nuts2_rp = df.set_index(nuts2_id_col)["_rp_filled"].to_dict()

    n_nuts2 = df[nuts2_id_col].notna().sum()
    n_with_data = df[rp_col].notna().sum()
    print(
        f"  [coastal] COASTPROS: {n_nuts2} NUTS2 regions, "
        f"{n_with_data} with modelled RP, "
        f"{n_nuts2 - n_with_data} filled from country mean"
    )

    # --- 2. Load NUTS2 geometries ---
    path = Path(nuts2_path)
    if path.suffix == ".parquet":
        nuts2_gdf = gpd.read_parquet(nuts2_path)
    else:
        nuts2_gdf = gpd.read_file(nuts2_path)

    nuts2_gdf = nuts2_gdf[nuts2_gdf["LEVL_CODE"] == 2].copy()
    nuts2_gdf = nuts2_gdf.to_crs(3035)

    # Primary lookup: NUTS_ID → RP; fallback: CNTR_CODE → country mean → 0
    nuts2_gdf["_rp"] = nuts2_gdf.apply(
        lambda r: nuts2_rp.get(r["NUTS_ID"], country_mean.get(r["CNTR_CODE"], 0.0)),
        axis=1,
    )

    # --- 3. Spatial join: feature centroids → NUTS2 ---
    # Two-step: within first, then nearest fallback for coastal/offshore features
    # whose centroids fall outside land polygons.
    features_3035 = features.to_crs(3035)
    centroids = gpd.GeoDataFrame(
        geometry=features_3035.geometry.centroid,
        index=features.index,
        crs=3035,
    )

    nuts2_cols = nuts2_gdf[["geometry", "_rp"]].copy()

    joined = gpd.sjoin(centroids, nuts2_cols, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    # Fallback: features not matched (centroid outside all NUTS2) → nearest NUTS2
    unmatched_idx = joined.index[joined["_rp"].isna()]
    if len(unmatched_idx) > 0:
        nearest = gpd.sjoin_nearest(
            centroids.loc[unmatched_idx],
            nuts2_cols,
            how="left",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched_idx, "_rp"] = nearest["_rp"].values

    protection = joined["_rp"].reindex(features.index).fillna(0.0).clip(lower=0)

    print(
        f"  [coastal] Protection standards assigned. "
        f"Mean: {protection.mean():.0f} yr, Max: {protection.max():.0f} yr, "
        f"Unprotected (0): {(protection == 0).sum()}"
    )
    return protection


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
    print(f"[coastal] Connecting to STAC: {stac_catalog_url}")
    try:
        catalog = pystac_client.Client.open(stac_catalog_url)
        collection = catalog.get_child(id=collection_id)
        items = list(collection.get_items())
        print(f"[coastal] Connected OK — {len(items)} items in collection '{collection_id}'")
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
    asset_type: str = None,
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
            asset_type=asset_type,
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
    asset_type: str = None,
    protection_standards: Optional[pd.Series] = None,
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
    def _norm_key(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    key_to_idx = {
        (_norm_key(row.get("osm_id")), _norm_key(row.get("LAU"))): idx
        for idx, row in features.iterrows()
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
                        key = (_norm_key(row.get("osm_id")), _norm_key(row.get("LAU")))
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
            asset_type=asset_type,
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

        # Cap at the feature's true geometry size — a feature exposed in
        # >1 overlapping STAC tile gets compute_exposure_metric's FULL
        # length/area/count summed once per tile, which can exceed the
        # feature's actual size. Sum is kept (legitimate for features that
        # genuinely intersect adjacent tiles), capping just removes the
        # double-count overshoot.
        geom_types = features.geometry.geom_type
        true_size = pd.Series(1.0, index=features.index)
        is_line = geom_types.isin(["LineString", "MultiLineString"])
        is_poly = geom_types.isin(["Polygon", "MultiPolygon"])
        true_size[is_line] = features.geometry[is_line].length
        true_size[is_poly] = features.geometry[is_poly].area
        exposure_series = exposure_series.clip(upper=true_size)

    if not rp_damage:
        features = features.copy()
        col_min = col_label.replace("EAD_mid_", "EAD_min_")
        col_max = col_label.replace("EAD_mid_", "EAD_max_")
        features[col_label] = 0.0
        features[col_min] = 0.0
        features[col_max] = 0.0
        return features, exposure_series, {}

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
        def _norm(v):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        agg["_idx"] = agg.apply(
            lambda r: key_to_idx.get((_norm(r["osm_id"]), _norm(r["LAU"]))), axis=1
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
        protection_standards=protection_standards,
    )

    features = features.copy()
    col_min = col_label.replace("EAD_mid_", "EAD_min_")
    col_max = col_label.replace("EAD_mid_", "EAD_max_")
    features[col_label] = ead_df["EAD_mid"].values
    features[col_min] = ead_df["EAD_min"].values
    features[col_max] = ead_df["EAD_max"].values

    print(f"  [coastal] Total {col_label}: {features[col_label].sum():.3e}")
    return features, exposure_series, rp_results


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
    return_rp_results: bool = False,
    pre_loaded_curves: Optional[tuple] = None,
    coastpros_path: Optional[Union[str, Path]] = None,
    nuts2_path: Optional[Union[str, Path]] = None,
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

    # Vulnerability curves — use pre-loaded (e.g. port-specific) or load from Excel
    if pre_loaded_curves is not None:
        damage_curves, multi_curves, maxdam_mean = pre_loaded_curves
    else:
        damage_curves, multi_curves, maxdam_mean, _, _ = prepare_flood_curves(
            asset_type, vulnerability_path
        )

    bounds_3035 = tuple(features.total_bounds)  # (minx, miny, maxx, maxy)
    exclusions = object_curve_exclusions or {}

    # --- Protection standards (optional) ---
    protection_standards = None
    if coastpros_path is not None and nuts2_path is not None:
        protection_standards = load_coastal_protection_standards(
            features, coastpros_path, nuts2_path
        )

    # --- Run each scenario ---
    # all_rp_results: scenario_key → rp_results dict (for TENT per-RP columns)
    # scenario_key = "current" for baseline, "2050_SSP245" etc. for future
    all_rp_results: dict[str, dict] = {}
    for (time_horizon, climate_scenario), col_label in active_scenarios.items():
        # Compute exposure for all scenarios (not just baseline)
        features, exposure, rp_results_scenario = _run_coastal_scenario(
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
            compute_exposure=True,
            asset_type=asset_type,
            protection_standards=protection_standards,
        )

        # Derive exposure column name and scenario key from col_label
        suffix = col_label.replace("EAD_mid_coastal_", "")  # "current", "2050_SSP245", ...
        exp_col = f"exposure_abs_coastal_{suffix}"
        all_rp_results[suffix] = rp_results_scenario

        if exposure is not None:
            features[exp_col] = exposure.reindex(features.index).values
        else:
            features[exp_col] = 0.0

        gc.collect()

    # Fallback for any missing exposure columns
    if "exposure_abs_coastal_current" not in features.columns:
        features["exposure_abs_coastal_current"] = np.nan

    elapsed = time.time() - t0
    print(f"\n[coastal] All scenarios completed in {elapsed / 60:.1f} min")
    if return_rp_results:
        # Return dict of {scenario_key: rp_results} so TENT wrapper can attach
        # per-RP vuln_ratio columns for all scenarios, not just baseline.
        return features, all_rp_results
    return features
