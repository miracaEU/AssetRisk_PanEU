"""
run_pipeline_tent.py

TENT-specific risk assessment pipeline.

Processes Europe-wide TENT data (road + rail edges, airport polygons) chunked by object_type
(tag_highway for roads, tag_railway for rail, aeroway for airports).

Each chunk is processed through all selected hazards, then saved as a
separate parquet file. Results can be concatenated afterwards.

Usage:
    python run_pipeline_tent.py                              # all assets, all hazards
    python run_pipeline_tent.py --assets roads               # roads only
    python run_pipeline_tent.py --assets rail                # rail only
    python run_pipeline_tent.py --hazards river windstorm    # specific hazards
    python run_pipeline_tent.py --corridors-only             # TEN-T corridor edges only
    python run_pipeline_tent.py --workers 4                  # parallel workers for RP calcs
    python run_pipeline_tent.py --skip-existing              # skip already-processed chunks
    python run_pipeline_tent.py --country BEL                # filter to Belgium only

Configuration:
    Uses the same config.yml as the main pipeline, plus a 'tent_data_dir' entry.
"""

import argparse
import os
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

# TENT modules reuse the hazard/risk modules of the paper pipeline in ../src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader_tent import (
    load_tent_edges,
    list_available_asset_types,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

sys.excepthook = lambda *args: None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_config_file() -> dict:
    config_path = Path(__file__).parent.parent / "config.yml"
    # Also check current directory
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"config.yml not found at {config_path}.\n"
            "Copy config.template.yml to config.yml and fill in your paths.\n"
            "Ensure 'tent_data_dir' is set to the TENT parquet directory."
        )


_cfg = _load_config_file()


class Config:
    TENT_DATA_DIR = Path(_cfg["tent_data_dir"])
    OUTPUT_DIR = Path(_cfg.get("tent_output_dir", _cfg["output_dir"])) / "TENT"
    RIVER_HAZARD_DIR = Path(_cfg["river_hazard_dir"])
    WIND_HAZARD_DIR = Path(_cfg["wind_hazard_dir"])
    EQ_HAZARD_DIR = Path(_cfg["eq_hazard_dir"])
    VULNERABILITY_PATH = Path(_cfg["vulnerability_path"])
    FRAGILITY_PATH = Path(_cfg["fragility_path"])
    PROTECTION_STANDARD_PATH = Path(_cfg["protection_standard_path"])
    BASIN_DATA_PATH = Path(_cfg["basin_data_path"])
    COASTAL_STAC_URL = _cfg.get(
        "coastal_stac_url",
        "https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json",
    )

    # No curve exclusions for roads/rail (only relevant for power)
    FLOOD_CURVE_EXCLUSIONS: dict = {}
    WIND_CURVE_EXCLUSIONS: dict = {}


# ---------------------------------------------------------------------------
# Output path + completeness check
# ---------------------------------------------------------------------------

COMPLETE_EAD_COLS = [
    "EAD_mid_river_current",
    "EAD_min_river_current",
    "EAD_max_river_current",
    "exposure_abs_river_current",
    "EAD_mid_river_2050_SSP245",
    "EAD_min_river_2050_SSP245",
    "EAD_max_river_2050_SSP245",
    "EAD_mid_river_2050_SSP585",
    "EAD_min_river_2050_SSP585",
    "EAD_max_river_2050_SSP585",
    "EAD_mid_river_2100_SSP245",
    "EAD_min_river_2100_SSP245",
    "EAD_max_river_2100_SSP245",
    "EAD_mid_river_2100_SSP585",
    "EAD_min_river_2100_SSP585",
    "EAD_max_river_2100_SSP585",
    "EAD_mid_coastal_current",
    "EAD_min_coastal_current",
    "EAD_max_coastal_current",
    "EAD_mid_coastal_2050_SSP245",
    "EAD_min_coastal_2050_SSP245",
    "EAD_max_coastal_2050_SSP245",
    "EAD_mid_coastal_2050_SSP585",
    "EAD_min_coastal_2050_SSP585",
    "EAD_max_coastal_2050_SSP585",
    "EAD_mid_coastal_2100_SSP245",
    "EAD_min_coastal_2100_SSP245",
    "EAD_max_coastal_2100_SSP245",
    "EAD_mid_coastal_2100_SSP585",
    "EAD_min_coastal_2100_SSP585",
    "EAD_max_coastal_2100_SSP585",
    "exposure_abs_coastal_current",
    "EAD_mid_windstorm_current",
    "EAD_min_windstorm_current",
    "EAD_max_windstorm_current",
    "exposure_abs_windstorm_current",
    "EAD_mid_earthquake_current",
    "EAD_min_earthquake_current",
    "EAD_max_earthquake_current",
    "exposure_abs_earthquake_current",
]


def is_complete(path: Path, hazards: list[str]) -> bool:
    """
    Check if an output parquet file exists and contains all required EAD columns
    for the specified hazards.
    """
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        existing_cols = set(schema.names)

        # Only check columns relevant to requested hazards
        required = []
        for col in COMPLETE_EAD_COLS:
            for h in hazards:
                if h in col:
                    required.append(col)
                    break

        missing = [c for c in required if c not in existing_cols]
        if missing:
            print(
                f"  [skip check] {path.name} incomplete — missing {len(missing)} cols "
                f"(e.g. {missing[:3]}), will re-run."
            )
            return False
        return True
    except Exception as e:
        print(f"  [skip check] Could not read {path.name}: {e} — will re-run.")
        return False


# ---------------------------------------------------------------------------
# Single asset type worker
# ---------------------------------------------------------------------------


def run_single_asset(
    features: gpd.GeoDataFrame,
    asset_type: str,
    config: Config,
    hazards: list[str],
    n_workers: Optional[int] = None,
) -> tuple[dict, Optional[gpd.GeoDataFrame]]:
    """
    Run the full risk assessment pipeline for one asset type (all edges).

    Runs all edges through hazard assessment in a single pass — hazard
    rasters are loaded once for the full bounding box.

    Args:
        features:       Full GeoDataFrame for this asset type (already in EPSG:3035)
        asset_type:     'roads' or 'rail'
        config:         Config object with all paths
        hazards:        List of hazards to run
        n_workers:      Workers for inner RP-level parallelism

    Returns:
        Tuple of (summary dict, processed GeoDataFrame or None on error)
    """
    label = f"TENT {asset_type}"
    t0 = time.time()

    print(f"\n{'=' * 60}")
    print(f"Processing: {label} ({len(features)} edges)")
    print(f"Object types: {sorted(features['object_type'].unique().tolist())}")
    print(f"{'=' * 60}")

    try:
        # --- 1. Load basin data once ---
        basin_data = None
        if "river" in hazards and config.BASIN_DATA_PATH.exists():
            print("[pipeline] Loading basin climate data...")
            basin_data = gpd.read_parquet(config.BASIN_DATA_PATH)

        # --- 2. River flood (with per-RP ratio + exposure length) ---
        if "river" in hazards:
            from hazard_wrappers_tent import assess_river_tent

            flood_exclusions = config.FLOOD_CURVE_EXCLUSIONS.get(asset_type, {})
            prot_path = (
                config.PROTECTION_STANDARD_PATH
                if config.PROTECTION_STANDARD_PATH.exists()
                else None
            )
            features = assess_river_tent(
                features=features,
                hazard_dir=config.RIVER_HAZARD_DIR,
                vulnerability_path=config.VULNERABILITY_PATH,
                asset_type=asset_type,
                protection_standard_path=prot_path,
                basin_data=basin_data,
                object_curve_exclusions=flood_exclusions,
                n_workers=n_workers,
            )

        # --- 3. Coastal flood (with per-RP ratio + exposure length) ---
        if "coastal" in hazards:
            from hazard_wrappers_tent import assess_coastal_tent

            flood_exclusions = config.FLOOD_CURVE_EXCLUSIONS.get(asset_type, {})
            features = assess_coastal_tent(
                features=features,
                vulnerability_path=config.VULNERABILITY_PATH,
                asset_type=asset_type,
                stac_catalog_url=config.COASTAL_STAC_URL,
                object_curve_exclusions=flood_exclusions,
            )

        # --- 4. Windstorm (with per-RP ratio + exposure length) ---
        if "windstorm" in hazards:
            from hazard_wrappers_tent import assess_windstorm_tent

            wind_exclusions = config.WIND_CURVE_EXCLUSIONS.get(asset_type, {})
            features = assess_windstorm_tent(
                features=features,
                hazard_dir=config.WIND_HAZARD_DIR,
                vulnerability_path=config.VULNERABILITY_PATH,
                asset_type=asset_type,
                object_curve_exclusions=wind_exclusions,
                n_workers=n_workers,
            )

        # --- 5. Earthquake (with per-RP ratio + exposure length) ---
        if "earthquake" in hazards:
            from hazard_wrappers_tent import assess_earthquake_tent

            features = assess_earthquake_tent(
                features=features,
                hazard_dir=config.EQ_HAZARD_DIR,
                fragility_path=config.FRAGILITY_PATH,
                asset_type=asset_type,
                n_workers=n_workers,
            )

        # --- 6. Compute asset_size and exposure_rel columns ---
        print("[pipeline] Computing asset_size and relative exposure columns...")

        if asset_type == "ports" and "area" in features.columns and features.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
            # Fallback: port points — use area attribute column
            asset_size = pd.to_numeric(features["area"], errors="coerce").fillna(0)
            asset_size = pd.Series(asset_size.values, index=features.index)
        else:
            geom_types = features.geometry.geom_type
            asset_size = pd.Series(0.0, index=features.index)

            is_line = geom_types.isin(["LineString", "MultiLineString"])
            is_poly = geom_types.isin(["Polygon", "MultiPolygon"])
            is_point = ~is_line & ~is_poly

            if is_line.any():
                asset_size[is_line] = features.geometry[is_line].length
            if is_poly.any():
                asset_size[is_poly] = features.geometry[is_poly].area
            if is_point.any():
                asset_size[is_point] = 1.0

        features["asset_size"] = asset_size.values

        for col in list(features.columns):
            if col.startswith("exposure_abs_"):
                rel_col = col.replace("exposure_abs_", "exposure_rel_")
                features[col] = features[col].astype(float)
                features[rel_col] = 0.0
                nonzero = features["asset_size"] > 0
                features.loc[nonzero, rel_col] = (
                    features.loc[nonzero, col].values / features.loc[nonzero, "asset_size"].values
                ).clip(0, 1)



        # --- 7. Summary stats ---
        elapsed = time.time() - t0
        stats = {
            "label": label,
            "status": "ok",
            "elapsed": elapsed,
            "n_features": len(features),
        }

        ead_cols = [c for c in features.columns if c.startswith("EAD_mid_")]
        for col in ead_cols:
            stats[f"total_{col}"] = float(features[col].sum())

        _print_summary(label, features, elapsed)
        return stats, features

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[pipeline] ERROR for {label}: {e}")
        print(traceback.format_exc())
        return {"label": label, "status": "error", "error": str(e), "elapsed": elapsed}, None


def _print_summary(label: str, features: gpd.GeoDataFrame, elapsed: float):
    """Print a concise summary of results."""
    print(f"\n{'─' * 50}")
    print(f"  {label} — completed in {elapsed:.1f}s")
    print(f"  Features: {len(features)}")

    ead_cols = [c for c in features.columns if c.startswith("EAD_mid_")]
    for col in ead_cols:
        total = features[col].sum()
        mean = features[col].mean()
        print(f"  {col}: total={total:.3e}, mean={mean:.3e}")

    exp_cols = [c for c in features.columns if c.startswith("exposure_")]
    for col in exp_cols:
        total = features[col].sum()
        print(f"  {col}: total={total:.3e}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    config: Config,
    asset_types: Optional[list[str]] = None,
    hazards: Optional[list[str]] = None,
    n_workers: Optional[int] = None,
    skip_existing: bool = False,
    corridors_only: bool = False,
    country_iso3: Optional[str] = None,
    corridor_id: Optional[str] = None,
):
    """
    Run the TENT risk pipeline.

    Loads each asset type as a single GeoDataFrame and runs all hazards
    in one pass — hazard rasters are loaded once per asset type.

    Args:
        config:         Config object with all paths
        asset_types:    List of asset types to process (default: all available)
        hazards:        List of hazards to run (default: all)
        n_workers:      Workers for inner RP-level parallelism
        skip_existing:  Skip asset types where output already exists
        corridors_only: Only process edges on TEN-T corridors
        country_iso3:   ISO 3166-1 alpha-3 code to filter edges by country (e.g. 'BEL')
    """
    all_hazards = ["river", "coastal", "windstorm", "earthquake"]
    hazards = hazards or all_hazards

    if asset_types is None:
        asset_types = list_available_asset_types(config.TENT_DATA_DIR)

    print(f"\n{'=' * 60}")
    print("MIRACA RISK PIPELINE — TENT EDITION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"Data dir:      {config.TENT_DATA_DIR}")
    print(f"Asset types:   {asset_types}")
    print(f"Hazards:       {hazards}")
    print(f"Corridors only: {corridors_only}")
    print(f"Country filter: {country_iso3.upper() if country_iso3 else 'none (all Europe)'}")
    print(f"Workers:       {n_workers or 'all CPUs'}")
    print(f"Output dir:    {config.OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    results = []

    for asset_type in asset_types:
        print(f"\n{'#' * 60}")
        print(f"# Loading {asset_type} edges")
        print(f"{'#' * 60}")

        # Build output filename including country and hazard(s)
        hazard_suffix = "_".join(sorted(hazards))
        country_suffix = f"_{country_iso3.upper()}" if country_iso3 else ""
        corridor_suffix = f"_corridor{corridor_id}" if corridor_id else ""
        out_path = config.OUTPUT_DIR / f"TENT_{asset_type}{country_suffix}{corridor_suffix}_{hazard_suffix}.parquet"

        # Check if output already exists and is complete
        if skip_existing and is_complete(out_path, hazards):
            print(f"[pipeline] {asset_type}/{hazard_suffix} already complete, skipping.")
            results.append({
                "label": f"TENT {asset_type}",
                "status": "skipped",
                "elapsed": 0,
            })
            continue

        gdf = load_tent_edges(
            data_dir=config.TENT_DATA_DIR,
            asset_type=asset_type,
            corridors_only=corridors_only,
            country_iso3=country_iso3,
            corridor_id=corridor_id,
        )

        if gdf is None or len(gdf) == 0:
            print(f"[pipeline] No data for {asset_type}, skipping.")
            results.append({
                "label": f"TENT {asset_type}",
                "status": "no_data",
                "elapsed": 0,
            })
            continue

        # Run all hazards on the full edge set in one pass
        result, processed_gdf = run_single_asset(
            features=gdf,
            asset_type=asset_type,
            config=config,
            hazards=hazards,
            n_workers=n_workers,
        )
        results.append(result)

        # Save output
        if processed_gdf is not None:
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            processed_gdf.to_parquet(str(out_path))
            print(f"[pipeline] Saved: {out_path} ({len(processed_gdf)} edges)")
            del processed_gdf

    # Final summary
    total_elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_nodata = sum(1 for r in results if r["status"] == "no_data")

    print(f"\n{'=' * 60}")
    print(f"TENT PIPELINE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"  Total time:  {total_elapsed / 60:.1f} minutes")
    print(f"  Completed:   {n_ok}")
    print(f"  Skipped:     {n_skipped}")
    print(f"  No data:     {n_nodata}")
    print(f"  Errors:      {n_error}")

    if n_error > 0:
        print("\n  Failed chunks:")
        for r in results:
            if r["status"] == "error":
                print(f"    - {r['label']}: {r.get('error', '?')}")

    # Save log
    log_dir = config.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"tent_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(log_path, index=False)
    print(f"\n  Run log saved to: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIRACA TENT multi-hazard risk assessment pipeline"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        choices=["roads", "rail", "airports", "ports", "iww", "energy_lines", "energy_buses"],
        help="Asset types to process (default: all available)",
    )
    parser.add_argument(
        "--hazards",
        nargs="+",
        default=["river", "coastal", "windstorm", "earthquake"],
        choices=["river", "coastal", "windstorm", "earthquake"],
        help="Hazards to assess (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for RP-level calculations "
        "(default: all CPUs)",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip chunks where output already exists (default: on)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-run all chunks even if output already exists",
    )
    parser.add_argument(
        "--corridors-only",
        dest="corridors_only",
        action="store_true",
        default=False,
        help="Only process edges on TEN-T corridors",
    )
    parser.add_argument(
        "--country",
        dest="country_iso3",
        default=None,
        help="ISO 3166-1 alpha-3 country code to filter edges (e.g. BEL for Belgium)",
    )
    parser.add_argument(
        "--corridor",
        dest="corridor_id",
        default=None,
        help="Corridor letter code to filter edges (e.g. L for Rhine-Alpine)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    args = parse_args()
    cfg = Config()

    print(f"\nTENT data dir: {cfg.TENT_DATA_DIR}")

    if not cfg.TENT_DATA_DIR.exists():
        print("\n⚠ TENT data directory not found — nothing to process.")
        print("  Please set 'tent_data_dir' in your config.yml")
        sys.exit(1)

    available = list_available_asset_types(cfg.TENT_DATA_DIR)
    print(f"Available TENT asset types: {available}")

    run_pipeline(
        config=cfg,
        asset_types=args.assets,
        hazards=args.hazards,
        n_workers=args.workers,
        skip_existing=args.skip_existing,
        corridors_only=args.corridors_only,
        country_iso3=args.country_iso3,
        corridor_id=args.corridor_id,
    )