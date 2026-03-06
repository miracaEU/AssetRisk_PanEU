"""
run_pipeline.py

Top-level MIRACA risk assessment pipeline.

For each country + asset type combination:
  1. Load exposure from the harmonised exposure database
  2. Run all hazard assessments (river, coastal, windstorm, earthquake)
  3. Save enriched GeoDataFrame to output directory as parquet

Parallelism: one process per country + asset combination.
             When n_workers > 1, inner RP-level parallelism is disabled
             to avoid nested process pools and memory exhaustion on Windows.

Usage:
    python run_pipeline.py                          # all countries, all assets
    python run_pipeline.py --countries PRT ESP      # specific countries (ISO2 or ISO3)
    python run_pipeline.py --assets power roads     # specific asset types
    python run_pipeline.py --hazards river coastal  # specific hazards only
    python run_pipeline.py --workers 4              # limit parallel workers
    python run_pipeline.py --skip-existing          # skip already-processed files

Configuration:
    Copy config.template.yml to config.yml in the repo root and fill in
    the paths for your machine. config.yml is gitignored and never committed.
"""

import argparse
import os
import sys
import time
import traceback
import warnings
import functools
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import pandas as pd
import geopandas as gpd

from data_loader import (
    load_exposure,
    list_available_countries,
    list_available_asset_types,
    to_iso3,
    to_iso2,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

sys.excepthook = lambda *args: None


# ---------------------------------------------------------------------------
# Configuration — loaded from config.yml in repo root
# ---------------------------------------------------------------------------
def _load_config_file() -> dict:
    config_path = Path(__file__).parent.parent / "config.yml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"config.yml not found at {config_path}.\n"
            "Copy config.template.yml to config.yml and fill in your paths."
        )

_cfg = _load_config_file()

class Config:
    EXPOSURE_DIR             = Path(_cfg["exposure_dir"])
    OUTPUT_DIR               = Path(_cfg["output_dir"])
    RIVER_HAZARD_DIR         = Path(_cfg["river_hazard_dir"])
    WIND_HAZARD_DIR          = Path(_cfg["wind_hazard_dir"])
    EQ_HAZARD_DIR            = Path(_cfg["eq_hazard_dir"])
    VULNERABILITY_PATH       = Path(_cfg["vulnerability_path"])
    FRAGILITY_PATH           = Path(_cfg["fragility_path"])
    PROTECTION_STANDARD_PATH = Path(_cfg["protection_standard_path"])
    BASIN_DATA_PATH          = Path(_cfg["basin_data_path"])
    COASTAL_STAC_URL         = _cfg.get(
        "coastal_stac_url",
        "https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json"
    )

    # Curve exclusions per asset type per hazard (environment-independent)
    FLOOD_CURVE_EXCLUSIONS = {
        "power": {
            "tower":       ["F1.1","F1.2","F1.3","F1.4","F1.5","F1.6","F1.7","F2.1","F2.2","F2.3","F5.1","F6.1","F6.2"],
            "plant":       ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "line":        ["F1.1","F1.2","F1.3","F1.4","F1.5","F1.6","F1.7","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "minor_line":  ["F1.1","F1.2","F1.3","F1.4","F1.5","F1.6","F1.7","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "substation":  ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "generator":   ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "transformer": ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "portal":      ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "terminal":    ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "switch":      ["F1.6","F6.1","F6.2","F10.1","F2.1","F2.2","F2.3","F5.1","F10.1"],
            "pole":        ["F1.1","F1.2","F1.3","F1.4","F1.5","F1.6","F1.7","F2.1","F2.2","F2.3","F5.1"],
        },
    }

    WIND_CURVE_EXCLUSIONS = {
        "power": {
            "tower":       ["W6.1","W6.2","W6.3","W7.2","W1.10","W1.11","W1.12","W1.13","W1.14","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "plant":       ["W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "line":        ["W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "minor_line":  ["W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "substation":  ["W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "generator":   ["W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "transformer": ["W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "portal":      ["W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "terminal":    ["W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "switch":      ["W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2","W4.33","W4.34","W4.35","W4.36","W4.37"],
            "pole":        ["W1.10","W1.13","W1.11","W1.12","W1.14","W3.5","W3.6","W3.7","W3.8","W3.9","W3.10","W3.11","W3.12","W3.13","W3.14","W6.1","W6.2","W6.3","W7.2"],
        },
    }


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def output_path(output_dir: Path, country_iso2: str, asset_type: str) -> Path:
    """Generate output file path for a country + asset combination."""
    iso3 = to_iso3(country_iso2)
    return output_dir / f"{iso3}_{asset_type}_risk.parquet"


# ---------------------------------------------------------------------------
# Single country + asset worker
# ---------------------------------------------------------------------------

def run_single(
    country_iso2: str,
    asset_type: str,
    config: Config,
    hazards: list[str],
    skip_existing: bool,
    n_outer_workers: int = 1,
) -> dict:
    """
    Run the full risk assessment pipeline for one country + asset combination.

    Args:
        n_outer_workers: Number of outer (country+asset) parallel workers.
                         When > 1, inner RP-level parallelism is disabled
                         to avoid nested process pools and memory exhaustion.

    Returns a summary dict with status, timing, and basic stats.
    """
    iso3 = to_iso3(country_iso2)
    label = f"{iso3} / {asset_type}"
    t0 = time.time()

    out_path = output_path(config.OUTPUT_DIR, country_iso2, asset_type)

    if skip_existing and out_path.exists():
        return {"label": label, "status": "skipped", "elapsed": 0}

    # When multiple outer workers are active, disable inner RP-level parallelism
    # to avoid nested ProcessPoolExecutors exhausting memory on Windows
    inner_workers = 1 if n_outer_workers > 1 else None

    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"{'='*60}")

    try:
        # --- 1. Load exposure ---
        print(f"[pipeline] Loading exposure data...")
        features = load_exposure(config.EXPOSURE_DIR, asset_type, country_iso2)
        if features is None or len(features) == 0:
            return {"label": label, "status": "no_data", "elapsed": 0}
        print(f"[pipeline] Loaded {len(features)} features")

        # --- 2. Load basin data for future river ---
        basin_data = None
        if "river" in hazards and config.BASIN_DATA_PATH.exists():
            print("[pipeline] Loading basin climate data...")
            basin_data = gpd.read_parquet(config.BASIN_DATA_PATH)

        # --- 3. River flood ---
        if "river" in hazards:
            from hazard_river import assess_river
            flood_exclusions = config.FLOOD_CURVE_EXCLUSIONS.get(asset_type, {})
            prot_path = config.PROTECTION_STANDARD_PATH if config.PROTECTION_STANDARD_PATH.exists() else None
            features = assess_river(
                features=features,
                hazard_dir=config.RIVER_HAZARD_DIR,
                vulnerability_path=config.VULNERABILITY_PATH,
                asset_type=asset_type,
                protection_standard_path=prot_path,
                basin_data=basin_data,
                object_curve_exclusions=flood_exclusions,
                n_workers=inner_workers,
            )

        # --- 4. Coastal flood ---
        if "coastal" in hazards:
            from hazard_coastal import assess_coastal
            flood_exclusions = config.FLOOD_CURVE_EXCLUSIONS.get(asset_type, {})
            features = assess_coastal(
                features=features,
                vulnerability_path=config.VULNERABILITY_PATH,
                asset_type=asset_type,
                stac_catalog_url=config.COASTAL_STAC_URL,
                object_curve_exclusions=flood_exclusions,
            )

        # --- 5. Windstorm ---
        if "windstorm" in hazards:
            from hazard_windstorm import assess_windstorm
            wind_exclusions = config.WIND_CURVE_EXCLUSIONS.get(asset_type, {})
            features = assess_windstorm(
                features=features,
                hazard_dir=config.WIND_HAZARD_DIR,
                vulnerability_path=config.VULNERABILITY_PATH,
                asset_type=asset_type,
                object_curve_exclusions=wind_exclusions,
                n_workers=inner_workers,
            )

        # --- 6. Earthquake ---
        if "earthquake" in hazards:
            from hazard_earthquake import assess_earthquake
            features = assess_earthquake(
                features=features,
                hazard_dir=config.EQ_HAZARD_DIR,
                fragility_path=config.FRAGILITY_PATH,
                asset_type=asset_type,
                n_workers=inner_workers,
            )

        # --- 7. Save output ---
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        features.to_parquet(str(out_path))
        print(f"[pipeline] Saved to {out_path}")

        # --- 8. Summary stats ---
        elapsed = time.time() - t0
        stats = {"label": label, "status": "ok", "elapsed": elapsed, "n_features": len(features)}

        ead_cols = [c for c in features.columns if c.startswith("EAD_") and not c.endswith(("_min", "_max"))]
        for col in ead_cols:
            stats[f"total_{col}"] = float(features[col].sum())

        _print_summary(label, features, elapsed)
        return stats

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[pipeline] ERROR for {label}: {e}")
        print(traceback.format_exc())
        return {"label": label, "status": "error", "error": str(e), "elapsed": elapsed}


def _print_summary(label: str, features: gpd.GeoDataFrame, elapsed: float):
    """Print a concise summary of results."""
    print(f"\n{'─'*50}")
    print(f"  {label} — completed in {elapsed:.1f}s")
    print(f"  Features: {len(features)}")

    ead_cols = [c for c in features.columns
                if c.startswith("EAD_") and not c.endswith(("_min", "_max"))]
    for col in ead_cols:
        total = features[col].sum()
        mean  = features[col].mean()
        print(f"  {col}: total={total:.3e}, mean={mean:.3e}")

    exp_cols = [c for c in features.columns if c.startswith("exposure_")]
    for col in exp_cols:
        total = features[col].sum()
        print(f"  {col}: total={total:.3e}")
    print(f"{'─'*50}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    config: Config,
    countries: Optional[list[str]] = None,
    asset_types: Optional[list[str]] = None,
    hazards: Optional[list[str]] = None,
    n_workers: Optional[int] = None,
    skip_existing: bool = False,
):
    """
    Run the full MIRACA risk pipeline.

    Args:
        config:        Config object with all paths and settings
        countries:     List of ISO2 or ISO3 codes to process (None = all available)
        asset_types:   List of internal asset type names to process (None = all available)
        hazards:       List of hazards to run: river, coastal, windstorm, earthquake
                       (None = all)
        n_workers:     Max parallel workers for country+asset combinations.
                       When > 1, inner RP-level parallelism is automatically disabled.
                       Set to 1 for sequential combinations with full inner parallelism.
                       (None = all CPUs, inner parallelism disabled)
        skip_existing: Skip combinations where output file already exists
    """
    all_hazards = ["river", "coastal", "windstorm", "earthquake"]
    hazards = hazards or all_hazards

    if countries is None:
        countries_iso2 = list_available_countries(config.EXPOSURE_DIR)
    else:
        countries_iso2 = [to_iso2(c) for c in countries]

    if asset_types is None:
        asset_types = list_available_asset_types(config.EXPOSURE_DIR)

    work_items = [
        (country, asset)
        for country in countries_iso2
        for asset in asset_types
        if (config.EXPOSURE_DIR / _folder(asset) / f"{_folder(asset)}_{country}.parquet").exists()
    ]

    # Resolve effective outer worker count
    effective_workers = n_workers if n_workers is not None else os.cpu_count()

    print(f"\n{'='*60}")
    print(f"MIRACA RISK PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Countries:   {countries_iso2}")
    print(f"Asset types: {asset_types}")
    print(f"Hazards:     {hazards}")
    print(f"Work items:  {len(work_items)}")
    print(f"Workers:     {effective_workers} outer"
          f" / {'1 (disabled)' if effective_workers > 1 else 'all CPUs'} inner")
    print(f"Output dir:  {config.OUTPUT_DIR}")
    print(f"{'='*60}\n")

    if not work_items:
        print("No work items found. Check exposure directory and country/asset filters.")
        return

    t_start = time.time()

    worker_fn = functools.partial(
        _run_single_unpacked,
        config=config,
        hazards=hazards,
        skip_existing=skip_existing,
        n_outer_workers=effective_workers,
    )

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(worker_fn, country, asset): (country, asset)
            for country, asset in work_items
        }
        for future in concurrent.futures.as_completed(futures):
            country, asset = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result.get("status", "?")
                elapsed = result.get("elapsed", 0)
                print(f"  ✓ {to_iso3(country)} / {asset} — {status} ({elapsed:.0f}s)")
            except Exception as e:
                results.append({"label": f"{country}/{asset}", "status": "error", "error": str(e)})
                print(f"  ✗ {to_iso3(country)} / {asset} — ERROR: {e}")

    # Final summary
    total_elapsed = time.time() - t_start
    n_ok      = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_error   = sum(1 for r in results if r["status"] == "error")
    n_nodata  = sum(1 for r in results if r["status"] == "no_data")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Total time:  {total_elapsed/60:.1f} minutes")
    print(f"  Completed:   {n_ok}")
    print(f"  Skipped:     {n_skipped}")
    print(f"  No data:     {n_nodata}")
    print(f"  Errors:      {n_error}")

    if n_error > 0:
        print("\n  Failed combinations:")
        for r in results:
            if r["status"] == "error":
                print(f"    - {r['label']}: {r.get('error', '?')}")

    log_path = config.OUTPUT_DIR / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(log_path, index=False)
    print(f"\n  Run log saved to: {log_path}")


def _run_single_unpacked(country, asset, config, hazards, skip_existing, n_outer_workers):
    """Unpacked wrapper for ProcessPoolExecutor (needs top-level picklable function)."""
    return run_single(country, asset, config, hazards, skip_existing, n_outer_workers)


def _folder(asset_type: str) -> str:
    """Get folder name from internal asset type (via data_loader mapping)."""
    from data_loader import to_folder_asset
    return to_folder_asset(asset_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MIRACA multi-hazard infrastructure risk assessment pipeline"
    )
    parser.add_argument(
        "--countries", nargs="+", default=None,
        help="ISO2 or ISO3 country codes to process (default: all available)"
    )
    parser.add_argument(
        "--assets", nargs="+", default=None,
        help="Asset types to process: power roads rail air telecom education healthcare "
             "gas oil ports (default: all available)"
    )
    parser.add_argument(
        "--hazards", nargs="+",
        default=["river", "coastal", "windstorm", "earthquake"],
        choices=["river", "coastal", "windstorm", "earthquake"],
        help="Hazards to assess (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for country+asset combinations. "
             "When > 1, inner RP-level parallelism is disabled automatically. "
             "Set to 1 for sequential combinations with full inner parallelism. "
             "(default: all CPUs, inner parallelism disabled)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip combinations where output file already exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()
    cfg = Config()

    print(f"\nConfig loaded from: {Path(__file__).parent / 'config.yml'}")
    print(f"Exposure dir: {cfg.EXPOSURE_DIR}")

    if cfg.EXPOSURE_DIR.exists():
        countries = list_available_countries(cfg.EXPOSURE_DIR)
        assets    = list_available_asset_types(cfg.EXPOSURE_DIR)
        print(f"Discovered countries: {countries}")
        print(f"Discovered assets:    {assets}")
    else:
        print("\n⚠ Exposure directory not found — nothing to process.")
        print("  Please check the paths in your config.yml")
        sys.exit(1)

    run_pipeline(
        config=cfg,
        countries=args.countries,
        asset_types=args.assets,
        hazards=args.hazards,
        n_workers=args.workers,
        skip_existing=args.skip_existing,
    )
