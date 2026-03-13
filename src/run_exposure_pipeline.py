"""
run_exposure_pipeline.py

Main runner for the MIRACA exposure pipeline (heat, wildfire, landslide).

Produces one parquet per country × asset_type:
  {exposure_output_dir}/{ISO3}_{asset_type}_exposure.parquet

with flat columns:
  landslide_min, landslide_avg, landslide_max, landslide_exposure, landslide_max_cat
  heat_{threshold}_{scenario}_{window}_{stat}
  wildfire_{scenario}_{window}_{stat}

Usage:
  uv run python src/run_exposure_pipeline.py
  uv run python src/run_exposure_pipeline.py --countries PRT ESP
  uv run python src/run_exposure_pipeline.py --countries PRT --assets healthcare rail
  uv run python src/run_exposure_pipeline.py --hazards heat wildfire
  uv run python src/run_exposure_pipeline.py --workers 4
"""

import argparse
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd

from exposure_utils import (
    load_config,
    load_infrastructure,
    to_iso2,
)
from exposure_landslide import assess_landslide
from exposure_heat import assess_heat
from exposure_wildfire import assess_wildfire

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_COUNTRIES = [
    "ALB",
    "AUT",
    "BEL",
    "BGR",
    "BIH",
    "CHE",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "EST",
    "ESP",
    "FIN",
    "FRA",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ISL",
    "ITA",
    "LIE",
    "LTU",
    "LUX",
    "LVA",
    "MKD",
    "MLT",
    "MNE",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "SRB",
    "SVK",
    "SVN",
    "SWE",
    "XKO",
]

DEFAULT_ASSETS = [
    "roads",
    "main_roads",
    "rail",
    "air",
    "telecom",
    "education",
    "healthcare",
    "power",
    "gas",
    "oil",
    "ports",
]

DEFAULT_HAZARDS = ["heat", "wildfire", "landslide"]

# ---------------------------------------------------------------------------
# Single combination runner
# ---------------------------------------------------------------------------


def run_single(
    country_iso3: str,
    asset_type: str,
    config: dict,
    hazards: list[str],
    n_outer_workers: int = 1,
) -> tuple[str, str, str]:
    """
    Run exposure assessment for one country × asset_type combination.

    Returns:
        (country_iso3, asset_type, status_message)
    """
    t0 = time.time()
    iso2 = to_iso2(country_iso3)
    tag = f"{country_iso3}/{asset_type}"

    # Output path
    out_dir = Path(config["exposure_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{country_iso3}_{asset_type}_exposure.parquet"

    if out_path.exists():
        print(f"[{tag}] Skipping — output already exists: {out_path.name}")
        return country_iso3, asset_type, "skipped"

    # Load features
    features = load_infrastructure(config["exposure_dir"], asset_type, iso2)
    if features is None or features.empty:
        print(f"[{tag}] No features found — skipping")
        return country_iso3, asset_type, "no_features"

    print(f"[{tag}] Loaded {len(features)} features")

    # Ensure osm_id column exists
    if "osm_id" not in features.columns:
        features = features.reset_index()
        if "osm_id" not in features.columns:
            features["osm_id"] = range(len(features))

    # Create composite key (osm_id, LAU) — the same OSM element can appear in multiple
    # LAU regions (e.g. a rail line crossing a municipal boundary), so osm_id alone is not
    # a unique identifier. We synthesise a feature_id string for internal use and restore
    # the original columns in the output.
    if "LAU" in features.columns:
        features["feature_id"] = (
            features["osm_id"].astype(str) + "__" + features["LAU"].astype(str)
        )
    else:
        features["feature_id"] = features["osm_id"].astype(str)

    # Deduplicate on the composite key
    n_before = len(features)
    features = features.drop_duplicates(subset="feature_id").reset_index(drop=True)
    if len(features) < n_before:
        print(
            f"[{tag}] Deduplicated {n_before - len(features)} rows on (osm_id, LAU) ({len(features)} remain)"
        )

    # Expose feature_id as osm_id to the assessment modules (they use osm_id as row key)
    features["_osm_id_orig"] = features["osm_id"]
    features["osm_id"] = features["feature_id"]

    # Accumulate output columns
    all_columns: dict = {}

    # ── Landslide ─────────────────────────────────────────────────────────
    if "landslide" in hazards:
        landslide_path = config.get("landslide_path")
        if not landslide_path or not Path(landslide_path).exists():
            print(f"[{tag}] WARNING: landslide_path not set or not found — skipping")
        else:
            try:
                enriched = assess_landslide(features, landslide_path, asset_type)
                for col in [
                    "landslide_min",
                    "landslide_avg",
                    "landslide_max",
                    "landslide_exposure",
                    "landslide_max_cat",
                ]:
                    if col in enriched.columns:
                        all_columns[col] = enriched.set_index("osm_id")[col]
            except Exception:
                print(
                    f"[{tag}] ERROR in landslide assessment:\n{traceback.format_exc()}"
                )

    # ── Heat ──────────────────────────────────────────────────────────────
    if "heat" in hazards:
        heat_dir = config.get("heat_hazard_dir")
        if not heat_dir or not Path(heat_dir).exists():
            print(f"[{tag}] WARNING: heat_hazard_dir not set or not found — skipping")
        else:
            try:
                heat_cols = assess_heat(features, heat_dir, asset_type)
                all_columns.update(heat_cols)
            except Exception:
                print(f"[{tag}] ERROR in heat assessment:\n{traceback.format_exc()}")

    # ── Wildfire ──────────────────────────────────────────────────────────
    if "wildfire" in hazards:
        wildfire_dir = config.get("wildfire_hazard_dir")
        if not wildfire_dir or not Path(wildfire_dir).exists():
            print(
                f"[{tag}] WARNING: wildfire_hazard_dir not set or not found — skipping"
            )
        else:
            try:
                wildfire_cols = assess_wildfire(features, wildfire_dir, asset_type)
                all_columns.update(wildfire_cols)
            except Exception:
                print(
                    f"[{tag}] ERROR in wildfire assessment:\n{traceback.format_exc()}"
                )

    if not all_columns:
        print(f"[{tag}] No exposure results generated — skipping output")
        return country_iso3, asset_type, "no_results"

    # ── Assemble output ───────────────────────────────────────────────────
    # Restore original osm_id (feature_id was used internally as row key)
    features["osm_id"] = features["_osm_id_orig"]
    features = features.drop(columns=["feature_id", "_osm_id_orig"], errors="ignore")

    # Base: geometry + metadata from features, indexed by feature_id for joining
    meta_cols = ["osm_id", "geometry"]
    for col in ["object_type", "LAU", "NUTS2"]:
        if col in features.columns:
            meta_cols.append(col)

    # Re-derive feature_id for the join index
    if "LAU" in features.columns:
        join_index = features["osm_id"].astype(str) + "__" + features["LAU"].astype(str)
    else:
        join_index = features["osm_id"].astype(str)

    output = features[meta_cols].copy()
    output.index = join_index

    for col_name, series in all_columns.items():
        series = series.rename(col_name)
        output = output.join(series, how="left")

    output = gpd.GeoDataFrame(output, geometry="geometry", crs=features.crs)
    output.to_parquet(out_path)

    elapsed = time.time() - t0
    n_cols = len(all_columns)
    print(f"[{tag}] Done in {elapsed:.1f}s — {n_cols} columns → {out_path.name}")
    return country_iso3, asset_type, "ok"


def _run_single_unpacked(args: tuple) -> tuple[str, str, str]:
    country, asset, config, hazards, n_outer = args
    return run_single(country, asset, config, hazards, n_outer)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    countries: list[str] | None = None,
    assets: list[str] | None = None,
    hazards: list[str] | None = None,
    config_path: str = "config.yml",
    workers: int | None = None,
) -> None:
    """
    Run the full exposure pipeline.

    Args:
        countries:    List of ISO3 codes (default: all European)
        assets:       List of asset types (default: all)
        hazards:      List of hazard types: heat, wildfire, landslide (default: all)
        config_path:  Path to YAML config file
        workers:      Number of parallel country×asset workers (None = sequential)
    """
    t_start = time.time()

    config = load_config(config_path)

    countries = [c.upper() for c in (countries or DEFAULT_COUNTRIES)]
    assets = assets or DEFAULT_ASSETS
    hazards = [h.lower() for h in (hazards or DEFAULT_HAZARDS)]

    combos = [(c, a) for c in countries for a in assets]
    effective_workers = workers if workers and workers > 1 else 1

    print(f"\n{'=' * 60}")
    print("MIRACA Exposure Pipeline")
    print(f"  Countries : {countries}")
    print(f"  Assets    : {assets}")
    print(f"  Hazards   : {hazards}")
    print(f"  Workers   : {effective_workers}")
    print(f"  Combos    : {len(combos)}")
    print(f"{'=' * 60}\n")

    results = {}

    if effective_workers == 1:
        # Sequential — full inner parallelism available per hazard module
        for country, asset in combos:
            c, a, status = run_single(
                country, asset, config, hazards, n_outer_workers=1
            )
            results[(c, a)] = status
    else:
        # Outer parallelism across country×asset combinations
        # Inner parallelism disabled to avoid memory exhaustion
        task_args = [
            (country, asset, config, hazards, effective_workers)
            for country, asset in combos
        ]
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_run_single_unpacked, args): (args[0], args[1])
                for args in task_args
            }
            for future in as_completed(futures):
                try:
                    c, a, status = future.result()
                    results[(c, a)] = status
                except Exception:
                    c, a = futures[future]
                    print(f"[{c}/{a}] UNHANDLED ERROR:\n{traceback.format_exc()}")
                    results[(c, a)] = "error"

    # Summary
    elapsed = time.time() - t_start
    ok = sum(1 for s in results.values() if s == "ok")
    skipped = sum(1 for s in results.values() if s == "skipped")
    errors = sum(1 for s in results.values() if s == "error")
    no_feat = sum(1 for s in results.values() if s == "no_features")

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  OK          : {ok}")
    print(f"  Skipped     : {skipped}")
    print(f"  No features : {no_feat}")
    print(f"  Errors      : {errors}")
    print(f"{'=' * 60}\n")

    if errors:
        failed = [f"{c}/{a}" for (c, a), s in results.items() if s == "error"]
        print(f"Failed combinations: {failed}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIRACA Exposure Pipeline — heat, wildfire, landslide"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="ISO3 country codes (default: all European)",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Asset types to process (default: all)",
    )
    parser.add_argument(
        "--hazards",
        nargs="+",
        default=None,
        choices=["heat", "wildfire", "landslide"],
        help="Hazards to run (default: heat wildfire landslide)",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to YAML config file (default: config.yml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel country×asset workers. "
            "1 = sequential with full inner parallelism (default). "
            "N>1 = N outer workers, inner parallelism disabled."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        countries=args.countries,
        assets=args.assets,
        hazards=args.hazards,
        config_path=args.config,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
