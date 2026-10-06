"""
merge_outputs.py

Merge hazard (risk pipeline) and exposure pipeline outputs into a single
parquet file per country × asset type, and produce a summary statistics CSV.

Post-processing steps:
  - Roads: remap object_type to road_class (5 classes)
  - Heat/wildfire: convert days → m/m²/units using country-specific p90 thresholds
  - Heat/wildfire: fix coverage gaps in future projections
  - Recompute exposure_rel = exposure_abs / asset_size

Usage (from repo root):
  uv run python src/merge_outputs.py
  uv run python src/merge_outputs.py --overwrite
  uv run python src/merge_outputs.py --output-dir /custom/path
"""

import argparse
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from constants import (
    EAD_HAZARDS,
    EXPOSURE_HAZARDS,
    FUTURE_PERIODS,
    ROAD_CLASS_MAP,
    load_config as _load_config,
)


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

SHARED_COLS = {"osm_id", "geometry", "object_type", "LAU", "NUTS2", "asset_size"}

# Percentile used for country-specific heat/wildfire thresholds.
# p90 of current-period days: only infrastructure above the country's
# 90th percentile of hazard intensity is considered "exposed".
THRESHOLD_PERCENTILE = 90

# Fallback threshold if a country has no exposed features in the current period
FALLBACK_THRESHOLD = 1.0


# ──────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────

def _parse_stem(stem: str, suffix: str) -> tuple[str, str] | None:
    tag = f"_{suffix}"
    if not stem.endswith(tag):
        return None
    base = stem[: -len(tag)]
    parts = base.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else None


def discover_pairs(
    hazard_dir: Path,
    exposure_dir: Path,
) -> list[dict]:
    hazard_files = {f.stem: f for f in hazard_dir.glob("*_hazards.parquet")}
    exposure_files = {f.stem: f for f in exposure_dir.glob("*_exposure.parquet")}

    combos: dict[tuple[str, str], dict] = {}

    for stem, path in hazard_files.items():
        parsed = _parse_stem(stem, "hazards")
        if parsed:
            combos[parsed] = {"iso3": parsed[0], "system": parsed[1],
                              "hazard_path": path, "exposure_path": None}

    for stem, path in exposure_files.items():
        parsed = _parse_stem(stem, "exposure")
        if parsed:
            if parsed in combos:
                combos[parsed]["exposure_path"] = path
            else:
                combos[parsed] = {"iso3": parsed[0], "system": parsed[1],
                                  "hazard_path": None, "exposure_path": path}

    return sorted(combos.values(), key=lambda x: (x["iso3"], x["system"]))


# ──────────────────────────────────────────────────────────────────────────
# Pass 1: Compute country-specific p90 thresholds for heat/wildfire
# ──────────────────────────────────────────────────────────────────────────

_THRESH_COL_PAT = re.compile(
    r"^(exposure_(?:abs|rel)_(?:min_|max_)?)heat_(\d+C)_(.+)$"
)


def _select_heat_threshold(
    gdf: gpd.GeoDataFrame,
    heat_threshold: str,
) -> gpd.GeoDataFrame:
    """
    Rename exposure_abs[_min|_max]_heat_{heat_threshold}_* columns to generic
    exposure_abs[_min|_max]_heat_* names and drop columns for other thresholds.
    """
    rename_map: dict[str, str] = {}
    drop_cols: list[str] = []
    for col in gdf.columns:
        m = _THRESH_COL_PAT.match(col)
        if not m:
            continue
        prefix, thresh, rest = m.group(1), m.group(2), m.group(3)
        if thresh == heat_threshold:
            rename_map[col] = f"{prefix}heat_{rest}"
        else:
            drop_cols.append(col)
    if rename_map or drop_cols:
        gdf = gdf.rename(columns=rename_map).drop(columns=drop_cols, errors="ignore")
    return gdf


def compute_country_thresholds(
    pairs: list[dict],
    percentile: int = THRESHOLD_PERCENTILE,
    heat_threshold: str = "30C",
) -> dict[str, dict[str, float]]:
    """
    Scan exposure files to compute p90 of current-period days per country
    for heat and wildfire.

    heat_threshold: temperature threshold label used in column names (e.g. '30C').

    Returns: {country_iso3: {"heat": threshold, "wildfire": threshold}}
    """
    print(f"Computing country-specific heat/wildfire thresholds (heat={heat_threshold})...")

    country_days: dict[str, dict[str, list]] = {}

    for combo in pairs:
        iso3 = combo["iso3"]
        exposure_path = combo["exposure_path"]
        if exposure_path is None:
            continue

        if iso3 not in country_days:
            country_days[iso3] = {"heat": [], "wildfire": []}

        try:
            gdf = gpd.read_parquet(exposure_path)
            # Heat: use threshold-specific column; fall back to legacy name
            heat_col = f"exposure_abs_heat_{heat_threshold}_current"
            if heat_col not in gdf.columns:
                heat_col = "exposure_abs_heat_current"
            if heat_col in gdf.columns:
                vals = gdf[heat_col].dropna()
                positive = vals[vals > 0].values
                country_days[iso3]["heat"].extend(positive.tolist())

            # Wildfire: no temperature threshold in column name
            wf_col = "exposure_abs_wildfire_current"
            if wf_col in gdf.columns:
                vals = gdf[wf_col].dropna()
                positive = vals[vals > 0].values
                country_days[iso3]["wildfire"].extend(positive.tolist())
        except Exception as e:
            print(f"  Warning: could not read {exposure_path.name}: {e}")

    thresholds: dict[str, dict[str, float]] = {}
    for iso3, hazard_vals in sorted(country_days.items()):
        thresholds[iso3] = {}
        for hazard_name, vals in hazard_vals.items():
            if len(vals) > 0:
                p90 = float(np.percentile(vals, percentile))
                thresholds[iso3][hazard_name] = max(p90, FALLBACK_THRESHOLD)
            else:
                thresholds[iso3][hazard_name] = FALLBACK_THRESHOLD

        heat_t = thresholds[iso3].get("heat", FALLBACK_THRESHOLD)
        wild_t = thresholds[iso3].get("wildfire", FALLBACK_THRESHOLD)
        print(f"  {iso3}: heat={heat_t:.1f} days, wildfire={wild_t:.1f} days")

    return thresholds


# ──────────────────────────────────────────────────────────────────────────
# Merge logic
# ──────────────────────────────────────────────────────────────────────────

def merge_single(
    hazard_path: Path | None,
    exposure_path: Path | None,
    output_path: Path,
    iso3: str,
    system: str,
    country_thresholds: dict[str, dict[str, float]],
    heat_threshold: str = "35C",
) -> tuple[str, gpd.GeoDataFrame]:
    """Merge one pair of hazard + exposure files with post-processing."""

    # --- Load files ---
    if hazard_path is None:
        hazard = gpd.read_parquet(exposure_path)
        status = "exposure_only"
    elif exposure_path is None:
        hazard = gpd.read_parquet(hazard_path)
        status = "hazard_only"
    else:
        hazard = gpd.read_parquet(hazard_path)
        exposure = gpd.read_parquet(exposure_path)

        exposure_only_cols = [c for c in exposure.columns
                              if c not in SHARED_COLS and c not in hazard.columns]

        if exposure_only_cols:
            if (len(hazard) == len(exposure)
                    and (hazard["osm_id"].values == exposure["osm_id"].values).all()):
                new_data = pd.DataFrame(
                    {col: exposure[col].values for col in exposure_only_cols},
                    index=hazard.index,
                )
                hazard = gpd.GeoDataFrame(
                    pd.concat([hazard, new_data], axis=1),
                    geometry=hazard.geometry.name, crs=hazard.crs,
                )
            else:
                join_keys = (["osm_id", "LAU"]
                             if ("LAU" in hazard.columns and "LAU" in exposure.columns)
                             else ["osm_id"])
                hazard = hazard.merge(
                    exposure[join_keys + exposure_only_cols],
                    on=join_keys, how="left",
                )

        status = "merged"

    # --- Post-processing ---

    # Select heat temperature threshold — rename chosen columns to generic names
    hazard = _select_heat_threshold(hazard, heat_threshold)

    # Roads: remap object_type to road_class (Issue 2)
    if system == "roads" and "object_type" in hazard.columns:
        hazard["object_type"] = hazard["object_type"].map(
            lambda x: ROAD_CLASS_MAP.get(x, x.lower().replace(" ", "_"))
        )

    # Heat/wildfire: country-specific threshold conversion
    if "asset_size" in hazard.columns:
        thresholds = country_thresholds.get(iso3, {})

        for hazard_name in ["heat", "wildfire"]:
            curr_col = f"exposure_abs_{hazard_name}_current"
            if curr_col not in hazard.columns:
                continue

            threshold = thresholds.get(hazard_name, FALLBACK_THRESHOLD)

            # Identify features exposed in current period
            currently_exposed = hazard[curr_col] >= threshold

            # Fix coverage gaps in future periods
            future_cols = [c for c in hazard.columns
                           if (c.startswith(f"exposure_abs_{hazard_name}_") or
                               c.startswith(f"exposure_abs_min_{hazard_name}_") or
                               c.startswith(f"exposure_abs_max_{hazard_name}_"))
                           and c != curr_col]
            for fcol in future_cols:
                coverage_gap = currently_exposed & (hazard[fcol] == 0)
                if coverage_gap.any():
                    hazard.loc[coverage_gap, fcol] = hazard.loc[coverage_gap, curr_col]

            # Apply threshold: >= threshold → asset_size, else 0
            all_cols = [curr_col] + future_cols
            for col in all_cols:
                hazard[col] = (hazard[col] >= threshold).astype(float) * hazard["asset_size"]

        # Recompute exposure_rel for all exposure_abs columns (concat once — avoids fragmentation)
        nonzero = hazard["asset_size"] > 0
        rel_parts = {}
        for col in list(hazard.columns):
            if col.startswith("exposure_abs_"):
                rel_col = col.replace("exposure_abs_", "exposure_rel_")
                s = pd.Series(0.0, index=hazard.index)
                s[nonzero] = hazard.loc[nonzero, col] / hazard.loc[nonzero, "asset_size"]
                rel_parts[rel_col] = s
        if rel_parts:
            stale = [c for c in hazard.columns if c.startswith("exposure_rel_")]
            if stale:
                hazard = hazard.drop(columns=stale)
            hazard = gpd.GeoDataFrame(
                pd.concat([hazard, pd.DataFrame(rel_parts, index=hazard.index)], axis=1),
                geometry=hazard.geometry.name, crs=hazard.crs,
            )

    hazard.to_parquet(output_path)
    return status, hazard


# ──────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────

def _col_stats(series: pd.Series) -> dict:
    nonzero = series[series > 0]
    return {
        "count": int(series.count()),
        "n_exposed": int((series > 0).sum()),
        "pct_exposed": round((series > 0).sum() / max(len(series), 1) * 100, 2),
        "sum": round(float(series.sum()), 4),
        "mean": round(float(series.mean()), 6),
        "median": round(float(series.median()), 6),
        "std": round(float(series.std()), 6),
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
        "p25": round(float(series.quantile(0.25)), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "p95": round(float(series.quantile(0.95)), 6),
        "mean_exposed": round(float(nonzero.mean()), 6) if len(nonzero) > 0 else 0.0,
    }


def compute_summary(
    gdf: gpd.GeoDataFrame,
    iso3: str,
    system: str,
) -> list[dict]:
    rows = []

    for hazard in EAD_HAZARDS:
        for bound in ["mid", "min", "max"]:
            col = f"EAD_{bound}_{hazard}_current"
            if col not in gdf.columns:
                continue
            row = {"country": iso3, "system": system,
                   "hazard": hazard, "period": "current",
                   "metric_type": "EAD", "metric_bound": bound}
            row.update(_col_stats(gdf[col]))
            rows.append(row)

        for period in FUTURE_PERIODS:
            for bound in ["mid", "min", "max"]:
                col = f"EAD_{bound}_{hazard}_{period}"
                if col not in gdf.columns:
                    continue
                row = {"country": iso3, "system": system,
                       "hazard": hazard, "period": period,
                       "metric_type": "EAD", "metric_bound": bound}
                row.update(_col_stats(gdf[col]))

                curr_col = f"EAD_{bound}_{hazard}_current"
                if curr_col in gdf.columns:
                    curr_sum = gdf[curr_col].sum()
                    fut_sum = gdf[col].sum()
                    row["change_abs_sum"] = round(fut_sum - curr_sum, 4)
                    row["change_pct_sum"] = (
                        round((fut_sum - curr_sum) / curr_sum * 100, 2)
                        if curr_sum > 0 else None
                    )
                rows.append(row)

    for hazard in EXPOSURE_HAZARDS:
        for kind in ["abs", "rel"]:
            col = f"exposure_{kind}_{hazard}_current"
            if col not in gdf.columns:
                continue
            row = {"country": iso3, "system": system,
                   "hazard": hazard, "period": "current",
                   "metric_type": f"exposure_{kind}", "metric_bound": "mean"}
            row.update(_col_stats(gdf[col]))
            rows.append(row)

            for period in FUTURE_PERIODS:
                fcol = f"exposure_{kind}_{hazard}_{period}"
                if fcol not in gdf.columns:
                    continue
                row = {"country": iso3, "system": system,
                       "hazard": hazard, "period": period,
                       "metric_type": f"exposure_{kind}", "metric_bound": "mean"}
                row.update(_col_stats(gdf[fcol]))

                curr_sum = gdf[col].sum()
                fut_sum = gdf[fcol].sum()
                row["change_abs_sum"] = round(fut_sum - curr_sum, 4)
                row["change_pct_sum"] = (
                    round((fut_sum - curr_sum) / curr_sum * 100, 2)
                    if curr_sum > 0 else None
                )
                rows.append(row)

    return rows


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge MIRACA hazard + exposure outputs into MIRACA_OUTPUT/"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: MIRACA_OUTPUT/ next to config.yml)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-merge even if output file already exists",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Only re-merge these systems (e.g. --systems rail). "
             "Other systems are read for summary stats but not re-merged.",
    )
    parser.add_argument(
        "--countries", nargs="+", default=None,
        help="Only re-merge these ISO3 countries (e.g. --countries GBR). "
             "Other countries are read for summary stats but not re-merged.",
    )
    parser.add_argument(
        "--use-cached-thresholds", action="store_true",
        help="Skip recomputing country heat/wildfire p90 thresholds (the slowest "
             "step -- scans every exposure file). Loads them from the existing "
             "country_thresholds.csv in the output dir instead. Only safe when "
             "exposure data for heat/wildfire hasn't changed since that file was "
             "written (e.g. a landslide/coastal-only rerun).",
    )
    args = parser.parse_args()

    config = _load_config()
    repo_root = Path(__file__).parent.parent

    hazard_dir = Path(config["output_dir"])
    exposure_dir = Path(config["exposure_output_dir"])

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "MIRACA_OUTPUT_FULL"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hazard dir:   {hazard_dir}")
    print(f"Exposure dir: {exposure_dir}")
    print(f"Output dir:   {output_dir}\n")

    pairs = discover_pairs(hazard_dir, exposure_dir)
    if not pairs:
        print("No files found to merge.")
        return

    print(f"Found {len(pairs)} country × asset combinations\n")

    # Pass 1: compute country-specific thresholds from exposure files
    heat_threshold = config.get("heat_threshold", "35C")
    threshold_path = output_dir / "country_thresholds.csv"

    if args.use_cached_thresholds and threshold_path.exists():
        print(f"Loading cached thresholds from {threshold_path} (skipping rescan)\n")
        cached_df = pd.read_csv(threshold_path)
        country_thresholds = {
            row["country"]: {
                "heat": row["heat_threshold_days"],
                "wildfire": row["wildfire_threshold_days"],
            }
            for _, row in cached_df.iterrows()
        }
    else:
        if args.use_cached_thresholds:
            print(f"--use-cached-thresholds set but {threshold_path} not found -- computing fresh.\n")
        country_thresholds = compute_country_thresholds(pairs, heat_threshold=heat_threshold)
        print()

        # Save thresholds for documentation
        threshold_rows = []
        for iso3, thresholds in sorted(country_thresholds.items()):
            threshold_rows.append({
                "country": iso3,
                "heat_threshold_days": thresholds.get("heat", FALLBACK_THRESHOLD),
                "wildfire_threshold_days": thresholds.get("wildfire", FALLBACK_THRESHOLD),
                "percentile": THRESHOLD_PERCENTILE,
            })
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(threshold_path, index=False)
        print(f"Thresholds saved to {threshold_path}\n")

    # Pass 2: merge and apply thresholds
    t0 = time.time()
    counts = {"merged": 0, "hazard_only": 0, "exposure_only": 0, "skipped": 0}
    all_summary_rows: list[dict] = []

    system_filter  = set(args.systems)   if args.systems   else None
    country_filter = set(args.countries) if args.countries else None

    for combo in pairs:
        iso3, system = combo["iso3"], combo["system"]
        out_path = output_dir / f"{iso3}_{system}.parquet"

        force_remerge = args.overwrite and (system_filter  is None or system in system_filter) \
                                       and (country_filter is None or iso3   in country_filter)
        if out_path.exists() and not force_remerge:
            gdf = gpd.read_parquet(out_path)
            all_summary_rows.extend(compute_summary(gdf, iso3, system))
            print(f"  {iso3}/{system}: skipped (exists) — stats collected")
            counts["skipped"] += 1
            continue

        status, gdf = merge_single(
            combo["hazard_path"], combo["exposure_path"], out_path,
            iso3, system, country_thresholds, heat_threshold=heat_threshold,
        )
        counts[status] += 1
        all_summary_rows.extend(compute_summary(gdf, iso3, system))
        print(f"  {iso3}/{system}: {status} — {len(gdf)} features, {len(gdf.columns)} columns")

    # Write summary statistics
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary statistics: {summary_path} ({len(summary_df)} rows)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Merged:        {counts['merged']}")
    print(f"  Hazard only:   {counts['hazard_only']}")
    print(f"  Exposure only: {counts['exposure_only']}")
    print(f"  Skipped:       {counts['skipped']}")


if __name__ == "__main__":
    main()
