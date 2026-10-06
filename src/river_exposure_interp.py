"""
river_exposure_interp.py

Estimate future river-flood exposure by inverting the basin RP-shift curve
and interpolating between present-day RP100/RP200/RP500 floodplain extents.

Background: river hazard has no future flood maps, only basin-level
return-period shift factors (data/basins_abs_shift_return_periods.parquet,
columns "{rp}_rp_change_{temp}" for rp in {10,100,500}). hazard_river.py's
future EAD already uses this to re-grade the damage-per-RP curve. Exposure,
however, was just frozen to the present RP100 floodplain (see
figures.py:fig_future_changes docstring) because exposure extent itself
isn't reassessed per scenario.

This script instead asks: under temp_scenario, what original return period
R has the same hazard intensity as a future 100-year event? (R = inverse of
the basin's RP-shift curve evaluated at 100.) It then estimates the future
RP100 floodplain as today's extent at RP=R, interpolated between the
RP100/RP200/RP500 anchors hazard_river.py already computes
(exposure_abs_river_current, flood_extent_river_RP200/500_current).

Both interpolations (the RP-shift inversion and the RP->extent lookup) are
done linearly in log(return period) space — log(RP) vs. magnitude is the
standard near-linear flood-frequency relationship, and it keeps a single
convention for both steps.

Requires the rerun_river_rp_anchors.sh patch to have completed for a given
country x asset (adds flood_extent_river_RP200/500_current) -- files
without those columns are skipped, and their existing (frozen) summary rows
are carried through unchanged.

Does NOT write anything back into the per-country parquet files. Output is
a single patched copy of summary_statistics.csv (the long-format table
figures.py reads) with the river exposure_abs/exposure_rel future-period
rows replaced by the interpolated totals. Written alongside the original as
summary_statistics_river_interp.csv -- review and rename to
summary_statistics.csv (backing up the original) when ready to regenerate
figures.

Usage (from repo root):
  uv run python src/river_exposure_interp.py
  uv run python src/river_exposure_interp.py --countries BEL --systems rail
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import interp1d

from constants import (
    FUTURE_PERIODS,
    RIVER_FUTURE_TEMP_CODES as TEMP_CODES,
    RIVER_FUTURE_TEMP_LABELS as TEMP_LABELS,
    load_config as _load_config,
)
from hazard_river import assign_basin_ids
from merge_outputs import _col_stats

RP_SHIFT_ANCHORS = [10, 100, 500]   # anchors in basins_abs_shift_return_periods.parquet
EXPOSURE_ANCHOR_RPS = [100, 200, 500]  # anchors hazard_river.py computes per asset

EXPOSURE_COLS = {
    100: "exposure_abs_river_current",
    200: "flood_extent_river_RP200_current",
    500: "flood_extent_river_RP500_current",
}


# ---------------------------------------------------------------------------
# RP-shift inversion (per basin)
# ---------------------------------------------------------------------------

def equivalent_present_rp(basin_row: pd.Series, temp_code: str, target_new_rp: float = 100.0) -> float:
    """
    Find the present-day return period R whose hazard intensity matches a
    future {target_new_rp}-year event under temp_code, by inverting the
    basin's RP-shift curve (log-log linear between the RP10/RP100/RP500
    anchors).

    Falls back to target_new_rp itself (no shift) if the basin has no valid
    shift data, or if equivalent_rp is below the lowest exposure anchor
    (RP100) since we have no extent data to interpolate below that.
    """
    new_rps = np.array(
        [basin_row.get(f"{rp}_rp_change_{temp_code}", np.nan) for rp in RP_SHIFT_ANCHORS],
        dtype=float,
    )
    orig_rps = np.array(RP_SHIFT_ANCHORS, dtype=float)

    valid = ~np.isnan(new_rps)
    if valid.sum() < 2:
        return target_new_rp

    log_new = np.log(np.clip(new_rps[valid], 1.0, None))
    log_orig = np.log(orig_rps[valid])

    order = np.argsort(log_new)
    inv = interp1d(
        log_new[order], log_orig[order],
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    equiv_rp = float(np.exp(inv(np.log(target_new_rp))))
    return max(equiv_rp, EXPOSURE_ANCHOR_RPS[0])


# ---------------------------------------------------------------------------
# Extent lookup at an arbitrary RP (vectorised over rows)
# ---------------------------------------------------------------------------

def interp_extent_at_rp(equiv_rp: np.ndarray, v100: np.ndarray, v200: np.ndarray, v500: np.ndarray) -> np.ndarray:
    """Linear interpolation/extrapolation of extent vs. log(RP), anchored at RP100/200/500."""
    x = np.log(equiv_rp)
    x100, x200, x500 = (np.log(rp) for rp in EXPOSURE_ANCHOR_RPS)

    frac_low = (x - x100) / (x200 - x100)
    val_low = v100 + frac_low * (v200 - v100)

    frac_high = (x - x200) / (x500 - x200)
    val_high = v200 + frac_high * (v500 - v200)

    out = np.where(x <= x200, val_low, val_high)
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# Per-file processing -- returns summary_statistics.csv-shaped rows only
# ---------------------------------------------------------------------------

REQUIRED_COLS = list(EXPOSURE_COLS.values()) + ["asset_size"]


def process_file(path: Path, iso3: str, system: str, basin_data: gpd.GeoDataFrame) -> list[dict] | None:
    schema_cols = set(pq.ParquetFile(path).schema.names)
    if not all(c in schema_cols for c in EXPOSURE_COLS.values()):
        return None  # no river hazard, or rerun_river_rp_anchors.sh hasn't landed yet

    df = gpd.read_parquet(path, columns=REQUIRED_COLS + ["geometry"])

    v100 = df[EXPOSURE_COLS[100]].fillna(0.0).to_numpy()
    v200 = df[EXPOSURE_COLS[200]].fillna(0.0).to_numpy()
    v500 = df[EXPOSURE_COLS[500]].fillna(0.0).to_numpy()
    asset_size = df["asset_size"].to_numpy()
    curr_abs_sum = v100.sum()
    curr_rel = np.where(asset_size > 0, v100 / asset_size, 0.0)
    curr_rel_sum = curr_rel.sum()

    basin_ids = assign_basin_ids(df, basin_data)
    unique_basins = basin_ids.dropna().unique()

    rows = []
    for temp_code, temp_label in zip(TEMP_CODES, TEMP_LABELS):
        equiv_rp_by_basin = {
            b: equivalent_present_rp(basin_data.loc[b], temp_code)
            for b in unique_basins
        }
        equiv_rp = basin_ids.map(equiv_rp_by_basin).fillna(100.0).to_numpy()

        new_abs = interp_extent_at_rp(equiv_rp, v100, v200, v500)
        with np.errstate(divide="ignore", invalid="ignore"):
            new_rel = np.where(asset_size > 0, new_abs / asset_size, 0.0)

        for kind, values, curr_sum in [("abs", new_abs, curr_abs_sum), ("rel", new_rel, curr_rel_sum)]:
            row = {"country": iso3, "system": system, "hazard": "river",
                   "period": temp_label, "metric_type": f"exposure_{kind}", "metric_bound": "mean"}
            row.update(_col_stats(pd.Series(values)))
            fut_sum = row["sum"]
            row["change_abs_sum"] = round(fut_sum - curr_sum, 4)
            row["change_pct_sum"] = round((fut_sum - curr_sum) / curr_sum * 100, 2) if curr_sum > 0 else None
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Patch summary_statistics.csv with interpolated future river exposure"
    )
    parser.add_argument("--output-dir", default=None,
                        help="Override MIRACA_OUTPUT_FULL directory (where parquets + summary_statistics.csv live)")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Only process these ISO3 countries")
    parser.add_argument("--systems", nargs="+", default=None,
                        help="Only process these asset systems")
    args = parser.parse_args()

    config = _load_config()
    repo_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "MIRACA_OUTPUT_FULL"

    basin_data = gpd.read_parquet(Path(config["basin_data_path"]))
    summary = pd.read_csv(output_dir / "summary_statistics.csv")

    files = sorted(output_dir.glob("*.parquet"))
    if args.countries:
        files = [f for f in files if f.stem.split("_", 1)[0] in args.countries]
    if args.systems:
        files = [f for f in files if f.stem.split("_", 1)[1] in args.systems]

    print(f"Output dir: {output_dir}")
    print(f"Basin data: {config['basin_data_path']} ({len(basin_data)} basins)")
    print(f"Candidate files: {len(files)}\n")

    new_rows, n_processed, n_skipped = [], 0, 0
    for path in files:
        iso3, system = path.stem.split("_", 1)
        rows = process_file(path, iso3, system, basin_data)
        if rows is None:
            n_skipped += 1
            continue
        n_processed += 1
        new_rows.extend(rows)
        for r in rows:
            print(f"  {iso3}_{system} {r['metric_type']} {r['period']}: "
                  f"sum={r['sum']:.2f} ({r['change_pct_sum']}%)")

    print(f"\nProcessed: {n_processed}, skipped (no flood_extent anchors yet): {n_skipped}")

    new_df = pd.DataFrame(new_rows)
    keys = ["country", "system", "hazard", "period", "metric_type", "metric_bound"]
    replace_mask = (
        (summary["hazard"] == "river") &
        (summary["metric_type"].isin(["exposure_abs", "exposure_rel"])) &
        (summary["period"].isin(FUTURE_PERIODS)) &
        summary.set_index(keys).index.isin(new_df.set_index(keys).index)
    )
    patched = pd.concat([summary[~replace_mask], new_df], ignore_index=True)

    out_path = output_dir / "summary_statistics_river_interp.csv"
    patched.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({replace_mask.sum()} rows replaced, {len(new_df)} new river future rows)")
    print("Review, then rename to summary_statistics.csv (backing up the original) to use in figures.py.")


if __name__ == "__main__":
    main()
