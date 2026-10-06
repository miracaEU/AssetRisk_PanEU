"""
compute_subsystem_summary.py

Read all merged output parquets, group by country × system × object_type × hazard,
and compute EAD + exposure statistics.

Output: subsystem_summary.csv — same structure as summary_statistics.csv
        but with an extra 'object_type' column (raw, ungrouped).

Usage (from repo root on HPC):
    uv run python src/compute_subsystem_summary.py
    uv run python src/compute_subsystem_summary.py \\
        --input-dir /scistor/ivm/eks510/projects/AssetRisk_PanEU/MIRACA_OUTPUT \\
        --output /scistor/ivm/eks510/projects/AssetRisk_PanEU/MIRACA_OUTPUT/subsystem_summary.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from constants import FUTURE_PERIODS

# Hazard loop order kept local: it sets the row order of subsystem_summary.csv
EAD_HAZARDS      = ["coastal", "earthquake", "river", "windstorm"]
EXPOSURE_HAZARDS = ["heat", "landslide", "wildfire", "coastal", "earthquake", "river", "windstorm"]

KNOWN_SYSTEMS = {
    "airports", "education", "gas", "healthcare", "oil",
    "ports", "power", "rail", "roads", "telecom",
}


def _col_stats(series: pd.Series) -> dict:
    s = series.dropna()
    n = len(s)
    return {
        "count":       n,
        "sum":         round(float(s.sum()), 4) if n > 0 else 0.0,
        "mean":        round(float(s.mean()), 6) if n > 0 else None,
        "pct_exposed": round(float((s > 0).sum() / n * 100), 2) if n > 0 else None,
    }


def _parse_stem(stem: str):
    """Return (iso3, system) from filename stem, or None."""
    stem = stem.replace("_hazards", "")
    # Match ISO3_system pattern
    m = re.match(r"^([A-Z]{2,3})_(.+)$", stem)
    if not m:
        return None
    iso3, system = m.group(1), m.group(2)
    if system not in KNOWN_SYSTEMS:
        return None
    return iso3, system


def process_file(path: Path) -> list[dict]:
    parsed = _parse_stem(path.stem)
    if parsed is None:
        return []
    iso3, system = parsed

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  Skip {path.name}: {e}")
        return []

    if "object_type" not in df.columns:
        print(f"  No object_type: {path.name}")
        return []

    rows = []
    for obj_type, grp in df.groupby("object_type", dropna=False):
        obj_type = str(obj_type) if obj_type is not None else "unknown"
        base = {"country": iso3, "system": system, "object_type": obj_type}

        # EAD — current + future
        for hazard in EAD_HAZARDS:
            for bound in ["mid", "min", "max"]:
                col = f"EAD_{bound}_{hazard}_current"
                if col not in grp.columns:
                    continue
                row = {**base, "hazard": hazard, "period": "current",
                       "metric_type": "EAD", "metric_bound": bound}
                row.update(_col_stats(grp[col]))
                rows.append(row)

            for period in FUTURE_PERIODS:
                for bound in ["mid", "min", "max"]:
                    col = f"EAD_{bound}_{hazard}_{period}"
                    if col not in grp.columns:
                        continue
                    row = {**base, "hazard": hazard, "period": period,
                           "metric_type": "EAD", "metric_bound": bound}
                    row.update(_col_stats(grp[col]))
                    rows.append(row)

        # Exposure — current + future
        for hazard in EXPOSURE_HAZARDS:
            for kind in ["abs", "rel"]:
                col = f"exposure_{kind}_{hazard}_current"
                if col not in grp.columns:
                    continue
                row = {**base, "hazard": hazard, "period": "current",
                       "metric_type": f"exposure_{kind}", "metric_bound": "mean"}
                row.update(_col_stats(grp[col]))
                rows.append(row)

            for period in FUTURE_PERIODS:
                for kind in ["abs", "rel"]:
                    col = f"exposure_{kind}_{hazard}_{period}"
                    if col not in grp.columns:
                        continue
                    row = {**base, "hazard": hazard, "period": period,
                           "metric_type": f"exposure_{kind}", "metric_bound": "mean"}
                    row.update(_col_stats(grp[col]))
                    rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Compute subsystem-level EAD summary")
    parser.add_argument("--input-dir", default=None,
                        help="Directory containing merged *_hazards.parquet files")
    parser.add_argument("--output", default=None,
                        help="Output CSV path")
    parser.add_argument("--pattern", default="*.parquet",
                        help="Glob pattern for input files (default: *.parquet)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    input_dir = Path(args.input_dir) if args.input_dir else repo_root / "MIRACA_OUTPUT"
    output_path = (Path(args.output) if args.output
                   else input_dir / "subsystem_summary.csv")

    all_files = sorted(input_dir.glob(args.pattern))
    files = [f for f in all_files
             if f.suffix == ".parquet" and "summary" not in f.stem]
    print(f"Input dir : {input_dir}")
    print(f"Pattern   : {args.pattern}")
    print(f"Files     : {len(files)}")
    print(f"Output    : {output_path}\n")

    all_rows = []
    for i, f in enumerate(files, 1):
        rows = process_file(f)
        all_rows.extend(rows)
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  rows so far: {len(all_rows)}")

    if not all_rows:
        print("No rows extracted — check input dir and pattern.")
        return

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(output_path, index=False)

    print(f"\nDone. {len(out_df):,} rows → {output_path}")
    print("\nObject types per system:")
    for sys, grp in out_df.groupby("system"):
        otypes = sorted(grp["object_type"].unique())
        print(f"  {sys:12s}: {otypes}")


if __name__ == "__main__":
    main()
