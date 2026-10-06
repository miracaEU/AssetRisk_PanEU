"""
add_river_basin_id.py

Attaches the lvl7 HydroBASINS ID (HYBAS_ID) to TENT energy network assets
(substations = points, lines = LineStrings) by spatial join against
data/basins_abs_shift_return_periods_lev07.parquet. Lets you group assets
into river-flood "events" by shared basin.

Points: joined with predicate="within"; points that fall just outside every
basin polygon (coastline/rounding mismatch) are snapped to the nearest basin
within --max-snap-dist metres.

Lines: intersected against all basin polygons; each line is assigned the
basin with the largest total intersection length. Lines with zero
intersection (e.g. fully outside coverage) are snapped to the nearest basin
centroid within --max-snap-dist metres.

Usage (cluster CLI):
    uv run python src/tools/add_river_basin_id.py \
        --basins data/basins_abs_shift_return_periods_lev07.parquet \
        --inputs /scistor/ivm/eks510/MIRACA_RISK/TENT/TENT_energy_buses_coastal_earthquake_river_windstorm.parquet \
                 /scistor/ivm/eks510/MIRACA_RISK/TENT/TENT_energy_lines_coastal_earthquake_river_windstorm.parquet \
        --overwrite
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASIN_COLS = ["HYBAS_ID", "MAIN_BAS", "PFAF_ID"]


def assign_basin_points(assets: gpd.GeoDataFrame, basins: gpd.GeoDataFrame,
                         max_snap_dist: float) -> pd.DataFrame:
    joined = gpd.sjoin(assets[["geometry"]], basins[BASIN_COLS + ["geometry"]],
                        how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    result = joined[BASIN_COLS]

    missing = result["HYBAS_ID"].isna()
    n_missing = int(missing.sum())
    if n_missing:
        print(f"[basin] {n_missing} points outside all basins, snapping to nearest "
              f"(max {max_snap_dist} m)")
        nearest = gpd.sjoin_nearest(
            assets.loc[missing, ["geometry"]],
            basins[BASIN_COLS + ["geometry"]],
            how="left", max_distance=max_snap_dist,
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        result.loc[missing, BASIN_COLS] = nearest[BASIN_COLS]

    still_missing = int(result["HYBAS_ID"].isna().sum())
    if still_missing:
        print(f"[basin] {still_missing} points have no basin within {max_snap_dist} m "
              f"(left as NaN)")
    return result


def assign_basin_lines(assets: gpd.GeoDataFrame, basins: gpd.GeoDataFrame,
                        max_snap_dist: float) -> pd.DataFrame:
    assets = assets[["geometry"]].copy()
    assets["_aid"] = assets.index

    overlay = gpd.overlay(assets, basins[BASIN_COLS + ["geometry"]],
                           how="intersection", keep_geom_type=False)
    overlay["_len"] = overlay.geometry.length

    best = (
        overlay.sort_values("_len", ascending=False)
        .drop_duplicates(subset="_aid", keep="first")
        .set_index("_aid")[BASIN_COLS]
    )

    result = pd.DataFrame(index=assets.index, columns=BASIN_COLS, dtype="float64")
    result.loc[best.index, BASIN_COLS] = best

    missing = result["HYBAS_ID"].isna()
    n_missing = int(missing.sum())
    if n_missing:
        print(f"[basin] {n_missing} lines with no intersection, snapping to nearest "
              f"basin centroid (max {max_snap_dist} m)")
        midpoints = assets.loc[missing, "geometry"].interpolate(0.5, normalized=True)
        midpoints = gpd.GeoDataFrame(geometry=midpoints, index=assets.loc[missing].index,
                                      crs=assets.crs)
        nearest = gpd.sjoin_nearest(midpoints, basins[BASIN_COLS + ["geometry"]],
                                     how="left", max_distance=max_snap_dist)
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        result.loc[missing, BASIN_COLS] = nearest[BASIN_COLS]

    still_missing = int(result["HYBAS_ID"].isna().sum())
    if still_missing:
        print(f"[basin] {still_missing} lines have no basin within {max_snap_dist} m "
              f"(left as NaN)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basins", required=True,
                        help="Path to basins_abs_shift_return_periods_lev07.parquet")
    parser.add_argument("--inputs", required=True, nargs="+",
                        help="TENT asset parquet files to annotate (points or lines)")
    parser.add_argument("--max-snap-dist", type=float, default=2000.0,
                        help="Max snap distance in metres for assets with no direct "
                             "basin match (default: 2000)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Write result back to the input file. Otherwise writes "
                             "<input>_with_basin.parquet alongside it.")
    args = parser.parse_args()

    basins = gpd.read_parquet(args.basins)
    print(f"[basin] Loaded {len(basins)} lvl7 basins from {args.basins}")

    for input_path in args.inputs:
        input_path = Path(input_path)
        print(f"[basin] Processing {input_path.name}...")
        assets = gpd.read_parquet(input_path)

        if assets.crs != basins.crs:
            assets = assets.to_crs(basins.crs)

        geom_type = assets.geometry.geom_type.iloc[0]
        if geom_type == "Point":
            basin_cols = assign_basin_points(assets, basins, args.max_snap_dist)
        elif geom_type in ("LineString", "MultiLineString"):
            basin_cols = assign_basin_lines(assets, basins, args.max_snap_dist)
        else:
            raise ValueError(f"Unsupported geometry type {geom_type} in {input_path}")

        for col in BASIN_COLS:
            assets[col] = basin_cols[col].values

        out_path = input_path if args.overwrite else (
            input_path.parent / f"{input_path.stem}_with_basin.parquet"
        )
        assets.to_parquet(out_path)
        print(f"[basin] Wrote {out_path} "
              f"({assets['HYBAS_ID'].notna().sum()}/{len(assets)} assets matched)")


if __name__ == "__main__":
    main()
