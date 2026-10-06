"""
build_basins_rp_shift.py

Rebuilds basins_abs_shift_return_periods.parquet (basin-level river RP-shift
factors used by hazard_river.py's future climate scenarios) from a HydroBASINS
shapefile + the JRC PESETA-IV disEnsemble_highExtremes_{10,100,500}.nc grids
(Mentaschi et al. 2020, 5km EPSG:3035 LAEA).

Reverse-engineered from the existing lev08 output: each output column
`{rp}_rp_change_{temp}` is the coverage-fraction-weighted zonal mean of the
`baseline_rp_shift_{temp}` variable (NOT `return_level_perc_chng_{temp}`) from
disEnsemble_highExtremes_{rp}.nc, per basin polygon. Verified against the
existing lev08 parquet: exact match on most basins, ~3% off on a handful of
small coastal/edge basins (sparse valid-pixel geometry quirk, not a method
error).

Usage:
    uv run python src/tools/build_basins_rp_shift.py \
        --hybas-shp data/hybas_eu_lev07_v1c.shp \
        --nc-dir data \
        --output data/basins_abs_shift_return_periods_lev07.parquet
"""
import argparse
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from exactextract import exact_extract

RETURN_PERIODS = [10, 100, 500]
TEMP_SCENARIOS = ["15", "20", "30", "40"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybas-shp", required=True,
                        help="Path to hybas_eu_lev{NN}_v1c.shp")
    parser.add_argument("--nc-dir", required=True,
                        help="Directory containing disEnsemble_highExtremes_{10,100,500}.nc")
    parser.add_argument("--output", required=True,
                        help="Output parquet path")
    args = parser.parse_args()

    basins = gpd.read_file(args.hybas_shp).to_crs(3035)
    print(f"[basins] Loaded {len(basins)} basins from {args.hybas_shp}")

    nc_dir = Path(args.nc_dir)
    rp_change_cols = []
    for rp in RETURN_PERIODS:
        nc_path = nc_dir / f"disEnsemble_highExtremes_{rp}.nc"
        print(f"[basins] Processing {nc_path}...")
        ds = xr.open_dataset(nc_path)
        for temp in TEMP_SCENARIOS:
            var = f"baseline_rp_shift_{temp}"
            da = ds[var].rio.write_crs("EPSG:3035").transpose("y", "x")
            results = exact_extract(da, basins, ["mean"])
            col = f"{rp}_rp_change_{temp}"
            basins[col] = [
                r["properties"]["mean"] for r in results
            ]
            rp_change_cols.append(col)
        ds.close()

    n_before = len(basins)
    basins = basins[~basins[rp_change_cols].isna().all(axis=1)].reset_index(drop=True)
    print(f"[basins] Dropped {n_before - len(basins)} basins with no valid river pixel "
          f"({len(basins)} remain)")

    basins.to_parquet(args.output)
    print(f"[basins] Wrote {args.output}")


if __name__ == "__main__":
    main()
