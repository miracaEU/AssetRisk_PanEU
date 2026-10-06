"""
prepare_energy_network.py

Convert NetworkOSM CSVs (lines + buses) to GeoParquet files for the TENT pipeline.

Reads:
    {networkosm_dir}/lines.csv   → europe_energy_lines_OSM.parquet
    {networkosm_dir}/buses.csv   → europe_energy_buses_OSM.parquet

object_type assignment:
    lines: underground="t" → "cable", else → "line"
    buses: always "substation"

CRS: EPSG:4326 (data is WGS84; TENT loader reprojects to EPSG:3035).

Usage:
    python prepare_energy_network.py
    python prepare_energy_network.py --networkosm-dir /path/to/NetworkOSM --output-dir /path/to/tent_data
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt

import yaml


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_lines(networkosm_dir: Path, output_dir: Path) -> None:
    print("[energy] Reading lines.csv...")
    df = pd.read_csv(networkosm_dir / "lines.csv", quotechar="'")
    print(f"[energy] {len(df)} lines loaded. Columns: {df.columns.tolist()}")

    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf["object_type"] = gdf["underground"].apply(
        lambda x: "cable" if str(x).lower() in ("t", "true", "1") else "line"
    )

    if "line_id" in gdf.columns:
        gdf = gdf.rename(columns={"line_id": "osm_id"})

    n_cable = (gdf["object_type"] == "cable").sum()
    n_line = (gdf["object_type"] == "line").sum()
    print(f"[energy] object_type: line={n_line}, cable={n_cable}")

    out_path = output_dir / "europe_energy_lines_OSM.parquet"
    gdf.to_parquet(str(out_path))
    print(f"[energy] Saved: {out_path} ({len(gdf)} features)")


def prepare_buses(networkosm_dir: Path, output_dir: Path) -> None:
    print("[energy] Reading buses.csv...")
    df = pd.read_csv(networkosm_dir / "buses.csv")
    print(f"[energy] {len(df)} buses loaded. Columns: {df.columns.tolist()}")

    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf["object_type"] = "substation"

    if "bus_id" in gdf.columns:
        gdf = gdf.rename(columns={"bus_id": "osm_id"})

    out_path = output_dir / "europe_energy_buses_OSM.parquet"
    gdf.to_parquet(str(out_path))
    print(f"[energy] Saved: {out_path} ({len(gdf)} features)")


def parse_args():
    cfg = _load_config()
    tent_data_dir = cfg.get("tent_data_dir", "")

    parser = argparse.ArgumentParser(
        description="Convert NetworkOSM CSVs to GeoParquet for the TENT energy pipeline"
    )
    parser.add_argument(
        "--networkosm-dir",
        default=None,
        help="Directory containing lines.csv and buses.csv (default: networkosm_dir from config.yml)",
    )
    parser.add_argument(
        "--output-dir",
        default=tent_data_dir,
        help="Output directory for parquet files (default: tent_data_dir from config.yml)",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["lines", "buses"],
        choices=["lines", "buses"],
        help="Which assets to prepare (default: both)",
    )
    return parser.parse_args(), cfg


if __name__ == "__main__":
    args, cfg = parse_args()

    networkosm_dir = Path(args.networkosm_dir) if args.networkosm_dir else Path(cfg.get("networkosm_dir", ""))
    output_dir = Path(args.output_dir)

    if not networkosm_dir or not networkosm_dir.exists():
        raise FileNotFoundError(
            f"NetworkOSM directory not found: {networkosm_dir}\n"
            "Set --networkosm-dir or add 'networkosm_dir' to config.yml"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[energy] NetworkOSM dir: {networkosm_dir}")
    print(f"[energy] Output dir:     {output_dir}")

    if "lines" in args.assets:
        prepare_lines(networkosm_dir, output_dir)

    if "buses" in args.assets:
        prepare_buses(networkosm_dir, output_dir)

    print("[energy] Done.")
