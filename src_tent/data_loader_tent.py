"""
data_loader_tent.py

Loader for the TENT (Trans-European Network for Transport) edge datasets.

Replaces the per-country exposure loader with a Europe-wide approach:
  - Loads full parquet files (one per transport mode)
  - Derives `object_type` from tag_highway (roads) or tag_railway (rail)
  - Reprojects from EPSG:3857 to EPSG:3035
  - Chunks by object_type for manageable processing

Data path structure:
    {data_dir}/europe_road_edges_TENT.parquet
    {data_dir}/europe_railways_edges_TENT.parquet
"""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------------

TENT_FILES = {
    "roads": "europe_road_edges_TENT.parquet",
    "rail": "europe_railways_edges_TENT.parquet",
    "airports": "europe_airport_areas_TENT.parquet",
    "ports": "europe_port_areas_TENT.parquet",  # fallback if gpkg absent
    "iww": "europe_IWW_nodes_TENT.parquet",
    "energy_lines": "europe_energy_lines_OSM.parquet",
    "energy_buses": "europe_energy_buses_OSM.parquet",
}

# GeoPackage sources — preferred over parquet when available (correct polygon geometry)
TENT_GPKG_FILES = {
    "ports": ("europe_port_areas_TENT.gpkg", "port_areas"),
}

# Column used to derive object_type per asset
OBJECT_TYPE_SOURCE = {
    "roads": "tag_highway",
    "rail": "tag_railway",
    "airports": "aeroway",
    "ports": "land_use",          # normalized in load_tent_edges
    "iww": "feature",             # filtered + mapped to "General Cargo" in load_tent_edges
    "energy_lines": "underground", # "t"→cable, else→line
    "energy_buses": "object_type", # already set to "substation" in parquet
}

# IWW feature types to keep (ports + intermodal nodes only)
IWW_RELEVANT_FEATURES = {"port", "Trimodal", "Road-IWW intermodal"}

# Source CRS of all TENT parquets
TENT_SOURCE_CRS = "EPSG:3857"

# Target CRS expected by the hazard pipeline
TENT_TARGET_CRS = "EPSG:3035"


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_tent_edges(
    data_dir: Union[str, Path],
    asset_type: str,
    target_crs: str = TENT_TARGET_CRS,
    corridors_only: bool = False,
    country_iso3: Optional[str] = None,
    corridor_id: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Load a full TENT edges parquet and prepare it for the risk pipeline.

    Steps:
      1. Read parquet
      2. Derive `object_type` from the appropriate tag column
      3. Reproject from EPSG:3857 → target_crs (default EPSG:3035)
      4. Optionally filter to TENT corridor edges only

    Args:
        data_dir:       Directory containing the TENT parquet files
        asset_type:     'roads' or 'rail'
        target_crs:     Target CRS string (default 'EPSG:3035')
        corridors_only: If True, keep only edges that belong to a TEN-T corridor
                        (CORRIDORS column is not null/nan)

    Returns:
        GeoDataFrame ready for the hazard pipeline, or None if file not found.
    """
    data_dir = Path(data_dir)
    filename = TENT_FILES.get(asset_type)
    if filename is None:
        print(f"[TENT loader] Unknown asset type: {asset_type}. "
              f"Available: {list(TENT_FILES.keys())}")
        return None

    filepath = data_dir / filename
    if not filepath.exists():
        print(f"[TENT loader] File not found: {filepath}")
        return None

    # Prefer gpkg over parquet when available (e.g. ports with correct polygon geometry)
    if asset_type in TENT_GPKG_FILES:
        gpkg_name, gpkg_layer = TENT_GPKG_FILES[asset_type]
        gpkg_path = data_dir / gpkg_name
        if gpkg_path.exists():
            print(f"[TENT loader] Loading {gpkg_name} (layer: {gpkg_layer})...")
            gdf = gpd.read_file(gpkg_path, layer=gpkg_layer)
            print(f"[TENT loader] Loaded {len(gdf)} features from gpkg ({asset_type})")
            # Keep only polygon geometries — point features in port gpkg are incomplete/ignored
            poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            n_dropped = (~poly_mask).sum()
            if n_dropped > 0:
                print(f"[TENT loader] Dropping {n_dropped} non-polygon features (Points etc.) from {asset_type}")
                gdf = gdf[poly_mask].reset_index(drop=True)
        else:
            print(f"[TENT loader] gpkg not found at {gpkg_path}, falling back to parquet...")
            gdf = gpd.read_parquet(filepath)
            print(f"[TENT loader] Loaded {len(gdf)} features ({asset_type})")
    else:
        print(f"[TENT loader] Loading {filepath.name}...")
        gdf = gpd.read_parquet(filepath)
        print(f"[TENT loader] Loaded {len(gdf)} edges ({asset_type})")

    # --- Derive object_type ---
    source_col = OBJECT_TYPE_SOURCE[asset_type]
    if source_col not in gdf.columns:
        raise ValueError(
            f"Expected column '{source_col}' not found in {filepath.name}. "
            f"Available columns: {list(gdf.columns)}"
        )

    if asset_type == "ports":
        # Normalize land_use case (industry→Industry, refinery→Refinery) and fill NaN
        from constants import PORT_LAND_USE_NORMALIZE
        gdf["object_type"] = (
            gdf[source_col]
            .map(lambda x: PORT_LAND_USE_NORMALIZE.get(x, x) if pd.notna(x) else "Other")
        )
        # Fill NaN area with median so asset_size computation doesn't break
        median_area = pd.to_numeric(gdf["area"], errors="coerce").median()
        gdf["area"] = pd.to_numeric(gdf["area"], errors="coerce").fillna(median_area)
    elif asset_type == "iww":
        # Keep only port + intermodal nodes; map all to General Cargo curve
        n_before = len(gdf)
        gdf = gdf[gdf[source_col].isin(IWW_RELEVANT_FEATURES)].copy()
        print(f"[TENT loader] IWW filter: {len(gdf)}/{n_before} nodes retained (port + intermodal)")
        gdf["object_type"] = "General Cargo"
    elif asset_type == "energy_lines":
        # underground="t" → cable (buried), else → line (overhead)
        gdf["object_type"] = gdf[source_col].apply(
            lambda x: "cable" if str(x).lower() in ("t", "true", "1") else "line"
        )
    elif asset_type == "energy_buses":
        # object_type already set to "substation" in parquet
        pass
    else:
        gdf["object_type"] = gdf[source_col]

    print(f"[TENT loader] object_type values: {gdf['object_type'].unique().tolist()}")

    # --- Filter corridors if requested (not applicable to energy assets) ---
    if corridors_only or corridor_id is not None:
        if asset_type in ("energy_lines", "energy_buses"):
            print(f"[TENT loader] Corridor filter not applicable to {asset_type}, skipping.")
        elif corridor_id is not None:
            n_before = len(gdf)
            mask = gdf["CORRIDORS"].astype(str).str.contains(corridor_id, na=False)
            gdf = gdf[mask].copy()
            print(f"[TENT loader] Corridor '{corridor_id}' filter: {len(gdf)}/{n_before} edges retained")
        else:
            # CORRIDORS column may contain actual NaN, string 'nan', or None
            mask = gdf["CORRIDORS"].notna() & (gdf["CORRIDORS"] != "nan")
            n_before = len(gdf)
            gdf = gdf[mask].copy()
            print(f"[TENT loader] Corridor filter: {len(gdf)}/{n_before} edges retained")

    # --- Filter by country ---
    if country_iso3 is not None:
        iso = country_iso3.upper()
        if asset_type == "airports":
            from data_loader import to_iso2
            iso2 = to_iso2(iso)
            mask = gdf["Country_code"] == iso2
        elif asset_type == "ports" and "iso3" in gdf.columns:
            mask = gdf["iso3"] == iso
        elif asset_type == "energy_buses" and "country" in gdf.columns:
            from data_loader import to_iso2
            iso2 = to_iso2(iso)
            mask = gdf["country"] == iso2
        elif asset_type == "energy_lines":
            print(f"[TENT loader] Country filter not supported for energy_lines, skipping.")
            mask = pd.Series(True, index=gdf.index)
        else:
            mask = (gdf["from_iso_a3"] == iso) | (gdf["to_iso_a3"] == iso)
        n_before = len(gdf)
        gdf = gdf[mask].copy()
        print(f"[TENT loader] Country filter '{iso}': {len(gdf)}/{n_before} features retained")

    # --- Reproject ---
    if gdf.crs is not None and str(gdf.crs) != target_crs:
        print(f"[TENT loader] Reprojecting {gdf.crs} → {target_crs}...")
        gdf = gdf.to_crs(target_crs)


    return gdf


def get_tent_chunks(
    gdf: gpd.GeoDataFrame,
    asset_type: str,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Split a TENT edges GeoDataFrame into chunks by object_type.

    For roads: one chunk per tag_highway value (motorway, trunk, primary, secondary, tertiary)
    For rail:  one chunk per tag_railway value (rail, narrow_gauge)

    Args:
        gdf:        Loaded TENT GeoDataFrame (must have 'object_type' column)
        asset_type: 'roads' or 'rail'

    Returns:
        Dict mapping object_type_value → GeoDataFrame subset
    """
    chunks = {}
    for obj_type in sorted(gdf["object_type"].unique()):
        subset = gdf[gdf["object_type"] == obj_type]
        chunks[obj_type] = subset
        print(f"[TENT loader] Chunk '{obj_type}': {len(subset)} edges")
    return chunks


def list_available_asset_types(data_dir: Union[str, Path]) -> list[str]:
    """List which TENT asset types have parquet files in data_dir."""
    data_dir = Path(data_dir)
    available = []
    for asset_type, filename in TENT_FILES.items():
        if (data_dir / filename).exists():
            available.append(asset_type)
    return sorted(available)
