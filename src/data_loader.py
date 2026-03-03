"""
data_loader.py

Modular loader for the harmonised exposure database.
Handles file discovery, asset type name mapping, and ISO2/ISO3 conversion.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

# Folder/file name → internal asset type name used in vulnerability dicts
ASSET_NAME_MAP = {
    "Roadway":   "roads",
    "Railway":   "rail",
    "Airports":  "air",
    "Power":     "power",
    "Telecom":   "telecom",
    "Education": "education",
    "Healthcare":"healthcare",
    "Gas":       "gas",
    "Oil":       "oil",
    "Ports":     "ports",
}

# Reverse map: internal name → folder name
INTERNAL_TO_FOLDER = {v: k for k, v in ASSET_NAME_MAP.items()}

# ISO3 → ISO2
ISO3_TO_ISO2 = {
    "ALB": "AL", "AUT": "AT", "BEL": "BE", "BGR": "BG", "BIH": "BA",
    "CHE": "CH", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK",
    "ESP": "ES", "EST": "EE", "FIN": "FI", "FRA": "FR", "GBR": "GB",
    "GRC": "EL", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ISL": "IS",
    "ITA": "IT", "LIE": "LI", "LTU": "LT", "LUX": "LU", "LVA": "LV",
    "MKD": "MK", "MLT": "MT", "MNE": "ME", "NLD": "NL", "NOR": "NO",
    "POL": "PL", "PRT": "PT", "ROU": "RO", "SRB": "RS", "SVK": "SK",
    "SVN": "SI", "SWE": "SE", "XKO": "XK",
}

# ISO2 → ISO3 (derived automatically)
ISO2_TO_ISO3 = {v: k for k, v in ISO3_TO_ISO2.items()}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def to_iso2(country: str) -> str:
    """Convert ISO3 to ISO2. Returns input unchanged if already ISO2."""
    return ISO3_TO_ISO2.get(country.upper(), country.upper())

def to_iso3(country: str) -> str:
    """Convert ISO2 to ISO3. Returns input unchanged if already ISO3."""
    return ISO2_TO_ISO3.get(country.upper(), country.upper())

def to_internal_asset(asset: str) -> str:
    """Convert folder-style asset name to internal name (e.g. 'Roadway' → 'roads')."""
    return ASSET_NAME_MAP.get(asset, asset.lower())

def to_folder_asset(asset: str) -> str:
    """Convert internal asset name to folder name (e.g. 'roads' → 'Roadway')."""
    return INTERNAL_TO_FOLDER.get(asset.lower(), asset.capitalize())


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------

def list_available_countries(exposure_dir: Union[str, Path], asset_type: Optional[str] = None) -> list[str]:
    """
    List all ISO2 country codes available in the exposure database.

    Args:
        exposure_dir: Path to the Exposure_files directory
        asset_type: Optional internal asset type to filter by (e.g. 'power').
                    If None, returns union of all countries across all asset types.

    Returns:
        Sorted list of ISO2 country codes
    """
    exposure_dir = Path(exposure_dir)
    countries = set()

    if asset_type:
        folder_name = to_folder_asset(asset_type)
        folder = exposure_dir / folder_name
        if folder.exists():
            for f in folder.glob("*.parquet"):
                # Filename format: {AssetType}_{ISO2}.parquet
                parts = f.stem.split("_")
                if len(parts) == 2:
                    countries.add(parts[1])
    else:
        for folder in exposure_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("_"):
                for f in folder.glob("*.parquet"):
                    parts = f.stem.split("_")
                    if len(parts) == 2:
                        countries.add(parts[1])

    return sorted(countries)


def list_available_asset_types(exposure_dir: Union[str, Path], country: Optional[str] = None) -> list[str]:
    """
    List all internal asset type names available in the exposure database.

    Args:
        exposure_dir: Path to the Exposure_files directory
        country: Optional ISO2 or ISO3 country code to filter by.
                 If None, returns all asset types present in any country.

    Returns:
        Sorted list of internal asset type names (e.g. ['air', 'education', ...])
    """
    exposure_dir = Path(exposure_dir)
    iso2 = to_iso2(country) if country else None
    asset_types = []

    for folder in sorted(exposure_dir.iterdir()):
        if folder.is_dir() and not folder.name.startswith("_"):
            internal_name = to_internal_asset(folder.name)
            if iso2:
                # Only include if file for this country exists
                if (folder / f"{folder.name}_{iso2}.parquet").exists():
                    asset_types.append(internal_name)
            else:
                asset_types.append(internal_name)

    return asset_types


def get_file_path(exposure_dir: Union[str, Path], asset_type: str, country: str) -> Optional[Path]:
    """
    Get the path to a specific exposure file.

    Args:
        exposure_dir: Path to the Exposure_files directory
        asset_type: Internal asset type name (e.g. 'power', 'roads')
        country: ISO2 or ISO3 country code

    Returns:
        Path to the parquet file, or None if not found
    """
    exposure_dir = Path(exposure_dir)
    iso2 = to_iso2(country)
    folder_name = to_folder_asset(asset_type)
    path = exposure_dir / folder_name / f"{folder_name}_{iso2}.parquet"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_exposure(
    exposure_dir: Union[str, Path],
    asset_type: str,
    country: str,
    target_crs: Optional[int] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Load exposure data for a single asset type and country.

    Args:
        exposure_dir: Path to the Exposure_files directory
        asset_type: Internal asset type name (e.g. 'power', 'roads', 'rail')
        country: ISO2 or ISO3 country code
        target_crs: Optional EPSG code to reproject to. If None, keeps original CRS (EPSG:3035).

    Returns:
        GeoDataFrame with columns: osm_id, geometry, object_type, CNTR_CODE, NUTS2, LAU,
        plus asset-specific columns. Returns None if file not found.
    """
    path = get_file_path(exposure_dir, asset_type, country)

    if path is None:
        iso2 = to_iso2(country)
        print(f"No exposure file found for {asset_type} / {iso2}")
        return None

    gdf = gpd.read_parquet(path)

    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(epsg=target_crs)

    return gdf


def load_exposure_all_countries(
    exposure_dir: Union[str, Path],
    asset_type: str,
    countries: Optional[list[str]] = None,
    target_crs: Optional[int] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Load and concatenate exposure data for one asset type across multiple (or all) countries.

    Args:
        exposure_dir: Path to the Exposure_files directory
        asset_type: Internal asset type name (e.g. 'power')
        countries: List of ISO2 or ISO3 country codes. If None, loads all available.
        target_crs: Optional EPSG code to reproject to.

    Returns:
        Combined GeoDataFrame, or None if nothing was found.
    """
    if countries is None:
        countries = list_available_countries(exposure_dir, asset_type)

    gdfs = []
    for country in countries:
        gdf = load_exposure(exposure_dir, asset_type, country, target_crs=target_crs)
        if gdf is not None:
            gdfs.append(gdf)

    if not gdfs:
        return None

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def load_exposure_all_assets(
    exposure_dir: Union[str, Path],
    country: str,
    asset_types: Optional[list[str]] = None,
    target_crs: Optional[int] = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Load all asset types for a single country.

    Args:
        exposure_dir: Path to the Exposure_files directory
        country: ISO2 or ISO3 country code
        asset_types: List of internal asset type names to load. If None, loads all available.
        target_crs: Optional EPSG code to reproject to.

    Returns:
        Dict mapping internal asset type name → GeoDataFrame
    """
    if asset_types is None:
        asset_types = list_available_asset_types(exposure_dir, country)

    result = {}
    for asset_type in asset_types:
        gdf = load_exposure(exposure_dir, asset_type, country, target_crs=target_crs)
        if gdf is not None:
            result[asset_type] = gdf

    return result
