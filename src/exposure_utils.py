"""
exposure_utils.py

Shared utilities for the MIRACA exposure pipeline (heat, wildfire, landslide).

Provides:
  - Config loading from YAML
  - Infrastructure data loading (from harmonised exposure database)
  - ISO code conversions
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import yaml

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path] = "config.yml") -> dict:
    """Load YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy config.template.yml to config.yml and fill in your paths."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ISO mappings
# ---------------------------------------------------------------------------

ISO3_TO_ISO2 = {
    "ALB": "AL", "AUT": "AT", "BEL": "BE", "BGR": "BG", "BIH": "BA",
    "CHE": "CH", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK",
    "EST": "EE", "ESP": "ES", "FIN": "FI", "FRA": "FR", "GRC": "EL",
    "HRV": "HR", "HUN": "HU", "IRL": "IE", "ISL": "IS", "ITA": "IT",
    "LIE": "LI", "LTU": "LT", "LUX": "LU", "LVA": "LV", "MKD": "MK",
    "MLT": "MT", "MNE": "ME", "NLD": "NL", "NOR": "NO", "POL": "PL",
    "PRT": "PT", "ROU": "RO", "SRB": "RS", "SVK": "SK", "SVN": "SI",
    "SWE": "SE", "XKO": "XK",
}

ISO2_TO_ISO3 = {v: k for k, v in ISO3_TO_ISO2.items()}


def to_iso2(code: str) -> str:
    if len(code) == 2:
        return code.upper()
    return ISO3_TO_ISO2.get(code.upper(), code[:2].upper())


def to_iso3(code: str) -> str:
    if len(code) == 3:
        return code.upper()
    return ISO2_TO_ISO3.get(code.upper(), code.upper())


# ---------------------------------------------------------------------------
# Infrastructure loading
# ---------------------------------------------------------------------------

_ASSET_FOLDER_MAP = {
    "roads":      "Roadway",
    "main_roads": "Roadway",
    "rail":       "Railway",
    "air":        "Airports",
    "telecom":    "Telecom",
    "education":  "Education",
    "healthcare": "Healthcare",
    "power":      "Power",
    "gas":        "Gas",
    "oil":        "Oil",
    "ports":      "Ports",
}


def load_infrastructure(
    exposure_dir: Union[str, Path],
    asset_type: str,
    country_iso2: str,
) -> Optional[gpd.GeoDataFrame]:
    """
    Load infrastructure features from the harmonised exposure database.

    Args:
        exposure_dir:   Root Exposure_files/ directory
        asset_type:     Internal asset type name (e.g. 'rail', 'power')
        country_iso2:   ISO2 country code (e.g. 'PT')

    Returns:
        GeoDataFrame or None if file not found
    """
    folder = _ASSET_FOLDER_MAP.get(asset_type)
    if folder is None:
        raise ValueError(f"Unknown asset type: {asset_type}")

    path = Path(exposure_dir) / folder / f"{folder}_{country_iso2}.parquet"
    if not path.exists():
        return None

    features = gpd.read_parquet(path)
    valid_mask = features.geometry.is_valid
    if not valid_mask.all():
        features = features[valid_mask].copy()

    return features
