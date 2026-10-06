"""
constants.py

Single source of truth for all vulnerability curves, fragility curve
references, and maximum damage values used across the MIRACA pipeline.

Updated to match the harmonised exposure database object_type values:
  - Road grouping names ("Motorways and Trunks", "Primary Roads", etc.)
  - narrow_gauge (rail)
  - communication (telecom)
  - apron (air)
  - harbour, port (ports)
  - gasometer, gas, LNG, natural_gas (gas)
  - refinery, oil, crude_oil, diesel, petroleum, fuel_oil, fuel, storage_tank (oil)
"""

# ---------------------------------------------------------------------------
# ISO3 ↔ ISO2 mapping (European countries)
# ---------------------------------------------------------------------------

ISO3_TO_ISO2 = {
    "ALB": "AL",
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "BIH": "BA",
    "CHE": "CH",
    "CYP": "CY",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "EST": "EE",
    "ESP": "ES",
    "FIN": "FI",
    "FRA": "FR",
    "GRC": "EL",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ISL": "IS",
    "ITA": "IT",
    "LIE": "LI",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
    "MKD": "MK",
    "MLT": "MT",
    "MNE": "ME",
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SRB": "RS",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
    "XKO": "XK",
}

ISO2_TO_ISO3 = {v: k for k, v in ISO3_TO_ISO2.items()}

# ---------------------------------------------------------------------------
# Dashboard system name mapping
# Maps internal asset type names → dashboard system names used in output
# file and column naming.
# ---------------------------------------------------------------------------

SYSTEM_NAME_MAP = {
    "roads": "roads",
    "main_roads": "roads",
    "rail": "rail",
    "air": "airports",
    "telecom": "telecom",
    "education": "education",
    "healthcare": "healthcare",
    "power": "power",
    "gas": "gas",
    "oil": "oil",
    "ports": "ports",
}


def to_system_name(asset_type: str) -> str:
    """Convert internal asset type name to dashboard system name."""
    return SYSTEM_NAME_MAP.get(asset_type, asset_type)


# ---------------------------------------------------------------------------
# River future scenario mapping: temperature codes → time periods + SSP
# ---------------------------------------------------------------------------

# The basin climate data uses temperature labels (1.5C, 2.0C, 3.0C, 4.0C).
# The dashboard uses time periods + SSP scenarios, aligned with coastal:
#   1.5°C → RCP4.5 / 2050 → SSP245
#   2.0°C → RCP8.5 / 2050 → SSP585
#   3.0°C → RCP4.5 / 2100 → SSP245
#   4.0°C → RCP8.5 / 2100 → SSP585

RIVER_FUTURE_TEMP_CODES = ("15", "20", "30", "40")
RIVER_FUTURE_TEMP_LABELS = ("2050_SSP245", "2050_SSP585", "2100_SSP245", "2100_SSP585")

# ---------------------------------------------------------------------------
# RCP → SSP scenario mapping (for heat / wildfire exposure)
# ---------------------------------------------------------------------------

RCP_TO_SSP = {
    "rcp_4_5": "SSP245",
    "rcp_8_5": "SSP585",
}

# ---------------------------------------------------------------------------
# Exposure time window → dashboard period mapping (for heat / wildfire)
# ---------------------------------------------------------------------------

WINDOW_TO_PERIOD = {
    "recent": "current",
    "near_future": None,       # dropped — no dashboard equivalent
    "mid_future": "2050",      # 2041-2060 → "2050"
    "far_future": "2100",      # 2061-2080 → "2100"
    "distant_future": None,    # dropped — no dashboard equivalent
}

# ---------------------------------------------------------------------------
# Shared pipeline settings (merge_outputs, aggregate_outputs, figures, ...)
# ---------------------------------------------------------------------------

# Hazards with a damage function (EAD) and hazards with an exposure metric
EAD_HAZARDS = ["river", "coastal", "windstorm", "earthquake"]
EXPOSURE_HAZARDS = ["river", "coastal", "windstorm", "earthquake",
                    "heat", "wildfire", "landslide"]

# Output period labels. For heat/wildfire "2050" = 2041-2060 and "2100" =
# 2061-2080 (WINDOW_TO_PERIOD); for river they are warming levels
# (RIVER_FUTURE_TEMP_LABELS); for coastal the CoCliCo 2050/2100 horizons.
FUTURE_PERIODS = ["2050_SSP245", "2050_SSP585", "2100_SSP245", "2100_SSP585"]

# The eight CI systems reported in the paper; oil and gas are computed by the
# pipeline but excluded from all paper figures and tables.
PAPER_SYSTEMS = ["power", "roads", "rail", "airports", "ports",
                 "education", "healthcare", "telecom"]
EXCLUDED_SYSTEMS = {"oil", "gas"}

# Roads object_type → road_class mapping (5 analysis classes)
ROAD_CLASS_MAP = {
    "motorway": "motorways_and_trunks",
    "motorway_link": "motorways_and_trunks",
    "trunk": "motorways_and_trunks",
    "trunk_link": "motorways_and_trunks",
    "Motorways and Trunks": "motorways_and_trunks",
    "primary": "primary_roads",
    "primary_link": "primary_roads",
    "Primary Roads": "primary_roads",
    "secondary": "secondary_roads",
    "secondary_link": "secondary_roads",
    "Secondary roads": "secondary_roads",
    "tertiary": "tertiary_roads",
    "tertiary_link": "tertiary_roads",
    "Tertiary roads": "tertiary_roads",
    "residential": "other_roads",
    "road": "other_roads",
    "unclassified": "other_roads",
    "track": "other_roads",
    "service": "other_roads",
    "Other roads": "other_roads",
}


def load_config() -> dict:
    """Read config.yml from the repository root (one level above src/)."""
    from pathlib import Path

    import yaml

    config_path = Path(__file__).parent.parent / "config.yml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"config.yml not found at {config_path}.\n"
            "Copy config.template.yml to config.yml and fill in your paths."
        )

# ---------------------------------------------------------------------------
# Maximum damage values per asset type
# Format: {object_type: [min, mean, max]}  (€/m for lines, €/m² for polygons,
#                                           €/unit for points)
#
# NOTE: storage_tank, gasometer, and petroleum_well values were converted
# from per-unit to per-m² by dividing by the median polygon area observed
# across the European exposure database (storage_tank: 193 m², gasometer:
# 54 m², petroleum_well: 8682 m²).  Point petroleum wells must be assigned
# a default pad area (DEFAULT_WELL_PAD_AREA_M2) before damage calculation
# so the per-m² rate produces a realistic total.
# ---------------------------------------------------------------------------

# Default well pad area (m²) assigned to point petroleum_well features.
# Based on European conventional well site sizes (literature range 1000–20000 m²,
# conservative estimate for a typical single-well pad).
DEFAULT_WELL_PAD_AREA_M2 = 3000


def assign_default_well_pad_area(features, asset_type: str):
    """
    For oil assets: convert point petroleum_well features to polygons
    by buffering with DEFAULT_WELL_PAD_AREA_M2, so the per-m² maxdam
    values produce realistic damage totals.

    Modifies features in place and returns them.
    Should be called after loading features, before damage assessment.
    """
    if asset_type != "oil":
        return features
    if "object_type" not in features.columns:
        return features

    is_point_well = (
        (features["object_type"] == "petroleum_well")
        & features.geometry.geom_type.isin(["Point", "MultiPoint"])
    )

    if not is_point_well.any():
        return features

    import numpy as np

    # Buffer points into circles with area = DEFAULT_WELL_PAD_AREA_M2
    # area = π r² → r = √(area / π)
    radius = np.sqrt(DEFAULT_WELL_PAD_AREA_M2 / np.pi)

    # Buffer in the current CRS (should be metric, e.g. EPSG:3035)
    features = features.copy()
    features.loc[is_point_well, "geometry"] = (
        features.loc[is_point_well].geometry.buffer(radius)
    )

    n = is_point_well.sum()
    print(f"[constants] Buffered {n} point petroleum_well features "
          f"to {DEFAULT_WELL_PAD_AREA_M2} m² polygons (r={radius:.1f}m)")

    return features

INFRASTRUCTURE_DAMAGE_VALUES = {
    # ----- Roads -----
    # Individual OSM types + grouped names used by the harmonised database.
    # Grouped names map to the same values as the constituent types.
    "roads": {
        # Individual types
        "motorway": [1106, 2895, 3931],
        "motorway_link": [1106, 2895, 3931],
        "trunk": [848, 1242, 1636],
        "trunk_link": [848, 1242, 1636],
        "primary": [917, 1137, 1357],
        "primary_link": [917, 1137, 1357],
        "secondary": [257, 452, 678],
        "secondary_link": [257, 452, 678],
        "tertiary": [203, 271, 339],
        "tertiary_link": [203, 271, 339],
        "residential": [66, 136, 305],
        "road": [66, 136, 305],
        "unclassified": [66, 136, 305],
        "track": [66, 136, 305],
        "service": [66, 136, 305],
        # Grouped names (harmonised exposure database)
        "Motorways and Trunks": [1106, 2895, 3931],
        "Primary Roads": [917, 1137, 1357],
        "Secondary roads": [257, 452, 678],
        "Tertiary roads": [203, 271, 339],
        "Other roads": [66, 136, 305],
    },
    "main_roads": {
        "motorway": [1106, 2895, 3931],
        "motorway_link": [1106, 2895, 3931],
        "trunk": [848, 1242, 1636],
        "trunk_link": [848, 1242, 1636],
        "primary": [917, 1137, 1357],
        "primary_link": [917, 1137, 1357],
        "secondary": [257, 452, 678],
        "secondary_link": [257, 452, 678],
        "tertiary": [203, 271, 339],
        "tertiary_link": [203, 271, 339],
        "Motorways and Trunks": [1106, 2895, 3931],
        "Primary Roads": [917, 1137, 1357],
        "Secondary roads": [257, 452, 678],
        "Tertiary roads": [203, 271, 339],
    },
    # ----- Rail -----
    "rail": {
        "rail": [491, 2858, 14186],
        "narrow_gauge": [491, 2858, 14186],
    },
    # ----- Air -----
    "air": {
        "aerodrome": [113, 135, 165],
        "apron": [113, 135, 165],
        "terminal": [113, 165, 4271],
        "runway": [4133, 5511, 9078],
    },
    # ----- Telecom -----
    "telecom": {
        "mast": [67506, 76630, 111998],
        "communications_tower": [139610, 152468, 229376],
        "tower": [139610, 152468, 229376],
        "communication": [67506, 76630, 111998],
    },
    # ----- Education -----
    "education": {
        "school": [267, 713, 1294],
        "kindergarten": [267, 713, 1294],
        "college": [267, 713, 1294],
        "university": [267, 713, 1294],
        "library": [267, 713, 1294],
    },
    # ----- Healthcare -----
    "healthcare": {
        "hospital": [591, 1294, 2227],
        "clinic": [591, 1294, 2227],
        "doctors": [591, 1294, 2227],
        "pharmacy": [591, 1294, 2227],
        "dentist": [591, 1294, 2227],
        "physiotherapist": [591, 1294, 2227],
        "alternative": [591, 1294, 2227],
        "laboratory": [591, 1294, 2227],
        "optometrist": [591, 1294, 2227],
        "rehabilitation": [591, 1294, 2227],
        "blood_donation": [591, 1294, 2227],
        "birthing_center": [591, 1294, 2227],
    },
    # ----- Power -----
    "power": {
        "line": [108, 183, 1151],
        "cable": [215, 1818, 5497],
        "minor_line": [71, 102, 103],
        "plant": [649, 1558, 11110],
        "generator": [1299, 1904, 6349],
        "substation": [1299, 1904, 6349],
        "transformer": [1299, 1904, 6349],
        "pole": [73005, 97627, 369547],
        "portal": [1299, 1904, 6349],
        "tower": [6171, 103928, 275472],
        "terminal": [1299, 1904, 6349],
        "switch": [1299, 1904, 6349],
        "catenary_mast": [67506, 76630, 111998],
    },
    # ----- Gas -----
    "gas": {
        "pipeline": [71, 102, 103],
        "storage_tank": [157, 4181, 7840],   #[30310, 808265, 1515497],
        "gasometer": [558, 14885, 27910], #[30310, 808265, 1515497],
        "substation": [1299, 1904, 6349], #[1299886, 19047345, 63491148],
        "gas": [1299, 1904, 6349], #[1299886, 19047345, 63491148],
        "LNG": [1299, 1904, 6349], #[1299886, 19047345, 63491148],
        "natural_gas": [1299, 1904, 6349], #[1299886, 19047345, 63491148],
    },
    # ----- Oil -----
    "oil": {
        "pipeline": [71, 102, 103],
        "petroleum_well": [35, 47, 58],   #[303100, 404133, 505166],
        "oil_refinery": [1299, 1904, 6349], #[6499430, 155817332, 1111095098],
        "storage_tank":  [157, 4181, 7840], #[30310, 808265, 1515497],
        "substation": [1299, 1904, 6349], #[1299886, 19047345, 63491148],
        "refinery": [1299, 1904, 6349], #[6499430, 155817332, 1111095098],
        "oil": [1299, 1904, 6349], #[6499430, 155817332, 1111095098],
        "crude_oil": [1299, 1904, 6349], #[6499430, 155817332, 1111095098],
        "diesel":[1299, 1904, 6349], # [6499430, 155817332, 1111095098],
        "petroleum": [1299, 1904, 6349], #[6499430, 155817332, 1111095098],
        "fuel_oil":[1299, 1904, 6349], # [6499430, 155817332, 1111095098],
        "fuel": [1299, 1904, 6349], #[6499430, 155817332, 1111095098],
    },
    # ----- Ports -----
    "ports": {
        "port": [113, 135, 165],
        "terminal": [113, 165, 4271],
        "harbour": [113, 135, 165],
    },
}

# ---------------------------------------------------------------------------
# Flood vulnerability curve IDs per asset type
# Format: {object_type: [curve_id, ...]}
# ---------------------------------------------------------------------------

DICT_CIS_VULNERABILITY_FLOOD = {
    "roads": {
        "motorway": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "motorway_link": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "trunk": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "trunk_link": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "primary": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "primary_link": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "secondary": ["F7.9", "F7.8"],
        "secondary_link": ["F7.9", "F7.8"],
        "tertiary": ["F7.9", "F7.8"],
        "tertiary_link": ["F7.9", "F7.8"],
        "residential": ["F7.9", "F7.8"],
        "road": ["F7.9", "F7.8"],
        "unclassified": ["F7.9", "F7.8"],
        "track": ["F7.9", "F7.8"],
        "service": ["F7.9", "F7.8"],
        "Motorways and Trunks": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "Primary Roads": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "Secondary roads": ["F7.9", "F7.8"],
        "Tertiary roads": ["F7.9", "F7.8"],
        "Other roads": ["F7.9", "F7.8"],
    },
    "main_roads": {
        "motorway": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "motorway_link": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "trunk": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "trunk_link": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "primary": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "primary_link": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "secondary": ["F7.9", "F7.8"],
        "secondary_link": ["F7.9", "F7.8"],
        "tertiary": ["F7.9", "F7.8"],
        "tertiary_link": ["F7.9", "F7.8"],
        "Motorways and Trunks": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "Primary Roads": ["F7.5", "F7.6", "F7.7", "F7.4"],
        "Secondary roads": ["F7.9", "F7.8"],
        "Tertiary roads": ["F7.9", "F7.8"],
    },
    "rail": {
        "rail": ["F8.1", "F8.2", "F8.3", "F8.4", "F8.5", "F8.6", "F8.7"],
        "narrow_gauge": ["F8.1", "F8.2", "F8.3", "F8.4", "F8.5", "F8.6", "F8.7"],
    },
    "air": {
        "aerodrome": ["F9.1", "F9.2", "F9.3"],
        "apron": ["F9.1", "F9.2", "F9.3"],
        "terminal": ["F9.1"],
        "runway": ["F7.4", "F7.5", "F7.6", "F7.7"],
    },
    "telecom": {
        "mast": ["F10.1"],
        "communications_tower": ["F6.1", "F6.2"],
        "tower": ["F6.1", "F6.2"],
        "communication": ["F10.1"],
    },
    "education": {
        "school": ["F21.6", "F21.7", "F21.8", "F21.10", "F21.11", "F21.13"],
        "kindergarten": ["F21.6", "F21.7", "F21.8", "F21.10", "F21.11", "F21.13"],
        "college": ["F21.6", "F21.7", "F21.8", "F21.10", "F21.11", "F21.13"],
        "university": ["F21.6", "F21.7", "F21.8", "F21.10", "F21.11", "F21.13"],
        "library": ["F21.6", "F21.7", "F21.8", "F21.10", "F21.11", "F21.13"],
    },
    "healthcare": {
        "hospital": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "clinic": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "doctors": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "pharmacy": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "dentist": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "physiotherapist": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "alternative": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "laboratory": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "optometrist": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "rehabilitation": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "blood_donation": ["F21.6", "F21.8", "F21.9", "F21.12"],
        "birthing_center": ["F21.6", "F21.8", "F21.9", "F21.12"],
    },
    "power": {
        "line": ["F6.1", "F6.2"],
        "cable": ["F5.1"],
        "minor_line": ["F6.1", "F6.2"],
        "plant": ["F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7"],
        "generator": ["F2.1", "F2.2", "F2.3"],
        "substation": ["F2.1", "F2.2", "F2.3"],
        "transformer": ["F2.1", "F2.2", "F2.3"],
        "pole": ["F6.1", "F6.2"],
        "portal": ["F2.1", "F2.2", "F2.3"],
        "tower": ["F6.1", "F6.2"],
        "terminal": ["F9.1"],
        "switch": ["F2.1", "F2.2", "F2.3"],
        "catenary_mast": ["F10.1"],
    },
    "gas": {
        "pipeline": ["F16.1", "F16.2", "F16.3"],
        "storage_tank": ["F2.1", "F2.2", "F2.3"],
        "gasometer": ["F13.1", "F13.2", "F13.3", "F13.5"],
        "substation": ["F2.1", "F2.2", "F2.3"],
        "gas": ["F2.1", "F2.2", "F2.3"],
        "LNG": ["F2.1", "F2.2", "F2.3"],
        "natural_gas": ["F2.1", "F2.2", "F2.3"],
    },
    "oil": {
        "pipeline": ["F16.1", "F16.2", "F16.3"],
        "petroleum_well": ["F15.1"],
        "oil_refinery": ["F1.4"],
        "storage_tank": ["F2.1", "F2.2", "F2.3"],
        "substation": ["F2.1", "F2.2", "F2.3"],
        "refinery": ["F2.1", "F2.2", "F2.3"],
        "oil": ["F2.1", "F2.2", "F2.3"],
        "crude_oil": ["F2.1", "F2.2", "F2.3"],
        "diesel": ["F2.1", "F2.2", "F2.3"],
        "petroleum": ["F2.1", "F2.2", "F2.3"],
        "fuel_oil": ["F2.1", "F2.2", "F2.3"],
        "fuel": ["F2.1", "F2.2", "F2.3"],
    },
    "ports": {
        "port": ["F9.1"],
        "terminal": ["F9.1"],
        "harbour": ["F9.1"],
    },
}

# ---------------------------------------------------------------------------
# Wind design standard (IEC 60826 lower bound — RP50 for all asset types)
# ---------------------------------------------------------------------------

WIND_PROTECTION_STANDARD_RP = 50  # years

# ---------------------------------------------------------------------------
# Wind vulnerability curve IDs per asset type
# ---------------------------------------------------------------------------

DICT_CIS_VULNERABILITY_WIND = {
    "roads": {
        "motorway": ["W7.2"],
        "motorway_link": ["W7.2"],
        "trunk": ["W7.2"],
        "trunk_link": ["W7.2"],
        "primary": ["W7.2"],
        "primary_link": ["W7.2"],
        "secondary": ["W7.2"],
        "secondary_link": ["W7.2"],
        "tertiary": ["W7.2"],
        "tertiary_link": ["W7.2"],
        "residential": ["W7.2"],
        "road": ["W7.2"],
        "unclassified": ["W7.2"],
        "track": ["W7.2"],
        "service": ["W7.2"],
        "Motorways and Trunks": ["W7.2"],
        "Primary Roads": ["W7.2"],
        "Secondary roads": ["W7.2"],
        "Tertiary roads": ["W7.2"],
        "Other roads": ["W7.2"],
    },
    "main_roads": {
        "motorway": ["W7.2"],
        "motorway_link": ["W7.2"],
        "trunk": ["W7.2"],
        "trunk_link": ["W7.2"],
        "primary": ["W7.2"],
        "primary_link": ["W7.2"],
        "secondary": ["W7.2"],
        "secondary_link": ["W7.2"],
        "tertiary": ["W7.2"],
        "tertiary_link": ["W7.2"],
        "Motorways and Trunks": ["W7.2"],
        "Primary Roads": ["W7.2"],
        "Secondary roads": ["W7.2"],
        "Tertiary roads": ["W7.2"],
    },
    "rail": {
        # W3.9/W3.6/W3.12 = power tower damage-factor curves at 150/120/180 km/h design speed.
        # Proxy for catenary mast damage per km of track (no dedicated rail catenary curve exists).
        "rail": ["W3.9", "W3.6", "W3.12"],
        "narrow_gauge": ["W3.9", "W3.6", "W3.12"],
    },
    "air": {
        "aerodrome": ["W7.2"],
        "apron": ["W7.2"],
        "terminal": ["W21.13", "W21.14"],
        "runway": ["W7.2"],
    },
    "telecom": {
        "mast": [
            "W3.5",
            "W3.6",
            "W3.7",
            "W3.8",
            "W3.9",
            "W3.10",
            "W3.11",
            "W3.12",
            "W3.13",
            "W3.14",
        ],
        "tower": [
            "W3.5",
            "W3.6",
            "W3.7",
            "W3.8",
            "W3.9",
            "W3.10",
            "W3.11",
            "W3.12",
            "W3.13",
            "W3.14",
        ],
        "communications_tower": [
            "W10.3",
            "W10.4",
            "W10.5",
            "W10.6",
            "W10.7",
            "W10.8",
            "W10.9",
        ],
        "communication": [
            "W3.5",
            "W3.6",
            "W3.7",
            "W3.8",
            "W3.9",
            "W3.10",
            "W3.11",
            "W3.12",
            "W3.13",
            "W3.14",
        ],
    },
    "education": {
        "school": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "kindergarten": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "college": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "university": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "library": ["W21.11", "W21.12", "W21.13", "W21.14"],
    },
    "healthcare": {
        "hospital": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "clinic": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "doctors": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "pharmacy": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "dentist": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "physiotherapist": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "alternative": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "laboratory": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "optometrist": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "rehabilitation": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "blood_donation": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "birthing_center": ["W21.11", "W21.12", "W21.13", "W21.14"],
    },
    "power": {
        "line": ["W6.1", "W6.2", "W6.3"],
        "cable": ["W7.2"],
        "minor_line": ["W6.1", "W6.2", "W6.3"],
        "pole": ["W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],
        "tower": [
            "W3.5",
            "W3.6",
            "W3.7",
            "W3.8",
            "W3.9",
            "W3.10",
            "W3.11",
            "W3.12",
            "W3.13",
            "W3.14",
        ],
        "catenary_mast": ["W4.33", "W4.34", "W4.35", "W4.36", "W4.37"],
    },
    "gas": {
        "pipeline": ["W7.2"],
        "storage_tank": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "gasometer": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "substation": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "gas": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "LNG": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "natural_gas": ["W21.11", "W21.12", "W21.13", "W21.14"],
    },
    "oil": {
        "pipeline": ["W7.2"],
        "petroleum_well": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "oil_refinery": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "storage_tank": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "substation": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "refinery": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "oil": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "crude_oil": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "diesel": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "petroleum": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "fuel_oil": ["W21.11", "W21.12", "W21.13", "W21.14"],
        "fuel": ["W21.11", "W21.12", "W21.13", "W21.14"],
    },
    "ports": {
        "port": ["W7.2"],
        "terminal": ["W21.13", "W21.14"],
        "harbour": ["W7.2"],
    },
}

# ---------------------------------------------------------------------------
# Earthquake fragility curve IDs per asset type
# ---------------------------------------------------------------------------

_EQ_ROAD_CURVES = [
    "E7.2",
    "E7.3",
    "E7.4",
    "E7.5",
    "E7.6",
    "E7.7",
    "E7.8",
    "E7.9",
    "E7.10",
]
_EQ_RAIL_CURVES = [
    "E8.1",
    "E8.2",
    "E8.3",
    "E8.4",
    "E8.5",
    "E8.6",
    "E8.7",
    "E8.8",
    "E8.9",
    "E8.10",
    "E8.11",
    "E8.12",
    "E8.13",
    "E8.14",
    "E8.15",
    "E8.16",
    "E8.17",
    "E8.18",
    "E8.19",
    "E8.20",
]
_EQ_EDU_CURVES = [
    "E21.26-C",
    "E21.27-C",
    "E21.29-C",
    "E21.30-C",
    "E21.31-C",
    "E21.32-C",
    "E21.33-C",
    "E21.34-C",
    "E21.35-C",
    "E21.36-C",
    "E21.37-C",
    "E21.38-C",
    "E21.39-C",
    "E21.40-C",
    "E21.41-C",
    "E21.42-C",
    "E21.43-C",
    "E21.48-C",
    "E21.49-C",
    "E21.50-C",
    "E21.51-C",
    "E21.52-C",
    "E21.53-C",
    "E21.54-C",
    "E21.55-C",
    "E21.56-C",
    "E21.57-C",
    "E21.58-C",
    "E21.59-C",
    "E21.60-C",
    "E21.61-C",
]
_EQ_HEALTH_CURVES = [
    "E21.67-C",
    "E21.68-C",
    "E21.69-C",
    "E21.70-C",
    "E21.71-C",
    "E21.72-C",
]
_EQ_SUB_CURVES = [
    "E2.1",
    "E2.2",
    "E2.3",
    "E2.4",
    "E2.5",
    "E2.6",
    "E2.7",
    "E2.8",
    "E2.9",
]
_EQ_GEN_CURVES = ["E1.1", "E1.2", "E1.3", "E1.4", "E1.5", "E1.6", "E1.7", "E1.8"]
_EQ_LINE_CURVES = ["E6.1", "E6.2", "E6.3", "E6.4"]

DICT_CIS_VULNERABILITY_EARTHQUAKE = {
    "roads": {
        "motorway": _EQ_ROAD_CURVES,
        "motorway_link": _EQ_ROAD_CURVES,
        "trunk": _EQ_ROAD_CURVES,
        "trunk_link": _EQ_ROAD_CURVES,
        "primary": _EQ_ROAD_CURVES,
        "primary_link": _EQ_ROAD_CURVES,
        "secondary": _EQ_ROAD_CURVES,
        "secondary_link": _EQ_ROAD_CURVES,
        "tertiary": _EQ_ROAD_CURVES,
        "tertiary_link": _EQ_ROAD_CURVES,
        "residential": _EQ_ROAD_CURVES,
        "road": _EQ_ROAD_CURVES,
        "unclassified": _EQ_ROAD_CURVES,
        "track": _EQ_ROAD_CURVES,
        "service": _EQ_ROAD_CURVES,
        "Motorways and Trunks": _EQ_ROAD_CURVES,
        "Primary Roads": _EQ_ROAD_CURVES,
        "Secondary roads": _EQ_ROAD_CURVES,
        "Tertiary roads": _EQ_ROAD_CURVES,
        "Other roads": _EQ_ROAD_CURVES,
    },
    "main_roads": {
        "motorway": _EQ_ROAD_CURVES,
        "motorway_link": _EQ_ROAD_CURVES,
        "trunk": _EQ_ROAD_CURVES,
        "trunk_link": _EQ_ROAD_CURVES,
        "primary": _EQ_ROAD_CURVES,
        "primary_link": _EQ_ROAD_CURVES,
        "secondary": _EQ_ROAD_CURVES,
        "secondary_link": _EQ_ROAD_CURVES,
        "tertiary": _EQ_ROAD_CURVES,
        "tertiary_link": _EQ_ROAD_CURVES,
        "Motorways and Trunks": _EQ_ROAD_CURVES,
        "Primary Roads": _EQ_ROAD_CURVES,
        "Secondary roads": _EQ_ROAD_CURVES,
        "Tertiary roads": _EQ_ROAD_CURVES,
    },
    "rail": {
        "rail": _EQ_RAIL_CURVES,
        "narrow_gauge": _EQ_RAIL_CURVES,
    },
    "air": {
        "aerodrome": ["E9.2", "E9.3", "E9.4"],
        "apron": ["E9.2", "E9.3", "E9.4"],
        "terminal": ["E9.2", "E9.3", "E9.4"],
        "runway": _EQ_ROAD_CURVES,
    },
    "telecom": {
        "mast": ["E11.1"],
        "communications_tower": ["E3.1", "E3.2"],
        "tower": ["E3.1", "E3.2"],
        "communication": ["E11.1"],
    },
    "education": {
        "school": _EQ_EDU_CURVES,
        "kindergarten": _EQ_EDU_CURVES,
        "college": _EQ_EDU_CURVES,
        "university": _EQ_EDU_CURVES,
        "library": _EQ_EDU_CURVES,
    },
    "healthcare": {
        "hospital": _EQ_HEALTH_CURVES,
        "clinic": _EQ_HEALTH_CURVES,
        "doctors": _EQ_HEALTH_CURVES,
        "pharmacy": _EQ_HEALTH_CURVES,
        "dentist": _EQ_HEALTH_CURVES,
        "physiotherapist": _EQ_HEALTH_CURVES,
        "alternative": _EQ_HEALTH_CURVES,
        "laboratory": _EQ_HEALTH_CURVES,
        "optometrist": _EQ_HEALTH_CURVES,
        "rehabilitation": _EQ_HEALTH_CURVES,
        "blood_donation": _EQ_HEALTH_CURVES,
        "birthing_center": _EQ_HEALTH_CURVES,
    },
    "power": {
        "line": _EQ_LINE_CURVES,
        "cable": _EQ_LINE_CURVES,
        "minor_line": _EQ_LINE_CURVES,
        "plant": _EQ_GEN_CURVES,
        "generator": _EQ_GEN_CURVES,
        "substation": _EQ_SUB_CURVES,
        "transformer": _EQ_SUB_CURVES,
        "pole": ["E4.1", "E4.2", "E4.3", "E4.4"],
        "portal": _EQ_SUB_CURVES,
        "tower": ["E3.1", "E3.2"],
        "terminal": _EQ_SUB_CURVES,
        "switch": _EQ_SUB_CURVES,
        "catenary_mast": ["E4.1", "E4.2", "E4.3", "E4.4"],
    },
    "gas": {
        "pipeline": _EQ_LINE_CURVES,
        "storage_tank": _EQ_GEN_CURVES,
        "gasometer": _EQ_GEN_CURVES,
        "substation": _EQ_SUB_CURVES,
        "gas": _EQ_SUB_CURVES,
        "LNG": _EQ_SUB_CURVES,
        "natural_gas": _EQ_SUB_CURVES,
    },
    "oil": {
        "pipeline": _EQ_LINE_CURVES,
        "petroleum_well": _EQ_GEN_CURVES,
        "oil_refinery": _EQ_GEN_CURVES,
        "storage_tank": _EQ_GEN_CURVES,
        "substation": _EQ_SUB_CURVES,
        "refinery": _EQ_GEN_CURVES,
        "oil": _EQ_GEN_CURVES,
        "crude_oil": _EQ_GEN_CURVES,
        "diesel": _EQ_GEN_CURVES,
        "petroleum": _EQ_GEN_CURVES,
        "fuel_oil": _EQ_GEN_CURVES,
        "fuel": _EQ_GEN_CURVES,
    },
    "ports": {
        "port": ["E9.2", "E9.3", "E9.4"],
        "terminal": ["E9.2", "E9.3", "E9.4"],
        "harbour": ["E9.2", "E9.3", "E9.4"],
    },
}

# TENT asset type aliases — map TENT names to existing vulnerability dict keys
INFRASTRUCTURE_DAMAGE_VALUES["airports"] = INFRASTRUCTURE_DAMAGE_VALUES["air"]
DICT_CIS_VULNERABILITY_FLOOD["airports"] = DICT_CIS_VULNERABILITY_FLOOD["air"]
DICT_CIS_VULNERABILITY_WIND["airports"] = DICT_CIS_VULNERABILITY_WIND["air"]
DICT_CIS_VULNERABILITY_EARTHQUAKE["airports"] = DICT_CIS_VULNERABILITY_EARTHQUAKE["air"]

# OSM energy network — lines (overhead/underground) and buses (substations)
_ENERGY_LINE_TYPES = ("line", "cable")
_ENERGY_BUS_TYPES = ("substation",)

INFRASTRUCTURE_DAMAGE_VALUES["energy_lines"] = {
    k: v for k, v in INFRASTRUCTURE_DAMAGE_VALUES["power"].items()
    if k in _ENERGY_LINE_TYPES
}
INFRASTRUCTURE_DAMAGE_VALUES["energy_buses"] = {
    k: v for k, v in INFRASTRUCTURE_DAMAGE_VALUES["power"].items()
    if k in _ENERGY_BUS_TYPES
}
DICT_CIS_VULNERABILITY_FLOOD["energy_lines"] = {
    k: v for k, v in DICT_CIS_VULNERABILITY_FLOOD["power"].items()
    if k in _ENERGY_LINE_TYPES
}
DICT_CIS_VULNERABILITY_FLOOD["energy_buses"] = {
    k: v for k, v in DICT_CIS_VULNERABILITY_FLOOD["power"].items()
    if k in _ENERGY_BUS_TYPES
}
DICT_CIS_VULNERABILITY_WIND["energy_lines"] = {
    k: v for k, v in DICT_CIS_VULNERABILITY_WIND["power"].items()
    if k in _ENERGY_LINE_TYPES
}
DICT_CIS_VULNERABILITY_WIND["energy_buses"] = {}  # substations skipped in wind wrapper
DICT_CIS_VULNERABILITY_EARTHQUAKE["energy_lines"] = {
    k: v for k, v in DICT_CIS_VULNERABILITY_EARTHQUAKE["power"].items()
    if k in _ENERGY_LINE_TYPES
}
DICT_CIS_VULNERABILITY_EARTHQUAKE["energy_buses"] = {
    k: v for k, v in DICT_CIS_VULNERABILITY_EARTHQUAKE["power"].items()
    if k in _ENERGY_BUS_TYPES
}

# ---------------------------------------------------------------------------
# Port TENT damage values — used with port-specific curves (curves_flooding.xlsx,
# curves_TC.xlsx, curves_earthquake.xlsx). Values in €/m².
# land_use column normalized to these keys (industry→Industry, refinery→Refinery).
# ---------------------------------------------------------------------------

PORT_TENT_MAXDAM = {
    # land_use:         cost (single value, no uncertainty range)
    "General Cargo":    [1063,   1063,   1063],
    "Container":        [ 835,    835,    835],
    "RoRo":             [ 760,    760,    760],
    "Liquid":           [ 988,    988,    988],
    "Dry Bulk":         [ 531,    531,    531],
    "Raw":              [ 531,    531,    531],
    "Refinery":         [1822,   1822,   1822],
    "Industry":         [ 400,    400,    400],
    "Warehouse":        [ 600,    600,    600],
    "Crane":            [9000000, 9000000, 9000000],
    "Storage":          [   0,      0,      0],
    "Other":            [   0,      0,      0],
    # IWW node values (used when IWW nodes are integrated)
    "Road":             [3000000,  3000000,  3000000],
    "Rail":             [17000000, 17000000, 17000000],
    "Electricity":      [1100000,  1100000,  1100000],
    "Power":            [300000000, 300000000, 300000000],
}

# land_use normalization map (port parquet → curve column names)
PORT_LAND_USE_NORMALIZE = {
    "industry": "Industry",
    "refinery": "Refinery",
}
