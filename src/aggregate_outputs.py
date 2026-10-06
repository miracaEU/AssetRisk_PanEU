"""
aggregate_outputs.py

Aggregate merged MIRACA output files to NUTS0, NUTS2, and LAU level,
following the dashboard data structure specification.

Addresses:
  1. Output folder is a sibling of MIRACA_OUTPUT (configurable)
  2. All levels get geometry via left-join to official boundary files
  3. Roads use road_class (from object_type mapping) not raw object_type
  4. All expected columns present for every system×geometry combination
  5. Geometry assignment uses SYSTEM_ASSET_TYPES, not raw geom_type
  6. Missing hazard data → NaN (not 0)
  7. All admin regions present (left join from boundaries), 0/NaN filled
  8. NUTS0 uses ISO2 country codes (CNTR_CODE), not ISO3 filenames
  9. Unexpected system×geometry combos filtered out
  10. Telecom included; spurious gas_points excluded

Output:
  {output_dir}/{NUTS0/NUTS2/LAU}_{system}_{polygons/lines/points}_hazards.parquet

Usage (from repo root):
  uv run python src/aggregate_outputs.py \
    --nuts-file data/NUTS_RG_01M_2024_3035.parquet \
    --lau-file data/LAU_RG_01M_2024_3035.parquet
  uv run python src/aggregate_outputs.py --systems roads power
"""

import argparse
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from constants import (
    EAD_HAZARDS,
    EXPOSURE_HAZARDS,
    ROAD_CLASS_MAP,
)

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


# ──────────────────────────────────────────────────────────────────────────
# Issue 5 & 9: Authoritative system × geometry → object_type mapping
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_ASSET_TYPES = {
    # TRANSPORT
    ("roads",      "lines"):    {"primary_roads", "secondary_roads", "tertiary_roads",
                                  "motorways_and_trunks", "other_roads"},
    ("rail",       "lines"):    {"rail", "narrow_gauge"},
    ("airports",   "lines"):    {"runway"},
    ("airports",   "polygons"): {"apron", "terminal", "aerodrome"},
    ("ports",      "polygons"): {"harbour", "port"},
    # ENERGY
    ("power",      "polygons"): {"substation", "generator", "plant"},
    ("power",      "lines"):    {"minor_line", "line", "cable"},
    ("power",      "points"):   {"pole", "tower", "terminal", "transformer",
                                  "portal", "switch", "catenary_mast"},
    ("gas",        "polygons"): {"storage_tank", "gasometer"},
    ("gas",        "lines"):    {"pipeline"},
    ("oil",        "polygons"): {"storage_tank", "petroleum_well", "oil_refinery"},
    ("oil",        "lines"):    {"pipeline"},
    # SOCIAL
    ("education",  "polygons"): {"school", "kindergarten", "college", "university"},
    ("healthcare", "polygons"): {"clinic", "hospital"},
    # TELECOM
    ("telecom",    "points"):   {"mast", "communications_tower", "tower"},
}

# Reverse lookup: (system, object_type) → geometry class
_OBJ_TO_GEOM = {}
for (sys, geom_class), obj_types in SYSTEM_ASSET_TYPES.items():
    for ot in obj_types:
        _OBJ_TO_GEOM[(sys, ot)] = geom_class

# Issue 3: Roads object_type → road_class mapping: constants.ROAD_CLASS_MAP

# Issue 8: ISO3 filename → ISO2 CNTR_CODE
ISO3_TO_ISO2 = {
    "AND": "AD",
    "ALB": "AL", "AUT": "AT", "BEL": "BE", "BGR": "BG", "BIH": "BA",
    "CHE": "CH", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK",
    "ESP": "ES", "EST": "EE", "FIN": "FI", "FRA": "FR", "GBR": "GB",
    "GRC": "EL", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ISL": "IS",
    "ITA": "IT", "LIE": "LI", "LTU": "LT", "LUX": "LU", "LVA": "LV",
    "MKD": "MK", "MLT": "MT", "MNE": "ME", "NLD": "NL", "NOR": "NO",
    "POL": "PL", "PRT": "PT", "ROU": "RO", "SRB": "RS", "SVK": "SK",
    "SVN": "SI", "SWE": "SE", "XKO": "XK",
}


def _to_iso2(code: str) -> str:
    """Convert ISO3 to ISO2. Returns input if already ISO2 or not found."""
    return ISO3_TO_ISO2.get(code.upper(), code.upper())


# Region column for each aggregation level
LEVEL_COLUMNS = {
    "NUTS0": "CNTR_CODE",
    "NUTS2": "NUTS2",
    "LAU": "LAU",
}

# ──────────────────────────────────────────────────────────────────────────
# Hazard columns we expect in the output
# ──────────────────────────────────────────────────────────────────────────

# EAD_HAZARDS, EXPOSURE_HAZARDS: imported from constants
PERIODS = ["current", "2050_SSP245", "2050_SSP585", "2100_SSP245", "2100_SSP585"]
EAD_BOUNDS = ["mid", "min", "max"]

# Not all hazards have future periods
EAD_FUTURE_HAZARDS = {"river", "coastal"}
EXPOSURE_FUTURE_HAZARDS = {"river", "coastal", "heat", "wildfire"}


def _expected_ead_columns(object_types: list[str]) -> list[str]:
    """Generate all expected EAD column names for given object types + total."""
    cols = []
    labels = list(object_types) + ["total"]
    for bound in EAD_BOUNDS:
        for label in labels:
            for hazard in EAD_HAZARDS:
                periods = PERIODS if hazard in EAD_FUTURE_HAZARDS else ["current"]
                for period in periods:
                    cols.append(f"EAD_{bound}_{label}_{hazard}_{period}")
    return cols


_SPREAD_EXPOSURE_HAZARDS = {"heat", "wildfire"}  # hazards with multi-model spread


def _expected_exposure_columns(object_types: list[str]) -> list[str]:
    """Generate all expected exposure column names for given object types + total."""
    cols = []
    labels = list(object_types) + ["total"]
    for kind in ["abs", "rel"]:
        for label in labels:
            for hazard in EXPOSURE_HAZARDS:
                periods = PERIODS if hazard in EXPOSURE_FUTURE_HAZARDS else ["current"]
                for period in periods:
                    cols.append(f"exposure_{kind}_{label}_{hazard}_{period}")
                    if period != "current" and hazard in _SPREAD_EXPOSURE_HAZARDS:
                        for bound in ["min", "max"]:
                            cols.append(f"exposure_{kind}_{bound}_{label}_{hazard}_{period}")
    return cols


def _expected_size_columns(object_types: list[str]) -> list[str]:
    """Generate asset_size and n_features columns."""
    labels = list(object_types) + ["total"]
    cols = []
    for label in labels:
        cols.append(f"asset_size_{label}")
        cols.append(f"n_features_{label}")
    return cols


def _expected_n_exposed_columns(object_types: list[str]) -> list[str]:
    """Generate n_exposed columns — count of assets with exposure > 0."""
    labels = list(object_types) + ["total"]
    cols = []
    for label in labels:
        for hazard in EXPOSURE_HAZARDS:
            periods = PERIODS if hazard in EXPOSURE_FUTURE_HAZARDS else ["current"]
            for period in periods:
                cols.append(f"n_exposed_{label}_{hazard}_{period}")
    return cols


# ──────────────────────────────────────────────────────────────────────────
# Column detection helpers
# ──────────────────────────────────────────────────────────────────────────

def _find_ead_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("EAD_")]


def _find_exposure_abs_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("exposure_abs_")]


def _parse_ead_suffix(col: str) -> tuple[str, str, str] | None:
    """Parse EAD_{bound}_{hazard}_{period} → (bound, hazard, period)."""
    parts = col.split("_", 2)
    if len(parts) < 3:
        return None
    bound = parts[1]
    remainder = parts[2]
    for h in EAD_HAZARDS:
        if remainder.startswith(h + "_"):
            return bound, h, remainder[len(h) + 1:]
    return None


def _parse_exposure_suffix(col: str) -> tuple[str, str, str] | None:
    """Parse exposure_abs_{hazard}_{period} → (bound, hazard, period).
    Also handles exposure_abs_min_{hazard}_{period} / _max_ variants.
    bound is 'mean' for the standard column, 'min' or 'max' for spread.
    """
    remainder = col[len("exposure_abs_"):]
    bound = "mean"
    for b in ("min_", "max_"):
        if remainder.startswith(b):
            bound = b.rstrip("_")
            remainder = remainder[len(b):]
            break
    for h in EXPOSURE_HAZARDS:
        if remainder.startswith(h + "_"):
            return bound, h, remainder[len(h) + 1:]
        if remainder == h:
            return bound, h, "current"
    return None


# ──────────────────────────────────────────────────────────────────────────
# Core aggregation
# ──────────────────────────────────────────────────────────────────────────

def attach_rel_columns(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """
    Compute exposure_rel_* = exposure_abs_* / asset_size_{label}.

    Matches each exposure_abs_* column against the known `labels` set
    (object_types + "total") instead of naive `remainder.split("_")[0]`
    parsing, which breaks for multi-word labels (e.g. "motorways_and_trunks",
    "minor_line") -- those would silently get no rel column, or worse, get
    matched against a wrong/missing asset_size_{first_word} column.

    Always recompute from abs/size rather than carrying forward an existing
    rel column -- summing pre-computed ratios across rows (e.g. when
    combining per-country chunks) is mathematically wrong and is exactly
    what produced the NUTS2 roads/power_lines >100% rel bug.
    """
    rel_parts = {}
    sorted_labels = sorted(set(labels), key=len, reverse=True)
    for col in df.columns:
        if not col.startswith("exposure_abs_"):
            continue
        remainder = col[len("exposure_abs_"):]
        for b in ("min_", "max_"):
            if remainder.startswith(b):
                remainder = remainder[len(b):]
                break
        label = next((l for l in sorted_labels if remainder.startswith(l + "_")), None)
        if label is None:
            continue
        size_col = f"asset_size_{label}"
        if size_col not in df.columns:
            continue
        rel_col = col.replace("exposure_abs_", "exposure_rel_")
        rel_series = pd.Series(np.nan, index=df.index)
        nonzero = df[size_col] > 0
        rel_series[nonzero] = df.loc[nonzero, col] / df.loc[nonzero, size_col]
        rel_parts[rel_col] = rel_series

    if not rel_parts:
        return df
    # Drop any stale exposure_rel_* (e.g. summed-ratio leftovers) before
    # attaching the freshly recomputed ones.
    stale = [c for c in df.columns if c.startswith("exposure_rel_")]
    if stale:
        df = df.drop(columns=stale)
    return pd.concat([df, pd.DataFrame(rel_parts, index=df.index)], axis=1)


def aggregate_group(
    gdf: gpd.GeoDataFrame,
    region_col: str,
    expected_object_types: set[str],
) -> pd.DataFrame:
    """
    Aggregate features by region, producing per-object_type and total columns.
    Only uses object types from the expected set.
    """
    ead_cols = _find_ead_cols(gdf.columns)
    exp_abs_cols = _find_exposure_abs_cols(gdf.columns)
    object_types = sorted(expected_object_types)

    result_parts = {}

    # ── asset_size and n_features per object_type + total ──
    for ot in object_types:
        mask = gdf["object_type"] == ot
        sub = gdf.loc[mask]
        result_parts[f"asset_size_{ot}"] = sub.groupby(region_col)["asset_size"].sum()
        result_parts[f"n_features_{ot}"] = sub.groupby(region_col)["osm_id"].count()

    result_parts["asset_size_total"] = gdf.groupby(region_col)["asset_size"].sum()
    result_parts["n_features_total"] = gdf.groupby(region_col)["osm_id"].count()

    # ── EAD columns: per object_type + total ──
    for col in ead_cols:
        parsed = _parse_ead_suffix(col)
        if parsed is None:
            continue
        bound, hazard, period = parsed

        for ot in object_types:
            mask = gdf["object_type"] == ot
            out_name = f"EAD_{bound}_{ot}_{hazard}_{period}"
            result_parts[out_name] = gdf.loc[mask].groupby(region_col)[col].sum()

        out_total = f"EAD_{bound}_total_{hazard}_{period}"
        result_parts[out_total] = gdf.groupby(region_col)[col].sum()

    # ── Exposure abs columns: per object_type + total ──
    for col in exp_abs_cols:
        parsed = _parse_exposure_suffix(col)
        if parsed is None:
            continue
        bound, hazard, period = parsed
        bi = f"{bound}_" if bound != "mean" else ""

        for ot in object_types:
            mask = gdf["object_type"] == ot
            out_name = f"exposure_abs_{bi}{ot}_{hazard}_{period}"
            result_parts[out_name] = gdf.loc[mask].groupby(region_col)[col].sum()

        out_total = f"exposure_abs_{bi}total_{hazard}_{period}"
        result_parts[out_total] = gdf.groupby(region_col)[col].sum()

    # ── n_exposed: count of assets with exposure > 0 (count-based metric) ──
    for col in exp_abs_cols:
        parsed = _parse_exposure_suffix(col)
        if parsed is None or parsed[0] != "mean":
            continue
        _, hazard, period = parsed

        exposed_flag = (gdf[col].fillna(0) > 0).astype(np.int8)

        for ot in object_types:
            mask = gdf["object_type"] == ot
            result_parts[f"n_exposed_{ot}_{hazard}_{period}"] = (
                exposed_flag[mask].groupby(gdf.loc[mask, region_col]).sum()
            )

        result_parts[f"n_exposed_total_{hazard}_{period}"] = (
            exposed_flag.groupby(gdf[region_col]).sum()
        )

    # Combine all series at once
    result = pd.concat(result_parts.values(), axis=1, keys=result_parts.keys())

    # ── Compute exposure_rel = exposure_abs / asset_size (all at once) ──
    result = attach_rel_columns(result, object_types + ["total"])

    return result


# ──────────────────────────────────────────────────────────────────────────
# Issue 6: Determine which countries have data for each hazard
# ──────────────────────────────────────────────────────────────────────────

def _detect_country_hazard_coverage(
    gdfs: list[gpd.GeoDataFrame],
) -> dict[str, set[str]]:
    """
    Detect which hazard data exists per country (vs genuinely missing).
    Returns {country_iso2: {hazards_with_data}}.
    """
    coverage: dict[str, set[str]] = {}
    for gdf in gdfs:
        if gdf is None or len(gdf) == 0:
            continue
        cc = gdf["CNTR_CODE"].iloc[0] if "CNTR_CODE" in gdf.columns else "??"

        hazards_present = set()
        # Check EAD columns — if any feature has non-NaN, hazard data exists
        for h in EAD_HAZARDS:
            col = f"EAD_mid_{h}_current"
            if col in gdf.columns and gdf[col].notna().any():
                hazards_present.add(h)

        # Check exposure columns
        for h in EXPOSURE_HAZARDS:
            col = f"exposure_abs_{h}_current"
            if col in gdf.columns and gdf[col].notna().any():
                hazards_present.add(h)

        coverage[cc] = hazards_present

    return coverage


def build_global_hazard_coverage(parquet_files: list) -> dict[str, set[str]]:
    """
    Hazard coverage per country across ALL systems (union), via cheap
    columnar reads of only the coverage-check columns.

    A country with no asset file for some system (e.g. CZ gas/oil, IS gas)
    never enters that system's per-run coverage, so its regions stayed NaN
    instead of 0. Global coverage tells us the hazard data exists for the
    country, so "no assets" can correctly become 0.
    """
    import pyarrow.parquet as pq

    check = ([("EAD", h, f"EAD_mid_{h}_current") for h in EAD_HAZARDS] +
             [("exp", h, f"exposure_abs_{h}_current") for h in EXPOSURE_HAZARDS])
    coverage: dict[str, set] = {}
    for path in parquet_files:
        cc = _to_iso2(path.stem.split("_")[0])
        try:
            schema_names = set(pq.read_schema(path).names)
            cols = [c for (_, _, c) in check if c in schema_names]
            if not cols:
                continue
            tbl = pq.read_table(path, columns=cols).to_pandas()
        except Exception as e:
            print(f"    coverage scan skip {path.name}: {e}")
            continue
        present = coverage.setdefault(cc, set())
        for _, h, c in check:
            if c in tbl.columns and tbl[c].notna().any():
                present.add(h)
    return coverage


# ──────────────────────────────────────────────────────────────────────────
# GBR 2016 boundary patch — fill NaN LAU/NUTS2 via spatial join
# ──────────────────────────────────────────────────────────────────────────

def _patch_gbr_regions(
    gdf: gpd.GeoDataFrame,
    gbr_lau: "gpd.GeoDataFrame | None",
    gbr_nuts2: "gpd.GeoDataFrame | None",
) -> gpd.GeoDataFrame:
    """Spatially join GBR features to 2016 boundaries to fill NaN LAU/NUTS2."""
    if gbr_lau is None and gbr_nuts2 is None:
        return gdf

    gdf = gdf.copy()
    if "LAU"   not in gdf.columns: gdf["LAU"]   = pd.NA
    if "NUTS2" not in gdf.columns: gdf["NUTS2"] = pd.NA

    mask = gdf["LAU"].isna() | gdf["NUTS2"].isna()
    if not mask.any():
        return gdf

    centroids = gpd.GeoDataFrame(
        index=gdf.index[mask],
        geometry=gdf.loc[mask, "geometry"].centroid,
        crs=gdf.crs,
    )

    def _sjoin_with_fallback(pts, bounds, id_col):
        joined = gpd.sjoin(pts, bounds[[id_col, "geometry"]],
                           how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        unmatched = joined[joined[id_col].isna()].index
        if len(unmatched):
            nearest = gpd.sjoin_nearest(
                pts.loc[unmatched], bounds[[id_col, "geometry"]], how="left"
            )
            nearest = nearest[~nearest.index.duplicated(keep="first")]
            joined.loc[unmatched, id_col] = nearest[id_col].values
        return joined[id_col]

    if gbr_lau is not None and gdf["LAU"].isna().any():
        lau_ids = _sjoin_with_fallback(centroids, gbr_lau, "LAU_ID")
        gdf.loc[mask, "LAU"] = lau_ids.reindex(gdf.index[mask]).values

    if gbr_nuts2 is not None and gdf["NUTS2"].isna().any():
        nuts2_ids = _sjoin_with_fallback(centroids, gbr_nuts2, "NUTS_ID")
        gdf.loc[mask, "NUTS2"] = nuts2_ids.reindex(gdf.index[mask]).values

    return gdf


# ──────────────────────────────────────────────────────────────────────────
# Process one system across all countries
# ──────────────────────────────────────────────────────────────────────────

def _load_and_prep(
    path: Path,
    system: str,
    gbr_lau: "gpd.GeoDataFrame | None" = None,
    gbr_nuts2: "gpd.GeoDataFrame | None" = None,
) -> gpd.GeoDataFrame:
    """Load one country parquet, assign CNTR_CODE and remap roads object_type."""
    gdf = gpd.read_parquet(path)
    country_iso3 = path.stem.split("_")[0]
    gdf["CNTR_CODE"] = _to_iso2(country_iso3)
    if country_iso3 == "GBR" and (gbr_lau is not None or gbr_nuts2 is not None):
        gdf = _patch_gbr_regions(gdf, gbr_lau, gbr_nuts2)
    if system == "roads" and "object_type" in gdf.columns:
        gdf["object_type"] = gdf["object_type"].map(
            lambda x: ROAD_CLASS_MAP.get(x, x.lower().replace(" ", "_"))
        )
    # Assign geom class and drop unexpected object types
    ot_col = gdf["object_type"] if "object_type" in gdf.columns else None
    if ot_col is not None:
        gdf["_geom_class"] = ot_col.map(lambda x: _OBJ_TO_GEOM.get((system, x)))
        gdf = gdf[gdf["_geom_class"].notna()].copy()
    return gdf


def process_system(
    parquet_files: list[Path],
    system: str,
    nuts_boundaries: dict[str, gpd.GeoDataFrame | pd.DataFrame],
    output_dir: Path,
    chunk: bool = False,
    global_coverage: "dict[str, set[str]] | None" = None,
    levels: "list[str] | None" = None,
) -> dict[str, int]:
    """
    Process all country files for one system.
    chunk=True: aggregate one country at a time — low peak memory, use for roads.
    chunk=False: load all countries at once (faster for small systems).

    Both paths converge to combined_aggs: {(geom_class, level_name): agg_df}
    then share the same boundary-join + write logic.

    levels: restrict to a subset of LEVEL_COLUMNS keys (e.g. ["NUTS2"]) for a
            targeted re-run; None = all levels (NUTS0, NUTS2, LAU).
    """
    expected_geom_classes = sorted({gc for (s, gc) in SYSTEM_ASSET_TYPES if s == system})
    country_hazard_coverage: dict[str, set] = {}
    active_levels = {k: v for k, v in LEVEL_COLUMNS.items() if levels is None or k in levels}

    gbr_lau   = nuts_boundaries.get("_gbr_lau")
    gbr_nuts2 = nuts_boundaries.get("_gbr_nuts2")

    # ── Phase 1: build combined_aggs ─────────────────────────────────────────

    if chunk:
        # Aggregate country by country; accumulate small result DataFrames
        agg_parts: dict[tuple, list[pd.DataFrame]] = {
            (gc, lv): [] for gc in expected_geom_classes for lv in active_levels
        }

        for path in parquet_files:
            print(f"      {path.stem}...")
            try:
                gdf = _load_and_prep(path, system, gbr_lau=gbr_lau, gbr_nuts2=gbr_nuts2)
            except Exception as e:
                print(f"      Skip {path.name}: {e}")
                continue

            country_hazard_coverage.update(_detect_country_hazard_coverage([gdf]))

            for geom_class in expected_geom_classes:
                subset = gdf[gdf["_geom_class"] == geom_class]
                if subset.empty:
                    continue
                expected_ots = SYSTEM_ASSET_TYPES.get((system, geom_class), set())
                for level_name, region_col in active_levels.items():
                    if region_col not in subset.columns:
                        continue
                    agg_parts[(geom_class, level_name)].append(
                        aggregate_group(subset, region_col, expected_ots)
                    )

        # Combine: different countries have different region codes → just concat + sum.
        # exposure_rel_* is a ratio, not summable -- a region that (rarely)
        # gets contributions from more than one country chunk would otherwise
        # sum two ratios together (can exceed 100%). Sum abs/size/EAD, then
        # recompute rel fresh from the summed totals.
        combined_aggs: dict[tuple, pd.DataFrame] = {}
        for key, parts in agg_parts.items():
            if not parts:
                continue
            stacked = pd.concat(parts)
            rel_cols = [c for c in stacked.columns if c.startswith("exposure_rel_")]
            num_cols = [c for c in stacked.select_dtypes("number").columns if c not in rel_cols]
            combined = stacked.groupby(stacked.index)[num_cols].sum()
            geom_class = key[0]
            expected_ots = SYSTEM_ASSET_TYPES.get((system, geom_class), set())
            combined_aggs[key] = attach_rel_columns(combined, sorted(expected_ots) + ["total"])

    else:
        # Load all countries at once
        gdfs = []
        for path in parquet_files:
            try:
                gdfs.append(_load_and_prep(path, system, gbr_lau=gbr_lau, gbr_nuts2=gbr_nuts2))
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
        if not gdfs:
            return {}

        country_hazard_coverage = _detect_country_hazard_coverage(gdfs)
        combined = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs
        )

        combined_aggs = {}
        for geom_class in expected_geom_classes:
            subset = combined[combined["_geom_class"] == geom_class].copy()
            if subset.empty:
                print(f"    {geom_class}: no features")
                continue
            expected_ots = SYSTEM_ASSET_TYPES.get((system, geom_class), set())
            for level_name, region_col in active_levels.items():
                if region_col not in subset.columns:
                    continue
                combined_aggs[(geom_class, level_name)] = aggregate_group(
                    subset, region_col, expected_ots
                )

    # ── Phase 2: boundary join + write ───────────────────────────────────────

    results: dict[str, int] = {}

    for (geom_class, level_name), agg in combined_aggs.items():
        expected_ots = SYSTEM_ASSET_TYPES.get((system, geom_class), set())

        # Build full expected column set (Issue 4)
        expected_cols = (
            _expected_size_columns(sorted(expected_ots))
            + _expected_n_exposed_columns(sorted(expected_ots))
            + _expected_ead_columns(sorted(expected_ots))
            + _expected_exposure_columns(sorted(expected_ots))
        )
        for col in expected_cols:
            if col not in agg.columns:
                agg[col] = np.nan
        ordered_cols = [c for c in expected_cols if c in agg.columns]
        extra_cols   = [c for c in agg.columns if c not in expected_cols]
        agg = agg[ordered_cols + extra_cols]

        # Issue 7 & 2: left-join boundaries so all regions are present
        if level_name in nuts_boundaries:
            boundaries = nuts_boundaries[level_name].copy()
            bid = {"NUTS0": "CNTR_CODE", "NUTS2": "NUTS_ID", "LAU": "LAU_ID"}.get(level_name)

            if bid and bid in boundaries.columns:
                merged = boundaries.set_index(bid).join(agg, how="left")

                # Issue 6: size/count → 0, hazard cols → 0 only if data exists
                size_count_cols = [c for c in merged.columns
                                   if c.startswith("asset_size_") or c.startswith("n_features_")
                                   or c.startswith("n_exposed_")]
                merged[size_count_cols] = merged[size_count_cols].fillna(0)

                if level_name == "NUTS0":
                    country_series = merged.index.to_series()
                elif "CNTR_CODE" in merged.columns:
                    country_series = merged["CNTR_CODE"]
                else:
                    country_series = merged.index.to_series().str[:2]

                hazard_cols_by_hazard: dict[str, list[str]] = {}
                for col in merged.columns:
                    if not (col.startswith("EAD_") or col.startswith("exposure_")):
                        continue
                    for h in EXPOSURE_HAZARDS + EAD_HAZARDS:
                        if f"_{h}_" in col:
                            hazard_cols_by_hazard.setdefault(h, []).append(col)
                            break

                countries_in_run = set(country_hazard_coverage.keys())
                for hazard_name, cols in hazard_cols_by_hazard.items():
                    countries_with_data = {
                        cc for cc, hazards in country_hazard_coverage.items()
                        if hazard_name in hazards
                    }
                    # Countries with no asset file for this system have no
                    # infrastructure — fill 0 wherever the hazard data exists
                    # (global coverage). Countries whose file exists but has
                    # all-NaN for a hazard keep NaN (genuinely missing data,
                    # e.g. GBR oil/gas landslide).
                    if global_coverage:
                        countries_with_data |= {
                            cc for cc, hazards in global_coverage.items()
                            if hazard_name in hazards and cc not in countries_in_run
                        }
                    fill_mask = country_series.isin(countries_with_data)
                    for col in cols:
                        merged.loc[fill_mask, col] = merged.loc[fill_mask, col].fillna(0)

                if "geometry" in merged.columns:
                    merged = gpd.GeoDataFrame(merged.copy(), geometry="geometry")
                else:
                    merged = merged.copy()
                merged = merged.reset_index()
            else:
                merged = agg.reset_index()
        else:
            merged = agg.reset_index()

        out_name = f"{level_name}_{system}_{geom_class}_hazards"
        merged.to_parquet(output_dir / f"{out_name}.parquet")
        results[out_name] = len(merged)
        print(f"    → {out_name}: {len(merged)} rows")

    return results


# ──────────────────────────────────────────────────────────────────────────
# Load boundary geometries
# ──────────────────────────────────────────────────────────────────────────

def load_boundaries(
    nuts_path: Path,
    lau_path: Path | None = None,
    nuts_2016_path: Path | None = None,
    lau_2016_path: Path | None = None,
) -> dict[str, gpd.GeoDataFrame | pd.DataFrame]:
    """Load NUTS and LAU boundary geometries, supplementing with 2016 UK data for GBR."""
    boundaries: dict = {}

    if nuts_path.exists():
        if nuts_path.suffix == ".parquet":
            nuts = gpd.read_parquet(nuts_path)
        else:
            nuts = gpd.read_file(nuts_path)

        boundaries["NUTS0"] = nuts[nuts["LEVL_CODE"] == 0][
            ["CNTR_CODE", "NAME_LATN", "geometry"]
        ].copy()

        boundaries["NUTS2"] = nuts[nuts["LEVL_CODE"] == 2][
            ["NUTS_ID", "CNTR_CODE", "NAME_LATN", "geometry"]
        ].copy()

    if lau_path and Path(lau_path).exists():
        lau_path = Path(lau_path)
        try:
            lau = gpd.read_parquet(lau_path)
        except Exception:
            try:
                lau = gpd.read_file(lau_path)
            except Exception:
                lau = pd.read_parquet(lau_path)
        if "GISCO_ID" in lau.columns:
            lau = lau.rename(columns={"GISCO_ID": "LAU_ID"})
        boundaries["LAU"] = lau

    # ── GBR supplement: 2016 boundaries for UK (absent from 2024 Eurostat data) ──
    gbr_lau_raw   = None
    gbr_nuts2_raw = None

    if nuts_2016_path and Path(nuts_2016_path).exists():
        try:
            print("  Loading 2016 NUTS for GBR supplement...")
            nuts16 = gpd.read_file(nuts_2016_path)
            uk16   = nuts16[nuts16["CNTR_CODE"] == "UK"].copy()
            keep   = [c for c in ["NUTS_ID", "LEVL_CODE", "CNTR_CODE", "NAME_LATN",
                                   "NUTS_NAME", "geometry"] if c in uk16.columns]
            uk16   = uk16[keep].copy()

            # NUTS0: remap UK → GB so it matches CNTR_CODE="GB" in feature files
            uk_nuts0 = uk16[uk16["LEVL_CODE"] == 0].copy()
            uk_nuts0["CNTR_CODE"] = "GB"
            if "NUTS0" in boundaries:
                boundaries["NUTS0"] = gpd.GeoDataFrame(
                    pd.concat([boundaries["NUTS0"], uk_nuts0[boundaries["NUTS0"].columns.intersection(uk_nuts0.columns)]],
                              ignore_index=True),
                    geometry="geometry", crs=boundaries["NUTS0"].crs,
                )

            # NUTS2: keep UK NUTS_IDs as-is (e.g. "UKC1") — matched via spatial join.
            # CNTR_CODE must be "GB" like NUTS0/LAU supplements and feature files
            # (_to_iso2("GBR")); "UK" broke the NaN→0 fill mask for zero-asset regions.
            uk_nuts2 = uk16[uk16["LEVL_CODE"] == 2].copy()
            uk_nuts2["CNTR_CODE"] = "GB"
            if "NUTS2" in boundaries:
                boundaries["NUTS2"] = gpd.GeoDataFrame(
                    pd.concat([boundaries["NUTS2"], uk_nuts2[boundaries["NUTS2"].columns.intersection(uk_nuts2.columns)]],
                              ignore_index=True),
                    geometry="geometry", crs=boundaries["NUTS2"].crs,
                )
            gbr_nuts2_raw = uk_nuts2[["NUTS_ID", "geometry"]].copy()
            print(f"    GBR NUTS2 regions added: {len(uk_nuts2)}")
        except Exception as e:
            print(f"  Warning: could not load 2016 NUTS supplement for GBR: {e}")

    if lau_2016_path and Path(lau_2016_path).exists():
        try:
            print("  Loading 2016 LAU for GBR supplement (this may take a moment)...")
            lau16   = gpd.read_file(lau_2016_path)
            uk_lau  = lau16[lau16["CNTR_CODE"] == "UK"].copy()
            uk_lau["CNTR_CODE"] = "GB"  # normalize to match _to_iso2("GBR") used in feature files
            # GISCO_ID ("UK_XXXXX") becomes LAU_ID — matches 2024 format.
            # 2016 file already has a local "LAU_ID" (numeric only); drop it first.
            if "GISCO_ID" in uk_lau.columns:
                if "LAU_ID" in uk_lau.columns:
                    uk_lau = uk_lau.drop(columns=["LAU_ID"])
                uk_lau = uk_lau.rename(columns={"GISCO_ID": "LAU_ID"})
            if "POP_2016" in uk_lau.columns:
                uk_lau = uk_lau.rename(columns={"POP_2016": "POP_2024",
                                                  "POP_DENS_2016": "POP_DENS_2024"})
            if "LAU" in boundaries:
                shared = boundaries["LAU"].columns.intersection(uk_lau.columns)
                boundaries["LAU"] = gpd.GeoDataFrame(
                    pd.concat([boundaries["LAU"], uk_lau[shared]], ignore_index=True),
                    geometry="geometry", crs=boundaries["LAU"].crs,
                )
            gbr_lau_raw = uk_lau[["LAU_ID", "geometry"]].copy()
            print(f"    GBR LAU regions added: {len(uk_lau)}")
        except Exception as e:
            print(f"  Warning: could not load 2016 LAU supplement for GBR: {e}")

    # Store raw GDFs for spatial patching in _load_and_prep
    boundaries["_gbr_lau"]   = gbr_lau_raw
    boundaries["_gbr_nuts2"] = gbr_nuts2_raw

    return boundaries


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate MIRACA outputs for the dashboard viewer"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory with merged parquets (default: MIRACA_OUTPUT/)",
    )
    parser.add_argument(
        "--nuts-file", default=None,
        help="Path to NUTS boundary file",
    )
    parser.add_argument(
        "--lau-file", default=None,
        help="Path to LAU boundary file",
    )
    parser.add_argument(
        "--nuts-file-2016", default=None,
        help="Path to 2016 NUTS boundary file for GBR supplement",
    )
    parser.add_argument(
        "--lau-file-2016", default=None,
        help="Path to 2016 LAU boundary file for GBR supplement",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: MIRACA_AGGREGATED/ sibling to MIRACA_OUTPUT)",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Only aggregate these systems (e.g. --systems roads power)",
    )
    parser.add_argument(
        "--levels", nargs="+", default=None, choices=["NUTS0", "NUTS2", "LAU"],
        help="Only aggregate these levels (e.g. --levels NUTS2). Default: all 3.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = repo_root / "MIRACA_OUTPUT_FULL"

    # Issue 1: Output as sibling folder
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / "MIRACA_AGGREGATED_FULL"

    output_dir.mkdir(parents=True, exist_ok=True)

    nuts_path      = Path(args.nuts_file)      if args.nuts_file      else repo_root / "data" / "NUTS_RG_01M_2024_3035.parquet"
    lau_path       = Path(args.lau_file)       if args.lau_file       else repo_root / "data" / "LAU_RG_01M_2024_3035.parquet"
    nuts_2016_path = Path(args.nuts_file_2016) if args.nuts_file_2016 else repo_root / "data" / "NUTS_RG_01M_2016_3035.geojson"
    lau_2016_path  = Path(args.lau_file_2016)  if args.lau_file_2016  else repo_root / "data" / "LAU_RG_01M_2016_3035.geojson"

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"NUTS file:  {nuts_path}")
    if lau_path:
        print(f"LAU file:   {lau_path}")
    print()

    boundaries = load_boundaries(nuts_path, lau_path, nuts_2016_path, lau_2016_path)
    for level, gdf in boundaries.items():
        if level.startswith("_") or gdf is None:
            continue
        print(f"  {level}: {len(gdf)} boundary regions")
    print()

    # Discover merged parquets grouped by system
    parquet_files = sorted([
        f for f in input_dir.glob("*.parquet")
        if not f.stem.startswith(("summary", "aggregate", "power_diag", "NUTS", "LAU"))
    ])

    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    # Group by system
    systems: dict[str, list[Path]] = {}
    for path in parquet_files:
        parts = path.stem.split("_", 1)
        if len(parts) < 2:
            continue
        system = parts[1]
        systems.setdefault(system, []).append(path)

    # Filter systems if requested
    if args.systems:
        systems = {k: v for k, v in systems.items() if k in args.systems}

    print(f"Found {len(parquet_files)} files, processing {len(systems)} systems\n")

    # Global hazard coverage across ALL systems (not just the ones requested),
    # so zero-infrastructure countries get 0 instead of NaN
    print("Scanning global hazard coverage across all country files...")
    global_coverage = build_global_hazard_coverage(parquet_files)
    print(f"  Coverage for {len(global_coverage)} countries\n")

    t0 = time.time()
    total_outputs = 0

    CHUNK_SYSTEMS = {"roads", "rail", "power"}  # large systems → low-memory path

    for system in sorted(systems.keys()):
        files = systems[system]
        use_chunk = system in CHUNK_SYSTEMS
        print(f"  {system} ({len(files)} countries){'  [chunked]' if use_chunk else ''}...")

        results = process_system(files, system, boundaries, output_dir,
                                 chunk=use_chunk,
                                 global_coverage=global_coverage,
                                 levels=args.levels)
        total_outputs += len(results)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {total_outputs} output files in {output_dir}")


if __name__ == "__main__":
    main()
