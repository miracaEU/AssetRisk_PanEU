"""
figures.py

Generate all main paper figures for the pan-European multi-hazard
CI risk assessment paper (ERL submission).

Function names, CLI keys and output filenames follow the figure numbers used
in the manuscript, so `--fig 3` produces what the paper calls Figure 3.

Main text:
  1. Exposure % + EAD by hazard × system, plus EAD by country
  2. LAU map: dominant hazard and dominant CI system by EAD
  3. Heat + wildfire exposure: SSP trajectories and LAU maps
  4. River + coastal flood risk: SSP trajectories and NUTS2 change maps

Supplementary:
  S1. NUTS2 version of Figure 2 (dominant hazard + system)
  S2. Multi-hazard exposure map (count-based, 3 thresholds)
  S3. Uncertainty map per NUTS2
  S4. Uncertainty heatmap by subsystem
  S5. Subsystem drivers
  S6. Dumbbell: climate model range by country
  S7. Future changes across all hazards
  S8. Sector vulnerability profiles
  S9. 2nd/3rd dominant CI system

Usage (from repo root, run on HPC):
  uv run python src/figures.py
  uv run python src/figures.py --output-dir /path/to/figures
  uv run python src/figures.py --fig 3   # single main figure
  uv run python src/figures.py --fig s1  # single supplementary figure
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Figure 1 colour system ─────────────────────────────────────────────
# Panel A (heatmap): sequential lajolla, reversed (pale = low %, dark = high %)
# Panels B–E (categorical): vivid hazards vs muted sectors — two distinct families

# --- Sequential map for Panel A ---
try:
    from cmcrameri import cm as _cmc
    LAJOLLA = _cmc.lajolla_r          # smooth 256-colour, reversed (low→pale, high→dark)
except ImportError:                    # fallback: 9-stop hardcoded lajolla (light→dark)
    _lajolla_stops = ["#FFFECB", "#F7DA74", "#EDAE54", "#E58851", "#D9604E",
                      "#A64644", "#67342A", "#372411", "#191900"]
    LAJOLLA = LinearSegmentedColormap.from_list("lajolla_r", _lajolla_stops, N=256)


def _label_colour(norm_value):
    """Readable in-cell label colour for the heatmap (dark on pale, light on dark)."""
    r, g, b, _ = LAJOLLA(norm_value)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b      # relative luminance
    return "white" if lum < 0.5 else "#1A1A1A"

# Hazard display order + colours
HAZARD_ORDER = ["windstorm", "coastal", "river", "earthquake",
                "heat", "wildfire", "landslide"]
HAZARD_LABELS = {
    "windstorm":  "Windstorm",
    "coastal":    "Coastal flood",
    "river":      "River flood",
    "earthquake": "Earthquake",
    "heat":       "Extreme heat",
    "wildfire":   "Wildfire",
    "landslide":  "Landslide",
}
# EAD hazards (windstorm/coastal/river/earthquake): vivid/saturated, per the
# Fig 1 colour system above. heat/wildfire/landslide keep the older
# Okabe-Ito-based colours (no EAD figure needs them alongside the vivid set).
HAZARD_COLORS = {
    "windstorm":  "#4477AA",
    "coastal":    "#CC3311",
    "river":      "#EE7733",
    "earthquake": "#4CB944",
    "heat":       "#E69F00",
    "wildfire":   "#D4A017",
    "landslide":  "#009E73",
}

# Emissions-scenario colours for trajectory panels (Figs 3, 4, 9).
# Constrained on three sides: blue is "EAD declines" on the Fig 4 change maps,
# grey is "no data / not applicable" in every map legend, and red is SSP5-8.5.
# Purple is unused elsewhere in the paper, so it reads as a distinct category
# rather than as a position on either scale.
SCENARIO_COLORS = {
    "SSP245": "#6A3D9A",   # purple
    "SSP585": "#F44336",
}

# System display order + labels
SYSTEM_ORDER = ["power", "roads", "rail", "airports", "ports",
                "education", "healthcare", "telecom"]
SYSTEM_LABELS = {
    "power":      "Power",
    "roads":      "Roads",
    "rail":       "Rail",
    "airports":   "Airports",
    "ports":      "Ports",
    "education":  "Education",
    "healthcare": "Healthcare",
    "telecom":    "Telecom",
}

# Muted / lower-chroma sector palette, per the Fig 1 colour system above —
# distinct family from the vivid HAZARD_COLORS so hazard- and sector-coloured
# bars never get confused.
SYSTEM_COLORS = {
    "power":      "#332288",
    "roads":      "#88CCEE",
    "rail":       "#44AA99",
    "airports":   "#117733",
    "ports":      "#DDCC77",
    "education":  "#999933",
    "healthcare": "#882255",
    "telecom":    "#AA4499",
}

EAD_HAZARDS = ["windstorm", "coastal", "river", "earthquake"]
ALL_HAZARDS = HAZARD_ORDER

# Oil and gas are excluded from every paper figure (constants.EXCLUDED_SYSTEMS)
from constants import EXCLUDED_SYSTEMS  # noqa: E402

# Object-type groupings per system (raw → display label)
SUBSYSTEM_GROUPS = {
    "power": {
        "Generation":    ["plant", "generator"],
        "Transmission":  ["line", "minor_line", "cable", "tower", "pole", "portal", "catenary_mast"],
        "Distribution":  ["substation", "transformer", "switch", "terminal"],
    },
    # roads object_type is already pre-grouped in the parquets
    "roads": {
        "Motorway/Trunk": ["motorways_and_trunks"],
        "Primary":        ["primary_roads"],
        "Secondary":      ["secondary_roads"],
        "Tertiary":       ["tertiary_roads"],
        "Other":          ["other_roads"],
    },
    "education": {
        "Higher":  ["college", "university"],
        "Primary": ["school", "kindergarten"],
    },
    "telecom": {
        "Towers": ["mast", "tower", "communications_tower"],
    },
    # single-level systems — keep object_type as label
    "rail":       None,
    "healthcare": None,
    "airports":   None,
    "ports":      None,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(miraca_dir: Path) -> pd.DataFrame:
    path = miraca_dir / "summary_statistics.csv"
    if not path.exists():
        raise FileNotFoundError(f"summary_statistics.csv not found at {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure 1: Multi-panel heatmap — exposure % and EAD M€
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 1 (combined): exposure heatmap + EAD by sector + EAD by country
# ---------------------------------------------------------------------------

def fig1_overview(df: pd.DataFrame, out_path: Path, presentation: bool = False):
    """
    Composite opening figure:
      A. Exposure heatmap — % assets exposed, all 7 hazards × 8 systems
      B. EAD per hazard, stacked by sector
      C. EAD per sector, stacked by hazard
      D. EAD per country (large) — paired bars: left = by hazard, right = by sector
      E. EAD per country (remaining) — same, independent y-axis
    """

    # --- Panel A data: mean pct_exposed across countries, current, mean bound ---
    exp = df[
        (df["metric_type"] == "exposure_rel") &
        (df["period"] == "current") &
        (df["metric_bound"] == "mean")
    ]
    exp_mat = pd.DataFrame(index=ALL_HAZARDS, columns=SYSTEM_ORDER, dtype=float)
    for h in ALL_HAZARDS:
        for s in SYSTEM_ORDER:
            vals = exp[(exp["hazard"] == h) & (exp["system"] == s)]["pct_exposed"]
            exp_mat.loc[h, s] = vals.mean() if len(vals) > 0 else np.nan

    # --- EAD data: current, mid bound ---
    ead = df[
        (df["metric_type"] == "EAD") &
        (df["period"] == "current") &
        (df["metric_bound"] == "mid")
    ]

    # Panel B: hazard × sector matrix (M€)
    mat_b = pd.DataFrame(0.0, index=EAD_HAZARDS, columns=SYSTEM_ORDER)
    for h in EAD_HAZARDS:
        for s in SYSTEM_ORDER:
            v = ead[(ead["hazard"] == h) & (ead["system"] == s)]["sum"]
            mat_b.loc[h, s] = v.sum() / 1e6

    # Panel C: sector × hazard matrix (M€)
    mat_c = mat_b.T.copy()

    # Panels D/E: country × hazard and country × sector, descending total EAD.
    # Drop countries with negligible total (e.g. AD, LIE).
    country_totals = (ead.groupby("country")["sum"].sum() / 1e6
                      ).sort_values(ascending=False)
    country_totals = country_totals[country_totals >= 1.0]
    country_order = country_totals.index.tolist()

    mat_d_haz = pd.DataFrame(0.0, index=country_order, columns=EAD_HAZARDS)
    mat_d_sys = pd.DataFrame(0.0, index=country_order, columns=SYSTEM_ORDER)
    for c in country_order:
        sub_c = ead[ead["country"] == c]
        for h in EAD_HAZARDS:
            mat_d_haz.loc[c, h] = sub_c[sub_c["hazard"] == h]["sum"].sum() / 1e6
        for s in SYSTEM_ORDER:
            mat_d_sys.loc[c, s] = sub_c[sub_c["system"] == s]["sum"].sum() / 1e6

    system_colors = SYSTEM_COLORS

    # Split countries: big (≥15% of largest total) left, rest right
    cutoff = 0.15 * country_totals.max()
    big = [c for c in country_order if country_totals[c] >= cutoff]
    small = [c for c in country_order if country_totals[c] < cutoff]

    # --- Layout: heatmap top (full width), B + C middle, D.1 + D.2 bottom ---
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 0.95],
                          hspace=0.32, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    gs_d = gs[2, :].subgridspec(1, 2, wspace=0.12,
                                width_ratios=[max(len(big), 1), max(len(small), 1)])
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    # --- Panel A: exposure heatmap ---
    data = exp_mat.values.astype(float)
    im = ax_a.imshow(data, cmap=LAJOLLA, aspect="auto", vmin=0, vmax=100)
    ax_a.set_xticks(range(len(SYSTEM_ORDER)))
    ax_a.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEM_ORDER], fontsize=15)
    ax_a.set_yticks(range(len(ALL_HAZARDS)))
    ax_a.set_yticklabels([HAZARD_LABELS[h] for h in ALL_HAZARDS], fontsize=15)
    if presentation:
        ax_a.set_title("A.  Proportion of assets exposed to each hazard (%)",
                       pad=6, fontsize=17, fontweight="bold", loc="center")
    else:
        ax_a.text(-0.02, 1.02, "A", transform=ax_a.transAxes, fontsize=17,
                  fontweight="bold", va="bottom", ha="left")
    # Inset colorbar so the heatmap box stays full width, aligned with rows below
    cax = ax_a.inset_axes((1.012, 0.05, 0.012, 0.90))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% of assets exposed", fontsize=14)
    cb.ax.tick_params(labelsize=13)

    for i, h in enumerate(ALL_HAZARDS):
        for j, s in enumerate(SYSTEM_ORDER):
            val = data[i, j]
            if not np.isnan(val):
                ax_a.text(j, i, f"{val:.0f}", ha="center", va="center",
                          fontsize=12, color=_label_colour(val / 100))

    # --- Panel B: EAD per hazard, stacked by sector (descending top→bottom) ---
    # _stacked_barh draws rows bottom-up, so ascending order puts largest on top
    # --- Panel B: EAD per sector, stacked by hazard (descending top→bottom) ---
    # Hazard-coloured stacks on the left, matching the left bar of each pair in D/E
    sec_order = mat_c.sum(axis=1).sort_values(ascending=True).index.tolist()
    _stacked_barh(
        ax_b, mat_c, sec_order,
        [SYSTEM_LABELS[s] for s in sec_order],
        HAZARD_COLORS, HAZARD_LABELS,
        "Total EAD (M€)", "B",
        "EAD per sector, broken down by hazard", presentation,
        annot_fontsize=10,
    )
    ax_b.legend(loc="lower right", framealpha=0.9, borderpad=0.6,
                fontsize=13)
    ax_b.tick_params(labelsize=14)
    ax_b.xaxis.label.set_size(15)

    # --- Panel C: EAD per hazard, stacked by sector (descending top→bottom) ---
    haz_order = mat_b.sum(axis=1).sort_values(ascending=True).index.tolist()
    _stacked_barh(
        ax_c, mat_b, haz_order,
        [HAZARD_LABELS[h] for h in haz_order],
        system_colors, SYSTEM_LABELS,
        "Total EAD (M€)", "C",
        "EAD per hazard, broken down by sector", presentation,
        annot_fontsize=10,
    )
    ax_c.legend(loc="lower right", framealpha=0.9, ncol=2, borderpad=0.6,
                fontsize=13)
    ax_c.tick_params(labelsize=14)
    ax_c.xaxis.label.set_size(15)

    # --- Panels D.1/D.2: paired bars per country — left by hazard, right by sector ---
    def _draw_country_pairs(ax, countries, letter, subtitle):
        x = np.arange(len(countries))
        width = 0.40
        off = width / 2 + 0.04  # small white gap between the two bars of a pair

        bottoms = np.zeros(len(countries))
        for h in EAD_HAZARDS:
            vals = mat_d_haz.loc[countries, h].values
            ax.bar(x - off, vals, bottom=bottoms, width=width,
                   color=HAZARD_COLORS[h])
            bottoms += vals

        bottoms = np.zeros(len(countries))
        for s in SYSTEM_ORDER:
            vals = mat_d_sys.loc[countries, s].values
            ax.bar(x + off, vals, bottom=bottoms, width=width,
                   color=system_colors[s])
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=14)
        ax.set_xlim(-0.7, len(countries) - 0.3)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        if presentation:
            ax.set_title(f"{letter}.  {subtitle}", pad=4, fontsize=16,
                         fontweight="bold", loc="center")
        else:
            ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
                    fontsize=17, fontweight="bold", va="bottom", ha="left")

    _draw_country_pairs(ax_d1, big, "D",
                        "EAD per country — left bar by hazard, right by sector")
    _draw_country_pairs(ax_d2, small, "E", "Remaining countries (note y-axis)")
    ax_d1.set_ylabel("Total EAD (M€)", fontsize=15)

    fig.savefig(out_path)
    print(f"  Fig 1 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: LAU map — dominant hazard by total EAD
# ---------------------------------------------------------------------------

def fig2_lau_map(agg_dir: Path, out_path: Path, presentation: bool = False):
    """Two-panel LAU choropleth: A = dominant hazard, B = dominant CI system."""

    print("  Loading LAU aggregated files...")

    system_colors = SYSTEM_COLORS

    lau_hazard  = {}  # {lau_id: {hazard: ead}}
    lau_system  = {}  # {lau_id: {system: ead}}
    geom_df = None

    lau_files = sorted(agg_dir.glob("LAU_*_hazards.parquet"))
    if not lau_files:
        print(f"  No LAU files found in {agg_dir}")
        return

    print(f"  Found {len(lau_files)} LAU files")
    for path in lau_files:
        # Extract system: LAU_{system}_{geometry}_hazards.parquet
        part = path.stem.replace("LAU_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue
        system = part if part in SYSTEM_ORDER else None

        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            try:
                gdf = pd.read_parquet(path)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
                continue

        id_col = next((c for c in ["LAU_ID", "LAU_CODE", "GEO_ID", "id", "GISCO_ID"]
                       if c in gdf.columns), None)
        if id_col is None:
            id_col = gdf.columns[0]

        if geom_df is None and "geometry" in gdf.columns:
            geom_df = gdf[[id_col, "geometry"]].copy()
            geom_df = geom_df.rename(columns={id_col: "LAU_ID"})

        for _, row in gdf.iterrows():
            lau_id = row[id_col]

            # Accumulate per-hazard EAD
            for h in EAD_HAZARDS:
                col = f"EAD_mid_total_{h}_current"
                if col not in gdf.columns:
                    col = next((c for c in gdf.columns
                                if "EAD_mid" in c and h in c and "current" in c), None)
                if col is None:
                    continue
                val = float(row[col]) if not pd.isna(row[col]) else 0.0
                if lau_id not in lau_hazard:
                    lau_hazard[lau_id] = {hh: 0.0 for hh in EAD_HAZARDS}
                lau_hazard[lau_id][h] = lau_hazard[lau_id].get(h, 0.0) + val

            # Accumulate per-system EAD — sum across all 4 EAD hazards
            if system:
                sys_total = 0.0
                for h in EAD_HAZARDS:
                    col = f"EAD_mid_total_{h}_current"
                    if col not in gdf.columns:
                        col = next((c for c in gdf.columns
                                    if "EAD_mid" in c and h in c and "current" in c), None)
                    if col:
                        sys_total += float(row[col]) if not pd.isna(row[col]) else 0.0
                if lau_id not in lau_system:
                    lau_system[lau_id] = {ss: 0.0 for ss in SYSTEM_ORDER}
                lau_system[lau_id][system] = lau_system[lau_id].get(system, 0.0) + sys_total

    if not lau_hazard or geom_df is None:
        print("  Could not build LAU EAD lookup — check aggregated file structure")
        return

    # --- Hazard dominant ---
    haz_df = pd.DataFrame.from_dict(lau_hazard, orient="index").fillna(0)
    haz_df.index.name = "LAU_ID"
    haz_df["total"] = haz_df[EAD_HAZARDS].sum(axis=1)
    haz_df["dominant_hazard"] = haz_df[EAD_HAZARDS].idxmax(axis=1)
    haz_df.loc[haz_df["total"] == 0, "dominant_hazard"] = "none"
    haz_df = haz_df.reset_index()

    # --- System dominant ---
    sys_df = pd.DataFrame.from_dict(lau_system, orient="index").fillna(0)
    sys_df.index.name = "LAU_ID"
    present = [s for s in SYSTEM_ORDER if s in sys_df.columns]
    sys_df["total_sys"] = sys_df[present].sum(axis=1)
    sys_df["dominant_system"] = sys_df[present].idxmax(axis=1)
    sys_df.loc[sys_df["total_sys"] == 0, "dominant_system"] = "none"
    sys_df = sys_df.reset_index()

    map_gdf = (geom_df
               .merge(haz_df[["LAU_ID", "dominant_hazard"]], on="LAU_ID", how="left")
               .merge(sys_df[["LAU_ID", "dominant_system"]], on="LAU_ID", how="left"))
    map_gdf["dominant_hazard"]  = map_gdf["dominant_hazard"].fillna("none")
    map_gdf["dominant_system"]  = map_gdf["dominant_system"].fillna("none")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception as e:
            print(f"  Warning: could not load NE background: {e}")

    def _draw_base(ax):
        ax.set_facecolor("#C8DCE8")
        ax.set_aspect("equal")
        if ne_gdf is not None:
            ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                        linewidth=0.3, zorder=1)
        ax.set_xlim(EUROPE_XLIM)
        ax.set_ylim(EUROPE_YLIM)
        ax.set_axis_off()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 10),
                                      gridspec_kw={"wspace": 0.04})

    # --- Panel A: dominant hazard ---
    _draw_base(ax_a)
    haz_cmap = {h: HAZARD_COLORS[h] for h in EAD_HAZARDS}
    haz_cmap["none"] = "#E8E8E8"
    for val, color in haz_cmap.items():
        sub = map_gdf[map_gdf["dominant_hazard"] == val]
        if len(sub):
            sub.plot(ax=ax_a, color=color, linewidth=0, edgecolor="none", zorder=2)

    haz_patches = [mpatches.Patch(color=HAZARD_COLORS[h], label=HAZARD_LABELS[h])
                   for h in EAD_HAZARDS]
    haz_patches.append(mpatches.Patch(color="#E8E8E8", label="No EAD"))
    ax_a.legend(handles=haz_patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                framealpha=0.95, fontsize=10, borderpad=1.0,
                handlelength=1.5, handleheight=1.2)

    if presentation:
        ax_a.set_title("A.  Dominant hazard (EAD, current)",
                       pad=6, fontsize=14, fontweight="bold", loc="center")
    else:
        ax_a.text(-0.02, 1.02, "A", transform=ax_a.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="left")

    # --- Panel B: dominant system ---
    _draw_base(ax_b)
    sys_cmap = {s: system_colors[s] for s in SYSTEM_ORDER}
    sys_cmap["none"] = "#E8E8E8"
    for val, color in sys_cmap.items():
        sub = map_gdf[map_gdf["dominant_system"] == val]
        if len(sub):
            sub.plot(ax=ax_b, color=color, linewidth=0, edgecolor="none", zorder=2)

    sys_patches = [mpatches.Patch(color=system_colors[s], label=SYSTEM_LABELS[s])
                   for s in SYSTEM_ORDER]
    sys_patches.append(mpatches.Patch(color="#E8E8E8", label="No EAD"))
    ax_b.legend(handles=sys_patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                framealpha=0.95, fontsize=10, borderpad=1.0,
                handlelength=1.5, handleheight=1.2)

    if presentation:
        ax_b.set_title("B.  Dominant CI system (EAD, current)",
                       pad=6, fontsize=14, fontweight="bold", loc="center")
    else:
        ax_b.text(-0.02, 1.02, "B", transform=ax_b.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="left")

    fig.savefig(out_path)
    print(f"  Fig 2 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Multi-panel EAD breakdown — hazard, system, country
# ---------------------------------------------------------------------------

def _stacked_barh(ax, mat, row_order, row_labels, stack_colors, stack_labels,
                  xlabel, letter, presentation_title=None, presentation=False,
                  annotate_totals=True, err_min=None, err_max=None,
                  annot_fontsize=6.5):
    """Reusable horizontal stacked bar helper."""
    y = np.arange(len(row_order))
    lefts = np.zeros(len(row_order))
    for key, color in stack_colors.items():
        if key not in mat.columns:
            continue
        vals = mat.reindex(row_order)[key].fillna(0).values
        ax.barh(y, vals, left=lefts, color=color,
                label=stack_labels.get(key, key), height=0.7)
        lefts += vals
    # Uncertainty range markers
    if err_min is not None and err_max is not None:
        for i, row in enumerate(row_order):
            mn = err_min.get(row, np.nan)
            mx = err_max.get(row, np.nan)
            if not (np.isnan(mn) or np.isnan(mx)):
                ax.plot([mn, mx], [i, i], "-", color="#111111", lw=1.5, zorder=5)
                for xv in [mn, mx]:
                    ax.plot([xv, xv], [i - 0.28, i + 0.28], "-",
                            color="#111111", lw=1.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    if presentation and presentation_title:
        ax.set_title(f"{letter}.  {presentation_title}", pad=4, fontsize=16,
                     fontweight="bold", loc="center")
    else:
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
    if annotate_totals:
        totals = mat.reindex(row_order).fillna(0).sum(axis=1).values
        xmax = ax.get_xlim()[1]
        for i, t in enumerate(totals):
            if t > 0:
                ax.text(t + xmax * 0.01, i, f"{t:,.0f}",
                        va="center", fontsize=annot_fontsize, color="#444444")


def _stacked_bar(ax, mat, col_order, stack_colors, stack_labels,
                 ylabel, letter, show_xlabels=True,
                 presentation_title=None, presentation=False,
                 title_y=1.02):
    """Vertical stacked bar helper — categories on x-axis."""
    x = np.arange(len(col_order))
    bottoms = np.zeros(len(col_order))
    for key, color in stack_colors.items():
        if key not in mat.columns:
            continue
        vals = mat.reindex(col_order)[key].fillna(0).values
        ax.bar(x, vals, bottom=bottoms, color=color,
               label=stack_labels.get(key, key), width=0.8)
        bottoms += vals
    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(col_order, rotation=90, ha="center")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    if presentation and presentation_title:
        ax.set_title(f"{letter}.  {presentation_title}", y=title_y,
                     fontsize=16, fontweight="bold", loc="center")
    else:
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")


def figS8_sector_profiles(df: pd.DataFrame, out_path: Path, presentation: bool = False):
    """4-panel EAD breakdown: by hazard, by system, by country×hazard, by country×system."""

    ead = df[
        (df["metric_type"] == "EAD") &
        (df["period"] == "current") &
        (df["metric_bound"] == "mid")
    ].copy()
    # --- Panel A: EAD per hazard, stacked by system (M€) ---
    mat_a = pd.DataFrame(0.0, index=EAD_HAZARDS, columns=SYSTEM_ORDER)
    for h in EAD_HAZARDS:
        for s in SYSTEM_ORDER:
            v = ead[(ead["hazard"] == h) & (ead["system"] == s)]["sum"]
            mat_a.loc[h, s] = v.sum() / 1e6

    # --- Panel B: EAD per system, stacked by hazard (M€) ---
    mat_b = pd.DataFrame(0.0, index=SYSTEM_ORDER, columns=EAD_HAZARDS)
    for s in SYSTEM_ORDER:
        for h in EAD_HAZARDS:
            v = ead[(ead["system"] == s) & (ead["hazard"] == h)]["sum"]
            mat_b.loc[s, h] = v.sum() / 1e6

    # --- Panels C & D: per country, descending total EAD left→right ---
    countries = sorted(ead["country"].unique())
    country_totals = (ead.groupby("country")["sum"].sum() / 1e6).sort_values(ascending=False)
    country_order = country_totals.index.tolist()

    mat_c = pd.DataFrame(0.0, index=countries, columns=EAD_HAZARDS)
    for c in countries:
        for h in EAD_HAZARDS:
            v = ead[(ead["country"] == c) & (ead["hazard"] == h)]["sum"]
            mat_c.loc[c, h] = v.sum() / 1e6

    mat_d = pd.DataFrame(0.0, index=countries, columns=SYSTEM_ORDER)
    for c in countries:
        for s in SYSTEM_ORDER:
            v = ead[(ead["country"] == c) & (ead["system"] == s)]["sum"]
            mat_d.loc[c, s] = v.sum() / 1e6

    system_colors = SYSTEM_COLORS

    # --- Layout: 3 rows, 2 cols; rows 1+2 span full width, share x-axis ---
    # Use nested gridspec so row-0 ↔ row-1 gap is larger than row-1 ↔ row-2
    fig = plt.figure(figsize=(16, 16))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.18)
    gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.32)
    gs_bot = gs_outer[1].subgridspec(2, 1, hspace=0.05)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_d = fig.add_subplot(gs_bot[1, 0], sharex=ax_c)

    _stacked_barh(
        ax_a, mat_a, EAD_HAZARDS,
        [HAZARD_LABELS[h] for h in EAD_HAZARDS],
        system_colors, SYSTEM_LABELS,
        "Total EAD (M€)", "A",
        "EAD per hazard, broken down by sector", presentation,
    )
    ax_a.legend(loc="upper right", framealpha=0.9, ncol=2, borderpad=0.6)

    _stacked_barh(
        ax_b, mat_b, SYSTEM_ORDER,
        [SYSTEM_LABELS[s] for s in SYSTEM_ORDER],
        HAZARD_COLORS, HAZARD_LABELS,
        "Total EAD (M€)", "B",
        "EAD per sector, broken down by hazard", presentation,
    )
    ax_b.legend(loc="upper right", framealpha=0.9, borderpad=0.6)

    _stacked_bar(
        ax_c, mat_c, country_order,
        HAZARD_COLORS, HAZARD_LABELS,
        "Total EAD (M€)", "C",
        show_xlabels=False,
        presentation_title="EAD per country, broken down by hazard",
        presentation=presentation,
        title_y=0.91,
    )

    _stacked_bar(
        ax_d, mat_d, country_order,
        system_colors, SYSTEM_LABELS,
        "Total EAD (M€)", "D",
        show_xlabels=True,
        presentation_title="EAD per country, broken down by sector",
        presentation=presentation,
        title_y=0.91,
    )

    fig.savefig(out_path)
    print(f"  Fig S8 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Future river / coastal flood risk
#   A, B = European EAD trajectories under SSP2-4.5 and SSP5-8.5
#   C, D = NUTS2 relative change in EAD, current -> 2100 SSP5-8.5
# ---------------------------------------------------------------------------

# Relative change in EAD is bounded below at -100% (a region cannot lose more
# than all of its present-day risk) but unbounded above: NUTS2 coastal ratios
# reach ~1e5% on near-zero baselines. A symmetric scale would therefore waste
# its whole lower half and render the Iberian/Greek declines as near-white, so
# the norm is centred on zero with different spans either side.
FIG4_PCT_MIN = -100.0
FIG4_PCT_MAX = 300.0


def _diverging_cmap():
    """Perceptually uniform diverging map (blue = decrease, red = increase)."""
    try:
        from cmcrameri import cm as cmc
        return cmc.vik
    except Exception:
        return plt.get_cmap("RdBu_r")


def _nuts2_ead_change(agg_dir: Path, hazards: list[str], future_period: str):
    """
    Sum EAD_mid_total_{hazard}_{period} per NUTS2 region across CI systems.

    Column-only reads (geometry from the first file only) and grouped sums.

    Returns (GeoDataFrame with NUTS_ID + geometry, dict[hazard] -> DataFrame
    indexed by NUTS_ID with 'cur', 'fut' and 'pct' columns).
    """
    import pyarrow.parquet as pq

    geom_df = None
    acc: dict[str, list] = {h: [] for h in hazards}

    files = sorted(agg_dir.glob("NUTS2_*_hazards.parquet"))
    if not files:
        print(f"  No NUTS2 files found in {agg_dir}")
        return None, {}

    for path in files:
        part = path.stem.replace("NUTS2_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue

        try:
            names = pq.ParquetFile(path).schema.names
        except Exception as e:
            print(f"    Skip {path.name}: {e}")
            continue
        if "NUTS_ID" not in names:
            continue
        if geom_df is None and "geometry" in names:
            geom_df = gpd.read_parquet(path, columns=["NUTS_ID", "geometry"])

        pairs = [(h, f"EAD_mid_total_{h}_current", f"EAD_mid_total_{h}_{future_period}")
                 for h in hazards]
        pairs = [p for p in pairs if p[1] in names and p[2] in names]
        if not pairs:
            continue
        cols = ["NUTS_ID"] + [c for _, c0, c1 in pairs for c in (c0, c1)]
        t = pd.read_parquet(path, columns=cols)
        for h, c0, c1 in pairs:
            acc[h].append(t.groupby("NUTS_ID")[[c0, c1]].sum()
                          .rename(columns={c0: "cur", c1: "fut"}))

    out = {}
    for h, frames in acc.items():
        if not frames:
            continue
        t = pd.concat(frames).groupby(level=0).sum()
        # Relative change is undefined where there is no present-day risk;
        # those regions get their own map categories rather than 0%.
        t["pct"] = np.where(t["cur"] > 0, 100.0 * (t["fut"] / t["cur"] - 1.0), np.nan)
        out[h] = t

    return geom_df, out


def fig4_flood_risk(df: pd.DataFrame, agg_dir: Path, out_path: Path,
                             presentation: bool = False):
    """
    4-panel figure on future flood risk.

    A / B: total European EAD for river (bn EUR) and coastal (M EUR) flooding
    under SSP2-4.5 and SSP5-8.5, every marker labelled, with the present-day
    -> 2100 change annotated for both scenarios.
    C / D: relative change in EAD per NUTS2 region, present day -> 2100
    SSP5-8.5, on a diverging scale. Regions with no present-day EAD are split
    into "no EAD in either period" (grey) and "EAD only by 2100" (hatched).

    River futures are global warming levels mapped to periods by convention:
    1.5 C -> 2050 SSP2-4.5, 2.0 C -> 2050 SSP5-8.5, 3.0 C -> 2100 SSP2-4.5,
    4.0 C -> 2100 SSP5-8.5 (hazard_river.TEMP_LABELS).
    """

    HAZARDS = ["river", "coastal"]
    FUTURE = "2100_SSP585"
    HAZ_TITLES = {"river": "River flooding", "coastal": "Coastal flooding"}
    # River in bn EUR, coastal in M EUR
    UNIT = {"river": (1e9, "bn €", "{:.1f}"), "coastal": (1e6, "M €", "{:.0f}")}

    ssp245_periods = ["current", "2050_SSP245", "2100_SSP245"]
    ssp585_periods = ["current", "2050_SSP585", "2100_SSP585"]

    print("  Loading NUTS2 aggregated files...")
    geom_df, change = _nuts2_ead_change(agg_dir, HAZARDS, FUTURE)
    have_maps = geom_df is not None and len(change) == len(HAZARDS)
    if not have_maps:
        print("  Could not build NUTS2 EAD change — drawing trajectory panels only")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception as e:
            print(f"  Warning: could not load NE background: {e}")

    def _draw_base(ax):
        ax.set_facecolor("#C8DCE8")
        ax.set_aspect("equal")
        if ne_gdf is not None:
            ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                        linewidth=0.3, zorder=1)
        ax.set_xlim(EUROPE_XLIM)
        ax.set_ylim(EUROPE_YLIM)
        ax.set_axis_off()

    if have_maps:
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 1.0, 0.05],
                              hspace=0.24, wspace=0.06)
    else:
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 2, wspace=0.3)

    # ---------------- Panels A / B: EAD trajectories ----------------
    traj_letters = ["A", "B"]

    def _ead_series(hazard, periods, bound="mid"):
        # 8 systems (main() already drops oil/gas); "AD" duplicates "AND".
        s = df[(df["metric_type"] == "EAD") &
               (df["metric_bound"] == bound) &
               (df["hazard"] == hazard) &
               (df["system"].isin(SYSTEM_ORDER)) &
               (df["country"] != "AD")]
        return [s[s["period"] == p]["sum"].sum()
                if len(s[s["period"] == p]) > 0 else np.nan for p in periods]

    _box = dict(boxstyle="round,pad=0.15", facecolor="white",
                edgecolor="none", alpha=0.85)

    for idx, hazard in enumerate(HAZARDS):
        ax = fig.add_subplot(gs[0, idx])
        div, unit, fmt = UNIT[hazard]

        t245 = [v / div for v in _ead_series(hazard, ssp245_periods)]
        t585 = [v / div for v in _ead_series(hazard, ssp585_periods)]

        ax.plot([0, 1, 2], t245, "o-", color=SCENARIO_COLORS["SSP245"], linewidth=2,
                label="SSP2-4.5", markersize=6)
        ax.plot([0, 1, 2], t585, "s--", color=SCENARIO_COLORS["SSP585"], linewidth=2,
                label="SSP5-8.5", markersize=6, zorder=3)

        # Value labels at every marker: SSP5-8.5 above-left, SSP2-4.5
        # below-right; at present day (shared value) the purple label sits
        # straight left of the marker.
        for xi in (0, 1, 2):
            ax.annotate(fmt.format(t585[xi]), (xi, t585[xi]), xytext=(-8, 8),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=11, color=SCENARIO_COLORS["SSP585"],
                        fontweight="bold", bbox=_box, zorder=4)
            off, va = ((-10, -2), "top") if xi == 0 else ((8, -8), "top")
            ax.annotate(fmt.format(t245[xi]), (xi, t245[xi]), xytext=off,
                        textcoords="offset points",
                        ha="right" if xi == 0 else "left", va=va,
                        fontsize=11, color=SCENARIO_COLORS["SSP245"],
                        fontweight="bold", bbox=_box, zorder=4)

        ax.set_xlim(-0.45, 2.45)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Present day", "2050", "2100"])
        top = np.nanmax(t245 + t585)
        ax.set_ylim(0, top * 1.22)
        ax.set_ylabel(f"Total EAD ({unit})")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(HAZ_TITLES[hazard], fontsize=14, fontweight="bold", pad=6)

        if not presentation:
            ax.text(-0.02, 1.02, traj_letters[idx], transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left")

        if idx == 0:
            ax.legend(framealpha=0.9, loc="upper left")

        # Relative change present day -> 2100 from the unrounded totals,
        # one per scenario, stacked bottom right (SSP5-8.5 on top).
        for y_frac, scen, traj in ((0.15, "SSP585", t585), (0.04, "SSP245", t245)):
            cur, end = traj[0], traj[2]
            if cur and cur > 0 and end is not None and not np.isnan(end):
                pct = (end - cur) / cur * 100
                ax.text(0.97, y_frac, f"{'+' if pct >= 0 else ''}{pct:.0f}%",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=12, color=SCENARIO_COLORS[scen], fontweight="bold")

    if not have_maps:
        fig.savefig(out_path)
        print(f"  Fig 4 saved: {out_path}")
        plt.close(fig)
        return

    # ---------------- Panels C / D: NUTS2 relative change ----------------
    cmap = _diverging_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=FIG4_PCT_MIN, vcenter=0.0, vmax=FIG4_PCT_MAX)
    map_letters = ["C", "D"]
    NONE_COLOR = "#E8E8E8"

    for idx, hazard in enumerate(HAZARDS):
        ax = fig.add_subplot(gs[1, idx])
        _draw_base(ax)

        t = change[hazard]
        gdf = geom_df.merge(t[["cur", "fut", "pct"]], left_on="NUTS_ID",
                            right_index=True, how="left")
        cur = gdf["cur"].fillna(0)
        fut = gdf["fut"].fillna(0)

        # No EAD in either period: neutral grey.
        none = gdf[(cur <= 0) & (fut <= 0)]
        if len(none):
            none.plot(ax=ax, color=NONE_COLOR, linewidth=0.3,
                      edgecolor="white", zorder=2)
        # No present-day EAD but EAD by 2100: relative change undefined; hatched.
        new = gdf[(cur <= 0) & (fut > 0)]
        if len(new):
            new.plot(ax=ax, facecolor="white", hatch="////", edgecolor="#777777",
                     linewidth=0.3, zorder=2)

        vals = gdf[gdf["pct"].notna()].copy()
        if len(vals):
            vals["pct_clipped"] = vals["pct"].clip(FIG4_PCT_MIN, FIG4_PCT_MAX)
            vals.plot(ax=ax, column="pct_clipped", cmap=cmap, norm=norm,
                      linewidth=0.3, edgecolor="white", zorder=2)

        noun = "river" if hazard == "river" else "coastal"
        patches = [mpatches.Patch(color=NONE_COLOR, label=f"No {noun} EAD"),
                   mpatches.Patch(facecolor="white", hatch="////", edgecolor="#777777",
                                  label="No present-day EAD,\nEAD by 2100")]
        ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                  framealpha=0.95, fontsize=10, borderpad=0.8,
                  handlelength=1.5, handleheight=1.2)

        ax.set_title(HAZ_TITLES[hazard], fontsize=14, fontweight="bold", pad=6)
        if not presentation:
            ax.text(-0.02, 1.02, map_letters[idx], transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left")

    # Shared horizontal colourbar under both maps, narrowed to the middle half
    cax = fig.add_subplot(gs[2, :])
    pos = cax.get_position()
    cax.set_position([pos.x0 + pos.width * 0.25, pos.y0,
                      pos.width * 0.5, pos.height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Only the upper end is clipped; -100% is a true floor, not a cut-off.
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("Change in EAD, present day to 2100 under SSP5-8.5 (%)", fontsize=12)
    cbar.set_ticks([FIG4_PCT_MIN, -50, 0, 100, 200, FIG4_PCT_MAX])
    cbar.ax.set_xticklabels(["-100", "-50", "0", "+100", "+200", "≥ +300"])

    fig.savefig(out_path)
    print(f"  Fig 4 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Future changes, 4 hazards — relative-change small multiples
# ---------------------------------------------------------------------------

def figS7_future_changes(df: pd.DataFrame, out_path: Path, presentation: bool = False):
    """
    2x2 small multiple: exposure change (percentage points from present) for
    heat, wildfire, river, and coastal — the four hazards where exposure
    itself (not just EAD) shifts with climate scenario. River exposure uses
    the RP-shift-inverted floodplain interpolation (see
    river_exposure_interp.py) rather than a future hazard map. Windstorm/
    earthquake/landslide have no future projection at all — none of those
    belong on an exposure-change figure.

    Percentage-point (not relative %) change: many country x system rows
    start near 0% exposed for heat/wildfire/coastal, so relative % change
    blows up arbitrarily (a 26,000%+ max was observed for heat) whenever the
    baseline is tiny — pp change is bounded and comparable across hazards.
    The line is the unweighted mean across all country x system rows; the
    shaded band is the 25th-75th percentile of that same row-level spread at
    each period. It shows cross-country/system heterogeneity, not climate-
    model uncertainty (that's the SI companion).
    """
    HAZARDS_HERE = ["heat", "wildfire", "river", "coastal"]
    LETTERS = "ABCD"
    BAND_Q = (25, 75)

    def _rows(hazard, period):
        sub = df[(df["metric_type"] == "exposure_rel") &
                 (df["metric_bound"] == "mean") &
                 (df["hazard"] == hazard) &
                 (df["period"] == period)]
        return sub["pct_exposed"].dropna()

    def _series_and_band(hazard, periods, cur):
        means, lo, hi = [], [], []
        for p in periods:
            vals = _rows(hazard, p)
            if len(vals) == 0 or np.isnan(cur):
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            means.append(vals.mean() - cur)
            lo.append(np.percentile(vals, BAND_Q[0]) - cur)
            hi.append(np.percentile(vals, BAND_Q[1]) - cur)
        return means, lo, hi

    ssp245_periods = ["current", "2050_SSP245", "2100_SSP245"]
    ssp585_periods = ["current", "2050_SSP585", "2100_SSP585"]

    data = {}
    for hazard in HAZARDS_HERE:
        cur_vals = _rows(hazard, "current")
        cur = cur_vals.mean() if len(cur_vals) else np.nan
        m245, lo245, hi245 = _series_and_band(hazard, ssp245_periods, cur)
        m585, lo585, hi585 = _series_and_band(hazard, ssp585_periods, cur)
        data[hazard] = dict(m245=m245, lo245=lo245, hi245=hi245,
                            m585=m585, lo585=lo585, hi585=hi585)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.4),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.38})
    axes = axes.flatten()

    all_vals = [v for h in HAZARDS_HERE for key in
                ("m245", "lo245", "hi245", "m585", "lo585", "hi585")
                for v in data[h][key] if not np.isnan(v)]
    y_min = min(0, min(all_vals) if all_vals else 0)
    y_max = max(all_vals) if all_vals else 100
    pad = (y_max - y_min) * 0.08
    shared_ylim = (y_min - pad, y_max + pad)

    x = [0, 1, 2]
    for idx, hazard in enumerate(HAZARDS_HERE):
        ax = axes[idx]
        letter = LETTERS[idx]
        d = data[hazard]

        ax.fill_between(x, d["lo245"], d["hi245"], color=SCENARIO_COLORS["SSP245"],
                        alpha=0.15, linewidth=0, zorder=1)
        ax.fill_between(x, d["lo585"], d["hi585"], color=SCENARIO_COLORS["SSP585"],
                        alpha=0.15, linewidth=0, zorder=1)
        ax.plot(x, d["m245"], "o-", color=SCENARIO_COLORS["SSP245"], linewidth=2,
                label="SSP2-4.5", markersize=6, zorder=3)
        ax.plot(x, d["m585"], "s--", color=SCENARIO_COLORS["SSP585"], linewidth=2,
                label="SSP5-8.5", markersize=6, zorder=3)

        ax.axhline(0, color="#CCCCCC", linewidth=0.8, zorder=2)
        ax.set_xlim(-0.3, 2.3)
        ax.set_xticks(x)
        ax.set_xticklabels(["Current", "2050", "2100"], fontsize=10)
        ax.set_ylim(*shared_ylim)
        ax.spines[["top", "right"]].set_visible(False)

        ax.text(-0.05, 1.05, letter, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="bottom", ha="left", color=HAZARD_COLORS[hazard])

        if idx % 2 == 0:
            ax.set_ylabel("Change from present (pp)", fontsize=11)
        if idx == 0:
            ax.legend(framealpha=0.9, loc="upper left", fontsize=10)

    fig.savefig(out_path)
    print(f"  Fig S7 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Subsystem drivers — EAD and uncertainty by object type
# ---------------------------------------------------------------------------

def _apply_subsystem_groups(sub_df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Map raw object_type to grouped label for a given system."""
    groups = SUBSYSTEM_GROUPS.get(system)
    if groups is None:
        return sub_df  # keep raw object_type as-is
    reverse = {ot: lbl for lbl, otypes in groups.items() for ot in otypes}
    sub_df = sub_df.copy()
    sub_df["object_type"] = sub_df["object_type"].map(reverse).fillna("Other")
    return sub_df


def _nice_ceil(val: float) -> float:
    """Round val up to one significant figure (e.g. 3700 → 4000, 12000 → 20000)."""
    if val <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(val))
    return float(np.ceil(val / mag) * mag)


def figS5_subsystem(sub_df: pd.DataFrame, out_path: Path, presentation: bool = False):
    """
    For each system: stacked bar of EAD mid (M€) by object-type group × hazard,
    plus uncertainty range markers (min/max).

    Layout: 3-3-4 rows (10 systems), with legend panel next to row 1.
    sub_df: subsystem_summary.csv loaded as DataFrame.
    """
    if sub_df is None or len(sub_df) == 0:
        print("  No subsystem summary data — run compute_subsystem_summary.py first.")
        return

    ead_mid = sub_df[(sub_df["metric_type"] == "EAD") & (sub_df["period"] == "current") & (sub_df["metric_bound"] == "mid")]
    ead_lo  = sub_df[(sub_df["metric_type"] == "EAD") & (sub_df["period"] == "current") & (sub_df["metric_bound"] == "min")]
    ead_hi  = sub_df[(sub_df["metric_type"] == "EAD") & (sub_df["period"] == "current") & (sub_df["metric_bound"] == "max")]

    # Layout: 2 rows × 4 cols + legend panel at (0,4)
    # Row 0: Power, Roads, Rail, Airports  + legend
    # Row 1: Ports, Education, Healthcare, Telecom
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.50, wspace=0.35,
                          width_ratios=[1, 1, 1, 1, 0.4])

    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),   # Power, Roads, Rail, Airports
        (1, 0), (1, 1), (1, 2), (1, 3),   # Ports, Education, Healthcare, Telecom
    ]
    letters = list("ABCDEFGH")

    sys_axes = {}
    for idx, system in enumerate(SYSTEM_ORDER):
        r, c = positions[idx]
        sys_axes[system] = fig.add_subplot(gs[r, c])

    # Legend axis (row 0, col 4)
    ax_legend = fig.add_subplot(gs[0, 4])
    ax_legend.axis("off")
    ax_legend2 = fig.add_subplot(gs[1, 4])
    ax_legend2.axis("off")

    for idx, system in enumerate(SYSTEM_ORDER):
        ax = sys_axes[system]

        mid_s = _apply_subsystem_groups(ead_mid[ead_mid["system"] == system], system)
        lo_s  = _apply_subsystem_groups(ead_lo [ead_lo ["system"] == system], system)
        hi_s  = _apply_subsystem_groups(ead_hi [ead_hi ["system"] == system], system)

        mid_grp = (mid_s.groupby(["object_type", "hazard"])["sum"].sum() / 1e6).unstack(fill_value=0)
        lo_grp  = (lo_s .groupby(["object_type", "hazard"])["sum"].sum() / 1e6).unstack(fill_value=0)
        hi_grp  = (hi_s .groupby(["object_type", "hazard"])["sum"].sum() / 1e6).unstack(fill_value=0)

        if mid_grp.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(SYSTEM_LABELS[system])
            continue

        obj_order = mid_grp.sum(axis=1).sort_values(ascending=True).index.tolist()
        mid_grp = mid_grp.reindex(obj_order)
        lo_grp  = lo_grp .reindex(obj_order, fill_value=0)
        hi_grp  = hi_grp .reindex(obj_order, fill_value=0)

        # Clean y-tick labels: remove underscores, title-case
        tick_labels = [o.replace("_", " ").title() for o in obj_order]

        y = np.arange(len(obj_order))
        lefts = np.zeros(len(obj_order))

        for h in EAD_HAZARDS:
            if h not in mid_grp.columns:
                continue
            vals = mid_grp[h].values
            ax.barh(y, vals, left=lefts, color=HAZARD_COLORS[h],
                    label=HAZARD_LABELS[h], height=0.7)
            lefts += vals

        # Uncertainty range markers
        lo_totals = lo_grp.reindex(columns=mid_grp.columns, fill_value=0).sum(axis=1).values
        hi_totals = hi_grp.reindex(columns=mid_grp.columns, fill_value=0).sum(axis=1).values
        for i in range(len(obj_order)):
            if hi_totals[i] > lo_totals[i]:
                ax.plot([lo_totals[i], hi_totals[i]], [i, i],
                        "-", color="#111111", lw=1.5, zorder=5)
                for xv in [lo_totals[i], hi_totals[i]]:
                    ax.plot([xv, xv], [i - 0.28, i + 0.28],
                            "-", color="#111111", lw=1.5, zorder=5)

        # Nice round x-axis limit
        xmax = max(hi_totals.max(), lefts.max()) if len(hi_totals) > 0 else 1.0
        ax.set_xlim(0, _nice_ceil(xmax))

        ax.set_yticks(y)
        ax.set_yticklabels(tick_labels, fontsize=10)
        ax.set_xlabel("EAD (M€)", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)

        title_txt = SYSTEM_LABELS[system]
        if presentation:
            ax.set_title(f"{letters[idx]}.  {title_txt}", pad=4, fontsize=14,
                         fontweight="bold", loc="center")
        else:
            ax.text(-0.02, 1.02, letters[idx], transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left")
            ax.set_title(title_txt, pad=4, fontsize=12, loc="center")

    # Legend in top-right panel
    hazard_patches = [mpatches.Patch(color=HAZARD_COLORS[h], label=HAZARD_LABELS[h])
                      for h in EAD_HAZARDS]
    ax_legend.legend(handles=hazard_patches, loc="center left",
                     fontsize=12, framealpha=0.9,
                     borderpad=1.2, handlelength=1.8, handleheight=1.4)

    fig.savefig(out_path)
    print(f"  Fig S5 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Uncertainty heatmap — max/mid ratio by system × hazard
# ---------------------------------------------------------------------------

def figS4_uncertainty_heatmap(sub_df: pd.DataFrame, out_path: Path, presentation: bool = False):
    """
    Two-panel uncertainty figure:
      A. Heatmap of max/mid EAD ratio by system × hazard (how uncertain?)
      B. Heatmap of absolute uncertainty gap (M€, log10) by system × hazard (how large?)
    Rows = systems, cols = EAD hazards.
    """
    if sub_df is None or len(sub_df) == 0:
        print("  No subsystem data for fig6.")
        return

    cur = sub_df[(sub_df["metric_type"] == "EAD") & (sub_df["period"] == "current")]

    # Aggregate across all object_types and countries → system × hazard totals
    mid_tot = (cur[cur["metric_bound"] == "mid"]
               .groupby(["system", "hazard"])["sum"].sum() / 1e6)
    max_tot = (cur[cur["metric_bound"] == "max"]
               .groupby(["system", "hazard"])["sum"].sum() / 1e6)

    ratio_mat = pd.DataFrame(index=SYSTEM_ORDER, columns=EAD_HAZARDS, dtype=float)
    gap_mat   = pd.DataFrame(index=SYSTEM_ORDER, columns=EAD_HAZARDS, dtype=float)

    for sys in SYSTEM_ORDER:
        for haz in EAD_HAZARDS:
            mid_v = mid_tot.get((sys, haz), 0)
            max_v = max_tot.get((sys, haz), 0)
            ratio_mat.loc[sys, haz] = max_v / mid_v if mid_v > 0.001 else np.nan
            gap_mat.loc[sys, haz]   = max_v - mid_v if mid_v > 0 else np.nan

    # Top country driver per system × hazard (for annotation on panel A)
    cur_country = sub_df[(sub_df["metric_type"] == "EAD") & (sub_df["period"] == "current")]
    mid_c = cur_country[cur_country["metric_bound"] == "mid"].groupby(
        ["system", "hazard", "country"])["sum"].sum()
    max_c = cur_country[cur_country["metric_bound"] == "max"].groupby(
        ["system", "hazard", "country"])["sum"].sum()
    gap_c = ((max_c - mid_c) / 1e6).rename("gap").reset_index()
    top_country = (gap_c.sort_values("gap", ascending=False)
                        .groupby(["system", "hazard"])
                        .first()
                        .reset_index()[["system", "hazard", "country"]])
    top_country_map = {(r.system, r.hazard): r.country
                       for r in top_country.itertuples()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={"wspace": 0.35})

    col_labels = [HAZARD_LABELS[h] for h in EAD_HAZARDS]
    row_labels  = [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]

    # --- Panel A: ratio heatmap ---
    ax = axes[0]
    R = ratio_mat.values.astype(float)
    im = ax.imshow(R, cmap="YlOrRd", aspect="auto", vmin=1, vmax=12)
    ax.set_xticks(range(len(EAD_HAZARDS)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(SYSTEM_ORDER)))
    ax.set_yticklabels(row_labels)

    for i, sys in enumerate(SYSTEM_ORDER):
        for j, haz in enumerate(EAD_HAZARDS):
            v = R[i, j]
            if np.isnan(v):
                continue
            ctry = top_country_map.get((sys, haz), "")
            txt_color = "white" if v > 7 else "black"
            ax.text(j, i - 0.15, f"{v:.1f}×",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=txt_color)
            ax.text(j, i + 0.25, ctry,
                    ha="center", va="center", fontsize=7,
                    color=txt_color, style="italic")

    plt.colorbar(im, ax=ax, label="max / mid ratio", shrink=0.85, pad=0.02)

    if presentation:
        ax.set_title("A.  Uncertainty ratio (max / mid EAD)",
                     pad=6, fontsize=14, fontweight="bold")
    else:
        ax.text(-0.02, 1.02, "A", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
        ax.set_title("Uncertainty ratio (max / mid)", pad=4, fontsize=12)

    # --- Panel B: absolute gap heatmap (log10 M€) ---
    ax = axes[1]
    G = gap_mat.values.astype(float)
    G_log = np.log10(np.where(G > 0.1, G, np.nan))
    vmax_log = np.nanmax(G_log)
    im2 = ax.imshow(G_log, cmap="PuBu", aspect="auto", vmin=0, vmax=vmax_log)
    ax.set_xticks(range(len(EAD_HAZARDS)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(SYSTEM_ORDER)))
    ax.set_yticklabels(row_labels)

    for i, sys in enumerate(SYSTEM_ORDER):
        for j, haz in enumerate(EAD_HAZARDS):
            v = G[i, j]
            if np.isnan(v) or v < 0.1:
                continue
            txt_color = "white" if G_log[i, j] > 0.7 * vmax_log else "black"
            ax.text(j, i, f"{v:,.0f}",
                    ha="center", va="center", fontsize=8.5, color=txt_color)

    cbar2 = plt.colorbar(im2, ax=ax, label="Gap (M€)", shrink=0.85, pad=0.02)
    tick_vals = [t for t in cbar2.get_ticks() if 0 <= t <= vmax_log]
    cbar2.set_ticks(tick_vals)
    cbar2.set_ticklabels([f"{10**t:,.0f}" for t in tick_vals])

    if presentation:
        ax.set_title("B.  Absolute uncertainty gap (M€, max − mid EAD)",
                     pad=6, fontsize=14, fontweight="bold")
    else:
        ax.text(-0.02, 1.02, "B", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
        ax.set_title("Absolute gap (M€, max − mid)", pad=4, fontsize=12)

    fig.savefig(out_path)
    print(f"  Fig S4 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplement figure S1: 2nd and 3rd dominant CI system per LAU
# ---------------------------------------------------------------------------

def figS9_dominant_ci_rank(agg_dir: Path, out_path: Path, presentation: bool = False):
    """
    Two-panel LAU map showing 2nd and 3rd most dominant CI system by EAD.
    Uses same LAU files as fig2 panel B.
    """
    system_colors = SYSTEM_COLORS

    lau_system: dict = {}
    geom_df = None

    lau_files = sorted(agg_dir.glob("LAU_*_hazards.parquet"))
    if not lau_files:
        print(f"  No LAU files found in {agg_dir}")
        return

    for path in lau_files:
        part = path.stem.replace("LAU_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue
        system = part if part in SYSTEM_ORDER else None

        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            try:
                gdf = pd.read_parquet(path)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
                continue

        id_col = next((c for c in ["LAU_ID", "LAU_CODE", "GEO_ID", "id", "GISCO_ID"]
                       if c in gdf.columns), None)
        if id_col is None:
            id_col = gdf.columns[0]

        if geom_df is None and "geometry" in gdf.columns:
            geom_df = gdf[[id_col, "geometry"]].copy().rename(columns={id_col: "LAU_ID"})

        if system:
            for _, row in gdf.iterrows():
                lau_id = row[id_col]
                sys_total = 0.0
                for h in EAD_HAZARDS:
                    col = f"EAD_mid_total_{h}_current"
                    if col not in gdf.columns:
                        col = next((c for c in gdf.columns
                                    if "EAD_mid" in c and h in c and "current" in c), None)
                    if col:
                        sys_total += float(row[col]) if not pd.isna(row[col]) else 0.0
                if lau_id not in lau_system:
                    lau_system[lau_id] = {ss: 0.0 for ss in SYSTEM_ORDER}
                lau_system[lau_id][system] = lau_system[lau_id].get(system, 0.0) + sys_total

    if not lau_system or geom_df is None:
        print("  Could not build LAU system lookup")
        return

    sys_df = pd.DataFrame.from_dict(lau_system, orient="index").fillna(0)
    sys_df.index.name = "LAU_ID"
    present = [s for s in SYSTEM_ORDER if s in sys_df.columns]

    def _nth_dominant(row, n):
        sorted_sys = row[present].sort_values(ascending=False)
        total = sorted_sys.sum()
        if total == 0 or len(sorted_sys) < n:
            return "none"
        return sorted_sys.index[n - 1]

    sys_df["rank2"] = sys_df.apply(_nth_dominant, n=2, axis=1)
    sys_df["rank3"] = sys_df.apply(_nth_dominant, n=3, axis=1)
    sys_df = sys_df.reset_index()

    map_gdf = (geom_df
               .merge(sys_df[["LAU_ID", "rank2", "rank3"]], on="LAU_ID", how="left"))
    map_gdf["rank2"] = map_gdf["rank2"].fillna("none")
    map_gdf["rank3"] = map_gdf["rank3"].fillna("none")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception:
            pass

    def _draw_base(ax):
        ax.set_facecolor("#C8DCE8")
        ax.set_aspect("equal")
        if ne_gdf is not None:
            ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                        linewidth=0.3, zorder=1)
        ax.set_xlim(EUROPE_XLIM)
        ax.set_ylim(EUROPE_YLIM)
        ax.set_axis_off()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 10),
                                      gridspec_kw={"wspace": 0.04})

    sys_cmap = {s: system_colors[s] for s in SYSTEM_ORDER}
    sys_cmap["none"] = "#E8E8E8"

    for ax, rank_col, label, letter in [
        (ax_a, "rank2", "2nd dominant CI system", "A"),
        (ax_b, "rank3", "3rd dominant CI system", "B"),
    ]:
        _draw_base(ax)
        for val, color in sys_cmap.items():
            sub = map_gdf[map_gdf[rank_col] == val]
            if len(sub):
                sub.plot(ax=ax, color=color, linewidth=0, edgecolor="none", zorder=2)

        patches = [mpatches.Patch(color=system_colors[s], label=SYSTEM_LABELS[s])
                   for s in SYSTEM_ORDER]
        patches.append(mpatches.Patch(color="#E8E8E8", label="No EAD"))
        ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                  framealpha=0.95, fontsize=9, borderpad=0.8,
                  handlelength=1.4, handleheight=1.2)

        if presentation:
            ax.set_title(f"{letter}.  {label} (EAD, current)",
                         pad=6, fontsize=14, fontweight="bold", loc="center")
        else:
            ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left")

    fig.savefig(out_path)
    print(f"  Fig S9 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplement figure S2: dominant uncertainty source per NUTS2
# ---------------------------------------------------------------------------

def figS3_uncertainty_map(agg_dir: Path, out_path: Path, presentation: bool = False):
    """
    Two-panel NUTS2 uncertainty choropleth.
    Panel A: dominant hazard by absolute EAD gap (max-mid).
    Panel B: overall relative uncertainty ratio sum(EAD_max)/sum(EAD_mid).
    """
    print("  Loading NUTS2 aggregated files...")

    nuts2_gap: dict = {}        # {nuts2_id: {hazard: gap}}
    nuts2_mid_sum: dict = {}    # {nuts2_id: float}
    nuts2_max_sum: dict = {}    # {nuts2_id: float}
    geom_df = None

    nuts2_files = sorted(agg_dir.glob("NUTS2_*_hazards.parquet"))
    if not nuts2_files:
        print(f"  No NUTS2 files found in {agg_dir}")
        return

    for path in nuts2_files:
        part = path.stem.replace("NUTS2_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue

        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            try:
                gdf = pd.read_parquet(path)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
                continue

        id_col = next((c for c in ["NUTS_ID", "NUTS2", "NUTS2_ID"]
                       if c in gdf.columns), None)
        if id_col is None:
            continue

        if geom_df is None and "geometry" in gdf.columns:
            geom_df = gdf[[id_col, "geometry"]].copy().rename(columns={id_col: "NUTS2_ID"})

        for _, row in gdf.iterrows():
            nuts2_id = row[id_col]
            for h in EAD_HAZARDS:
                mid_col = f"EAD_mid_total_{h}_current"
                max_col = f"EAD_max_total_{h}_current"
                if mid_col not in gdf.columns or max_col not in gdf.columns:
                    mid_col = next((c for c in gdf.columns
                                    if "EAD_mid" in c and h in c and "current" in c), None)
                    max_col = next((c for c in gdf.columns
                                    if "EAD_max" in c and h in c and "current" in c), None)
                if not mid_col or not max_col:
                    continue
                mid_v = float(row[mid_col]) if not pd.isna(row[mid_col]) else 0.0
                max_v = float(row[max_col]) if not pd.isna(row[max_col]) else 0.0
                gap = max(0.0, max_v - mid_v)
                if nuts2_id not in nuts2_gap:
                    nuts2_gap[nuts2_id] = {hh: 0.0 for hh in EAD_HAZARDS}
                nuts2_gap[nuts2_id][h] = nuts2_gap[nuts2_id].get(h, 0.0) + gap
                nuts2_mid_sum[nuts2_id] = nuts2_mid_sum.get(nuts2_id, 0.0) + mid_v
                nuts2_max_sum[nuts2_id] = nuts2_max_sum.get(nuts2_id, 0.0) + max_v

    if not nuts2_gap or geom_df is None:
        print("  Could not build NUTS2 uncertainty lookup")
        return

    gap_df = pd.DataFrame.from_dict(nuts2_gap, orient="index").fillna(0)
    gap_df.index.name = "NUTS2_ID"
    gap_df["total_gap"] = gap_df[EAD_HAZARDS].sum(axis=1)
    gap_df["dominant_uncertainty"] = gap_df[EAD_HAZARDS].idxmax(axis=1)
    gap_df.loc[gap_df["total_gap"] == 0, "dominant_uncertainty"] = "none"
    gap_df = gap_df.reset_index()

    ratio_df = pd.DataFrame({
        "NUTS2_ID": list(nuts2_mid_sum.keys()),
        "mid_sum":  list(nuts2_mid_sum.values()),
        "max_sum":  list(nuts2_max_sum.values()),
    })
    ratio_df["ratio"] = ratio_df["max_sum"] / ratio_df["mid_sum"].clip(lower=1e-9)
    ratio_df.loc[ratio_df["mid_sum"] < 1.0, "ratio"] = float("nan")

    map_gdf = geom_df.merge(
        gap_df[["NUTS2_ID", "dominant_uncertainty", "total_gap"]],
        on="NUTS2_ID", how="left"
    )
    map_gdf["dominant_uncertainty"] = map_gdf["dominant_uncertainty"].fillna("none")
    map_gdf = map_gdf.merge(ratio_df[["NUTS2_ID", "ratio"]], on="NUTS2_ID", how="left")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception:
            pass

    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    plt.subplots_adjust(wspace=0.04)

    # ── Panel A: dominant hazard by absolute gap ──────────────────────────
    ax = axes[0]
    ax.set_facecolor("#C8DCE8")
    ax.set_aspect("equal")
    if ne_gdf is not None:
        ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                    linewidth=0.3, zorder=1)

    haz_cmap = {h: HAZARD_COLORS[h] for h in EAD_HAZARDS}
    haz_cmap["none"] = "#E8E8E8"
    for val, color in haz_cmap.items():
        sub = map_gdf[map_gdf["dominant_uncertainty"] == val]
        if len(sub):
            sub.plot(ax=ax, color=color, linewidth=0.3, edgecolor="white", zorder=2)

    ax.set_xlim(EUROPE_XLIM)
    ax.set_ylim(EUROPE_YLIM)

    patches = [mpatches.Patch(color=HAZARD_COLORS[h], label=HAZARD_LABELS[h])
               for h in EAD_HAZARDS]
    patches.append(mpatches.Patch(color="#E8E8E8", label="No data"))
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
              framealpha=0.95, fontsize=10, borderpad=1.0,
              handlelength=1.5, handleheight=1.2)

    if presentation:
        ax.set_title("Dominant uncertainty source\n(largest EAD gap, current)",
                     pad=6, fontsize=13, fontweight="bold", loc="center")
    else:
        ax.text(-0.02, 1.02, "A", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax.set_axis_off()

    # ── Panel B: relative uncertainty ratio (max/mid) ─────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#C8DCE8")
    ax2.set_aspect("equal")
    if ne_gdf is not None:
        ne_gdf.plot(ax=ax2, color="#D8D8D8", edgecolor="#AAAAAA",
                    linewidth=0.3, zorder=1)

    valid_ratios = map_gdf["ratio"].dropna()
    vmin = 1.0
    vmax = float(np.percentile(valid_ratios, 98)) if len(valid_ratios) else 5.0
    vmax = max(vmax, 1.5)

    cmap_ratio = plt.colormaps["YlOrRd"]
    norm_ratio = mcolors.Normalize(vmin=vmin, vmax=vmax)

    no_data = map_gdf[map_gdf["ratio"].isna()]
    if len(no_data):
        no_data.plot(ax=ax2, color="#E8E8E8", linewidth=0.3, edgecolor="white", zorder=2)

    has_data = map_gdf[map_gdf["ratio"].notna()].copy()
    if len(has_data):
        has_data.plot(ax=ax2, column="ratio", cmap=cmap_ratio, norm=norm_ratio,
                      linewidth=0.3, edgecolor="white", zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap_ratio, norm=norm_ratio)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.65)
    cbar.set_label("EAD max / mid ratio (×)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax2.set_xlim(EUROPE_XLIM)
    ax2.set_ylim(EUROPE_YLIM)

    if presentation:
        ax2.set_title("Relative uncertainty\n(EAD max/mid ratio, current)",
                      pad=6, fontsize=13, fontweight="bold", loc="center")
    else:
        ax2.text(-0.02, 1.02, "B", transform=ax2.transAxes,
                 fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax2.set_axis_off()

    fig.savefig(out_path)
    print(f"  Fig S3 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: heat + wildfire exposure — trajectory panels + LAU maps
# ---------------------------------------------------------------------------

def fig3_heat_wildfire(df: pd.DataFrame, agg_dir: Path, out_path: Path,
                  presentation: bool = False):
    """
    Combined heat/wildfire exposure figure:
      A. Extreme heat exposure trajectory (SSP2-4.5 vs SSP5-8.5)
      B. Wildfire exposure trajectory
      C/D/E. Heat exposure maps — current | 2050 SSP5-8.5 | 2100 SSP5-8.5
      F/G/H. Wildfire exposure maps — current | 2050 SSP5-8.5 | 2100 SSP5-8.5
    """
    print("  Loading data for Fig 3...")

    HAZARDS = ["heat", "wildfire"]
    # Output column keys keep the pipeline's "2100" label, but for heat and
    # wildfire that column is the 2061-2080 window (far_future in
    # WINDOW_TO_PERIOD), so it is displayed as 2070 throughout.
    PERIODS = ["current", "2050_SSP585", "2100_SSP585"]
    _PERIOD_NAMES = {
        "current":      "Baseline",
        "2050_SSP585":  "2050 SSP5-8.5",
        "2100_SSP585":  "2070 SSP5-8.5",
    }
    TRAJ_TICKLABELS = ["Baseline (1990 to 2016)", "2050 (2041 to 2060)", "2070 (2061 to 2080)"]
    MAP_LETTERS = "CDEFGH"

    PANEL_LABELS = {}
    for r, h in enumerate(HAZARDS):
        for c, p in enumerate(PERIODS):
            idx = r * len(PERIODS) + c
            pname = _PERIOD_NAMES.get(p, p)
            hlabel = "Extreme heat" if h == "heat" else "Wildfire"
            PANEL_LABELS[(h, p)] = (MAP_LETTERS[idx], f"{hlabel} — {pname}")

    # --- Map data: accumulate per LAU (abs exposure, asset size) ---
    # Vectorised: read only the needed columns (geometry from the first file
    # only) and accumulate per-LAU sums with groupby instead of iterrows.
    import pyarrow.parquet as pq

    keys = [(h, p) for h in HAZARDS for p in PERIODS]
    abs_acc = {k: pd.Series(dtype=float) for k in keys}
    size_acc = {k: pd.Series(dtype=float) for k in keys}
    seen_ids = pd.Index([])
    geom_df = None

    lau_files = sorted(agg_dir.glob("LAU_*_hazards.parquet"))
    if not lau_files:
        print(f"  No LAU files found in {agg_dir}")
        return

    for path in lau_files:
        part = path.stem.replace("LAU_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue

        try:
            names = pq.ParquetFile(path).schema.names
        except Exception as e:
            print(f"    Skip {path.name}: {e}")
            continue

        id_col = next((c for c in ["LAU_ID", "LAU_CODE", "GEO_ID", "id", "GISCO_ID"]
                       if c in names), None)
        if id_col is None:
            continue

        size_col = "asset_size_total" if "asset_size_total" in names else None
        col_for = {}
        for h, p in keys:
            col = f"exposure_abs_total_{h}_{p}"
            if col not in names:
                col = next((c for c in names
                            if "exposure_abs" in c and h in c and p in c), None)
            if col is not None:
                col_for[(h, p)] = col

        read_cols = list(dict.fromkeys(
            [id_col] + ([size_col] if size_col else []) + list(col_for.values())))
        if geom_df is None and "geometry" in names:
            g = gpd.read_parquet(path, columns=[id_col, "geometry"])
            geom_df = g.rename(columns={id_col: "LAU_ID"})
        t = pd.read_parquet(path, columns=read_cols)

        ids = t[id_col]
        seen_ids = seen_ids.union(pd.Index(ids.unique()))
        size = (t[size_col].fillna(0.0) if size_col else pd.Series(0.0, index=t.index))
        size_by_lau = size.groupby(ids).sum()
        for k, col in col_for.items():
            abs_acc[k] = abs_acc[k].add(t[col].fillna(0.0).groupby(ids).sum(), fill_value=0.0)
            size_acc[k] = size_acc[k].add(size_by_lau, fill_value=0.0)

    if len(seen_ids) == 0 or geom_df is None:
        print("  Could not build exposure lookup — check LAU files")
        return

    exp_df = pd.DataFrame({"LAU_ID": seen_ids})
    for h, p in keys:
        a = abs_acc[(h, p)].reindex(seen_ids, fill_value=0.0).to_numpy()
        s = size_acc[(h, p)].reindex(seen_ids, fill_value=0.0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            exp_df[f"{h}_{p}"] = np.where(s > 0, a / s * 100, np.nan)

    map_gdf = geom_df.merge(exp_df, on="LAU_ID", how="left")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception as e:
            print(f"  Warning: could not load NE background: {e}")

    CMAPS = {"heat": "YlOrRd", "wildfire": "YlOrBr"}

    HAZ_ROW_LABELS = {"heat": "Extreme heat", "wildfire": "Wildfire"}

    HAZ_TRAJ_LETTER = {"heat": "A", "wildfire": "B"}
    HAZ_TRAJ_SUBTITLE = {"heat": "Extreme heat exposure trajectory",
                         "wildfire": "Wildfire exposure trajectory"}

    # --- Layout: for each hazard, trajectory row directly above its map row.
    # Columns are shared across ALL rows (label | colorbar | 3 map cols) so
    # the trajectory's x-axis lines up with the map columns below it: with
    # xlim=(-0.5, 2.5) over 3 equally-spaced points, each point sits exactly
    # at the horizontal centre of its corresponding map column. Map columns
    # are baseline / 2050 SSP5-8.5 / 2070 SSP5-8.5, matching the trajectory's
    # Baseline/2050/2070 x-axis (SSP5-8.5 line specifically).
    fig = plt.figure(figsize=(20, 19))
    gs = fig.add_gridspec(4, len(PERIODS) + 1,
                          height_ratios=[0.42, 1.0, 0.42, 1.0],
                          width_ratios=[0.09] + [1] * len(PERIODS),
                          hspace=0.14, wspace=0.08)

    ssp245_periods = ["current", "2050_SSP245", "2100_SSP245"]
    ssp585_periods = ["current", "2050_SSP585", "2100_SSP585"]

    def _get_pct(hazard, periods):
        # FigA convention (as Fig 1A): unweighted mean of country-level
        # pct_exposed per system, then unweighted mean across the 8 systems.
        # Oil/gas are excluded; "AD" duplicates "AND" (Andorra) and is dropped.
        sub = df[(df["metric_type"] == "exposure_rel") &
                 (df["metric_bound"] == "mean") &
                 (df["hazard"] == hazard) &
                 (df["system"].isin(SYSTEM_ORDER)) &
                 (df["country"] != "AD")]
        out = []
        for p in periods:
            s = sub[sub["period"] == p]
            out.append(s.groupby("system")["pct_exposed"].mean().mean()
                       if len(s) > 0 else np.nan)
        return out

    for hi, h in enumerate(HAZARDS):
        traj_row = hi * 2
        maps_row = hi * 2 + 1

        # --- Trajectory panel (spans only the 3 map columns) ---
        ax_t = fig.add_subplot(gs[traj_row, 1:])
        t245 = _get_pct(h, ssp245_periods)
        t585 = _get_pct(h, ssp585_periods)

        ax_t.plot([0, 1, 2], t245, "o-", color=SCENARIO_COLORS["SSP245"], linewidth=2,
                  label="SSP2-4.5", markersize=6)
        ax_t.plot([0, 1, 2], t585, "s--", color=SCENARIO_COLORS["SSP585"], linewidth=2.5,
                  label="SSP5-8.5 (shown in maps)", markersize=7, zorder=3)

        # Value labels at every marker: SSP5-8.5 above-left of its markers,
        # SSP2-4.5 below-right, so neither crosses the rising lines.
        _box = dict(boxstyle="round,pad=0.15", facecolor="white",
                    edgecolor="none", alpha=0.85)
        for xi in (0, 1, 2):
            ax_t.annotate(f"{t585[xi]:.0f}%", (xi, t585[xi]), xytext=(-8, 9),
                          textcoords="offset points", ha="right", va="bottom",
                          fontsize=13, color=SCENARIO_COLORS["SSP585"],
                          fontweight="bold", bbox=_box, zorder=4)
            # Baseline values sit near the x axis, so the purple label goes
            # straight left of the marker there instead of below it.
            off, va = ((-10, 0), "center") if xi == 0 else ((8, -10), "top")
            ax_t.annotate(f"{t245[xi]:.0f}%", (xi, t245[xi]), xytext=off,
                          textcoords="offset points",
                          ha="right" if xi == 0 else "left", va=va,
                          fontsize=13, color=SCENARIO_COLORS["SSP245"],
                          fontweight="bold", bbox=_box, zorder=4)

        ax_t.set_xlim(-0.5, len(PERIODS) - 1 + 0.5)
        ax_t.set_xticks([0, 1, 2])
        # Tick labels only on the top (heat) panel; wildfire shares the axis.
        ax_t.set_xticklabels(TRAJ_TICKLABELS if h == "heat" else [], fontsize=14)
        ax_t.set_ylim(0, 55)
        ax_t.set_yticks(range(0, 51, 10))
        ax_t.set_ylabel("Mean exposure\n(% assets)", fontsize=14)
        ax_t.tick_params(axis="y", labelsize=13)
        ax_t.spines[["top", "right"]].set_visible(False)

        letter = HAZ_TRAJ_LETTER[h]
        if presentation:
            ax_t.set_title(f"{letter}.  {HAZ_TRAJ_SUBTITLE[h]}", pad=4,
                           fontsize=17, fontweight="bold", loc="center")
        else:
            ax_t.text(-0.09, 0.94, letter, transform=ax_t.transAxes, fontsize=19,
                      fontweight="bold", va="top", ha="left")
            # Axes-fraction y (not pad= in points, which barely moves on a
            # figure this large) — lower value = lower on the page.
            title_y = 1.06 if h == "heat" else 0.97
            ax_t.text(0.5, title_y, HAZ_ROW_LABELS[h], transform=ax_t.transAxes,
                      ha="center", va="bottom", fontsize=18, fontweight="bold")
        if h == "heat":
            ax_t.legend(framealpha=0.9, loc="upper left", fontsize=12)

        # Relative change baseline -> 2070 (2061-2080) from the unrounded
        # means, one per scenario, stacked bottom right (SSP5-8.5 on top).
        for y_frac, scen, traj in ((0.17, "SSP585", t585), (0.05, "SSP245", t245)):
            cur, end = traj[0], traj[2]
            if cur is not None and cur > 0 and end is not None and not np.isnan(end):
                pct = (end - cur) / cur * 100
                sign = "+" if pct >= 0 else ""
                ax_t.text(0.97, y_frac, f"{sign}{pct:.0f}%", transform=ax_t.transAxes,
                          ha="right", va="bottom", fontsize=14,
                          color=SCENARIO_COLORS[scen], fontweight="bold")

        # --- Map row for this hazard ---
        # Shared colour scale across the whole row (one legend per row)
        row_cols = [f"{h}_{p}" for p in PERIODS]
        row_valid = map_gdf[row_cols].values.flatten()
        row_valid = row_valid[~pd.isna(row_valid)]
        vmax = float(np.nanpercentile(row_valid, 98)) if len(row_valid) else 1.0
        vmax = max(vmax, 0.1)

        # Shared colorbar for the row (its own dedicated axes — col 0).
        # Ticks + label on the left side of the bar (away from the maps).
        ax_cbar = fig.add_subplot(gs[maps_row, 0])
        sm = plt.cm.ScalarMappable(cmap=CMAPS[h], norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label("% CI assets exposed", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        for col_idx, period in enumerate(PERIODS):
            ax = fig.add_subplot(gs[maps_row, col_idx + 1])
            col = f"{h}_{period}"
            letter, subtitle = PANEL_LABELS[(h, period)]

            ax.set_facecolor("#C8DCE8")
            ax.set_aspect("equal")
            if ne_gdf is not None:
                ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                            linewidth=0.3, zorder=1)

            valid = map_gdf[map_gdf[col].notna()].copy()
            if len(valid):
                valid.plot(ax=ax, column=col, cmap=CMAPS[h],
                           vmin=0, vmax=vmax, linewidth=0, edgecolor="none",
                           zorder=2, legend=False)

            ax.set_xlim(EUROPE_XLIM)
            ax.set_ylim(EUROPE_YLIM)
            ax.set_axis_off()

            if presentation:
                ax.set_title(f"{letter}.  {subtitle}",
                             pad=6, fontsize=17, fontweight="bold", loc="center")
            else:
                ax.text(0.02, 0.95, letter, transform=ax.transAxes,
                        fontsize=17, fontweight="bold", va="top", ha="left")

    # Thin divider between the Heat block (rows 0-1) and Wildfires block
    # (rows 2-3) — they're stacked with no other visual separation.
    from matplotlib.lines import Line2D
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    divider_y = (bottoms[1] + tops[2]) / 2
    fig.add_artist(Line2D([lefts[0], rights[-1]], [divider_y, divider_y],
                          transform=fig.transFigure, color="#BBBBBB",
                          linewidth=1.0))

    fig.savefig(out_path)
    print(f"  Fig 3 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure S2: Multi-hazard exposure map — count-based, 3 thresholds
# ---------------------------------------------------------------------------

def figS2_multihazard_map(agg_dir: Path, out_path: Path, presentation: bool = False):
    """
    3-panel LAU choropleth: number of hazards to which ≥X% of CI assets
    (count-based) are exposed, at X = 25 / 50 / 75.

    Metric: n_exposed_total_{h}_current / n_features_total per LAU,
    summed across all system files so each asset counts once.
    """
    print("  Loading LAU files for multi-hazard map...")

    THRESHOLDS = [25, 50, 75]
    N_HAZ = len(ALL_HAZARDS)

    # Accumulate per LAU: {lau_id: {hazard: {"exposed": int, "total": int}}}
    lau_counts: dict = {}
    geom_df = None

    lau_files = sorted(agg_dir.glob("LAU_*_hazards.parquet"))
    if not lau_files:
        print(f"  No LAU files found in {agg_dir}")
        return

    for path in lau_files:
        part = path.stem.replace("LAU_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue

        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            try:
                gdf = pd.read_parquet(path)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
                continue

        id_col = next((c for c in ["LAU_ID", "LAU_CODE", "GEO_ID", "id", "GISCO_ID"]
                       if c in gdf.columns), None)
        if id_col is None:
            continue

        if geom_df is None and "geometry" in gdf.columns:
            geom_df = gdf[[id_col, "geometry"]].copy().rename(columns={id_col: "LAU_ID"})

        n_total_col = "n_features_total" if "n_features_total" in gdf.columns else None

        for _, row in gdf.iterrows():
            lau_id = row[id_col]
            if lau_id not in lau_counts:
                lau_counts[lau_id] = {h: {"exposed": 0, "total": 0} for h in ALL_HAZARDS}

            n_total = float(row[n_total_col]) if n_total_col and not pd.isna(row[n_total_col]) else 0.0
            for h in ALL_HAZARDS:
                col = f"n_exposed_total_{h}_current"
                if col not in gdf.columns:
                    continue
                n_exp = float(row[col]) if not pd.isna(row[col]) else 0.0
                lau_counts[lau_id][h]["exposed"] += n_exp
                lau_counts[lau_id][h]["total"]   += n_total

    if not lau_counts or geom_df is None:
        print("  Could not build multi-hazard lookup — check n_exposed columns in LAU files")
        return

    # Build per-LAU pct_exposed per hazard
    records = []
    for lau_id, haz_dict in lau_counts.items():
        rec = {"LAU_ID": lau_id}
        for h, vals in haz_dict.items():
            rec[f"pct_{h}"] = (
                vals["exposed"] / vals["total"] * 100
                if vals["total"] > 0 else np.nan
            )
        records.append(rec)

    pct_df = pd.DataFrame(records)

    # For each threshold: count hazards ≥ threshold
    for t in THRESHOLDS:
        cols = [f"pct_{h}" for h in ALL_HAZARDS]
        pct_df[f"n_haz_{t}"] = (pct_df[cols] >= t).sum(axis=1).where(
            pct_df[cols].notna().any(axis=1), other=np.nan
        )

    map_gdf = geom_df.merge(pct_df, on="LAU_ID", how="left")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception:
            pass

    # Discrete colorscale: 0 = grey, 1-7 = sequential blue-green-red
    import matplotlib.colors as mcolors

    base_colors = ["#E8E8E8"] + [
        plt.colormaps["YlOrRd"](i / (N_HAZ - 1)) for i in range(N_HAZ)
    ]
    cmap_disc = mcolors.ListedColormap(base_colors)
    bounds_disc = np.arange(-0.5, N_HAZ + 1, 1)
    norm_disc = mcolors.BoundaryNorm(bounds_disc, cmap_disc.N)

    letters = ["A", "B", "C"]
    fig, axes = plt.subplots(1, 3, figsize=(27, 10),
                             gridspec_kw={"wspace": 0.04})

    for idx, t in enumerate(THRESHOLDS):
        ax = axes[idx]
        col = f"n_haz_{t}"

        ax.set_facecolor("#C8DCE8")
        ax.set_aspect("equal")
        if ne_gdf is not None:
            ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                        linewidth=0.3, zorder=1)

        no_data = map_gdf[map_gdf[col].isna()]
        if len(no_data):
            no_data.plot(ax=ax, color="#C8C8C8", linewidth=0, edgecolor="none", zorder=2)

        has_data = map_gdf[map_gdf[col].notna()].copy()
        if len(has_data):
            has_data.plot(ax=ax, column=col, cmap=cmap_disc, norm=norm_disc,
                          linewidth=0, edgecolor="none", zorder=3)

        ax.set_xlim(EUROPE_XLIM)
        ax.set_ylim(EUROPE_YLIM)
        ax.set_axis_off()

        subtitle = f"≥{t}% of assets exposed"
        if presentation:
            ax.set_title(f"{letters[idx]}.  {subtitle}",
                         pad=6, fontsize=14, fontweight="bold", loc="center")
        else:
            ax.text(-0.02, 1.02, letters[idx], transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left")
            ax.set_title(subtitle, pad=4, fontsize=12, loc="center")

    # Shared colorbar on right
    sm = plt.cm.ScalarMappable(cmap=cmap_disc, norm=norm_disc)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.01, shrink=0.7,
                        ticks=range(N_HAZ + 1))
    cbar.set_label("Number of hazards", fontsize=11)
    cbar.set_ticklabels([str(i) for i in range(N_HAZ + 1)])

    fig.savefig(out_path)
    print(f"  Fig S2 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure S4: NUTS2 map — dominant hazard and dominant CI system
# ---------------------------------------------------------------------------

def figS1_nuts2_map(agg_dir: Path, out_path: Path, presentation: bool = False):
    """NUTS2-level version of fig2: A = dominant hazard by EAD, B = dominant CI system."""

    print("  Loading NUTS2 aggregated files...")

    system_colors = SYSTEM_COLORS

    nuts2_hazard: dict = {}
    nuts2_system: dict = {}
    geom_df = None

    nuts2_files = sorted(agg_dir.glob("NUTS2_*_hazards.parquet"))
    if not nuts2_files:
        print(f"  No NUTS2 files found in {agg_dir}")
        return

    print(f"  Found {len(nuts2_files)} NUTS2 files")
    for path in nuts2_files:
        part = path.stem.replace("NUTS2_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue
        system = part if part in SYSTEM_ORDER else None

        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            try:
                gdf = pd.read_parquet(path)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
                continue

        id_col = next((c for c in ["NUTS_ID", "NUTS2", "NUTS2_ID"] if c in gdf.columns), None)
        if id_col is None:
            continue

        if geom_df is None and "geometry" in gdf.columns:
            geom_df = gdf[[id_col, "geometry"]].copy().rename(columns={id_col: "NUTS_ID"})

        for _, row in gdf.iterrows():
            nid = row[id_col]

            for h in EAD_HAZARDS:
                col = f"EAD_mid_total_{h}_current"
                if col not in gdf.columns:
                    col = next((c for c in gdf.columns
                                if "EAD_mid" in c and h in c and "current" in c), None)
                if col is None:
                    continue
                val = float(row[col]) if not pd.isna(row[col]) else 0.0
                if nid not in nuts2_hazard:
                    nuts2_hazard[nid] = {hh: 0.0 for hh in EAD_HAZARDS}
                nuts2_hazard[nid][h] = nuts2_hazard[nid].get(h, 0.0) + val

            if system:
                # Prefer the exact "_total_" column, same as the hazard loop
                # above — a loose substring match here previously could grab
                # a per-subtype column (e.g. "EAD_mid_aerodrome_..." before
                # "EAD_mid_total_..."), undercounting the system's true EAD.
                sys_total = 0.0
                for h in EAD_HAZARDS:
                    c = f"EAD_mid_total_{h}_current"
                    if c not in gdf.columns:
                        c = next((cc for cc in gdf.columns
                                  if "EAD_mid" in cc and h in cc and "current" in cc), None)
                    if c is not None and not pd.isna(row[c]):
                        sys_total += float(row[c])
                if nid not in nuts2_system:
                    nuts2_system[nid] = {ss: 0.0 for ss in SYSTEM_ORDER}
                nuts2_system[nid][system] = nuts2_system[nid].get(system, 0.0) + sys_total

    if not nuts2_hazard or geom_df is None:
        print("  Could not build NUTS2 EAD lookup")
        return

    haz_df = pd.DataFrame.from_dict(nuts2_hazard, orient="index").fillna(0)
    haz_df.index.name = "NUTS_ID"
    haz_df["total"] = haz_df[EAD_HAZARDS].sum(axis=1)
    haz_df["dominant_hazard"] = haz_df[EAD_HAZARDS].idxmax(axis=1)
    haz_df.loc[haz_df["total"] == 0, "dominant_hazard"] = "none"
    haz_df = haz_df.reset_index()

    sys_df = pd.DataFrame.from_dict(nuts2_system, orient="index").fillna(0)
    sys_df.index.name = "NUTS_ID"
    present = [s for s in SYSTEM_ORDER if s in sys_df.columns]
    sys_df["total_sys"] = sys_df[present].sum(axis=1)
    sys_df["dominant_system"] = sys_df[present].idxmax(axis=1)
    sys_df.loc[sys_df["total_sys"] == 0, "dominant_system"] = "none"
    sys_df = sys_df.reset_index()

    map_gdf = (geom_df
               .merge(haz_df[["NUTS_ID", "dominant_hazard"]], on="NUTS_ID", how="left")
               .merge(sys_df[["NUTS_ID", "dominant_system"]], on="NUTS_ID", how="left"))
    map_gdf["dominant_hazard"] = map_gdf["dominant_hazard"].fillna("none")
    map_gdf["dominant_system"] = map_gdf["dominant_system"].fillna("none")

    EUROPE_XLIM = (2_500_000, 6_500_000)
    EUROPE_YLIM = (1_400_000, 5_500_000)

    ne_path = Path(__file__).parent.parent / "data" / "ne_10m_admin_0_countries.shp"
    ne_gdf = None
    if ne_path.exists():
        try:
            ne_gdf = gpd.read_file(ne_path).to_crs("EPSG:3035")
        except Exception as e:
            print(f"  Warning: could not load NE background: {e}")

    def _draw_base(ax):
        ax.set_facecolor("#C8DCE8")
        ax.set_aspect("equal")
        if ne_gdf is not None:
            ne_gdf.plot(ax=ax, color="#D8D8D8", edgecolor="#AAAAAA",
                        linewidth=0.3, zorder=1)
        ax.set_xlim(EUROPE_XLIM)
        ax.set_ylim(EUROPE_YLIM)
        ax.set_axis_off()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 10),
                                      gridspec_kw={"wspace": 0.04})

    # Panel A: dominant hazard
    _draw_base(ax_a)
    haz_cmap = {h: HAZARD_COLORS[h] for h in EAD_HAZARDS}
    haz_cmap["none"] = "#E8E8E8"
    for val, color in haz_cmap.items():
        sub = map_gdf[map_gdf["dominant_hazard"] == val]
        if len(sub):
            sub.plot(ax=ax_a, color=color, linewidth=0.3, edgecolor="white", zorder=2)

    haz_patches = [mpatches.Patch(color=HAZARD_COLORS[h], label=HAZARD_LABELS[h])
                   for h in EAD_HAZARDS]
    haz_patches.append(mpatches.Patch(color="#E8E8E8", label="No EAD"))
    ax_a.legend(handles=haz_patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                framealpha=0.95, fontsize=10, borderpad=1.0,
                handlelength=1.5, handleheight=1.2)
    if presentation:
        ax_a.set_title("A.  Dominant hazard by EAD (NUTS2, current)",
                       pad=6, fontsize=14, fontweight="bold", loc="center")
    else:
        ax_a.text(-0.02, 1.02, "A", transform=ax_a.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="left")

    # Panel B: dominant CI system
    _draw_base(ax_b)
    sys_cmap = {s: system_colors[s] for s in SYSTEM_ORDER}
    sys_cmap["none"] = "#E8E8E8"
    for val, color in sys_cmap.items():
        sub = map_gdf[map_gdf["dominant_system"] == val]
        if len(sub):
            sub.plot(ax=ax_b, color=color, linewidth=0.3, edgecolor="white", zorder=2)

    sys_patches = [mpatches.Patch(color=system_colors[s], label=SYSTEM_LABELS[s])
                   for s in SYSTEM_ORDER]
    sys_patches.append(mpatches.Patch(color="#E8E8E8", label="No EAD"))
    ax_b.legend(handles=sys_patches, loc="upper right", bbox_to_anchor=(0.97, 0.97),
                framealpha=0.95, fontsize=10, borderpad=1.0,
                handlelength=1.5, handleheight=1.2)
    if presentation:
        ax_b.set_title("B.  Dominant CI system by EAD (NUTS2, current)",
                       pad=6, fontsize=14, fontweight="bold", loc="center")
    else:
        ax_b.text(-0.02, 1.02, "B", transform=ax_b.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="left")

    fig.savefig(out_path)
    print(f"  Fig S1 saved: {out_path}")
    plt.close(fig)



# ---------------------------------------------------------------------------
# Figure S6: Dumbbell — climate model range by country, heat + wildfire
# ---------------------------------------------------------------------------

def figS6_dumbbell(agg_dir: Path, out_path: Path, presentation: bool = False):
    """
    Standalone dumbbell chart: min→max model range per country for heat and
    wildfire exposure at 2100 SSP5-8.5. Countries sorted by heat mean exposure.
    """
    PERIOD = "2100_SSP585"
    HCOLS = {h: HAZARD_COLORS[h] for h in ["heat", "wildfire"]}

    print("  Loading NUTS2 files for dumbbell...")

    cntr_acc: dict = {}

    for path in sorted(agg_dir.glob("NUTS2_*_hazards.parquet")):
        part = path.stem.replace("NUTS2_", "").replace("_hazards", "")
        for geom in ("_lines", "_points", "_polygons"):
            part = part.replace(geom, "")
        if part in EXCLUDED_SYSTEMS:
            continue
        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            try:
                gdf = pd.read_parquet(path)
            except Exception as e:
                print(f"    Skip {path.name}: {e}")
                continue

        id_col = next((c for c in ["NUTS_ID", "NUTS2", "NUTS2_ID"] if c in gdf.columns), None)
        if id_col is None:
            continue

        for _, row in gdf.iterrows():
            cid = row["CNTR_CODE"] if "CNTR_CODE" in gdf.columns else str(row[id_col])[:2]
            if cid not in cntr_acc:
                cntr_acc[cid] = {"asset_size": 0.0}
                for h in ("heat", "wildfire"):
                    for b in ("mean", "min", "max"):
                        cntr_acc[cid][f"{h}_{b}"] = 0.0

            for h in ("heat", "wildfire"):
                for b, col in [("mean", f"exposure_abs_total_{h}_{PERIOD}"),
                               ("min",  f"exposure_abs_min_total_{h}_{PERIOD}"),
                               ("max",  f"exposure_abs_max_total_{h}_{PERIOD}")]:
                    if col in gdf.columns and not pd.isna(row[col]):
                        cntr_acc[cid][f"{h}_{b}"] += float(row[col])

            size_col = "asset_size_total"
            if size_col in gdf.columns and not pd.isna(row[size_col]):
                cntr_acc[cid]["asset_size"] += float(row[size_col])

    cntr_df = pd.DataFrame.from_dict(cntr_acc, orient="index")
    cntr_df.index.name = "CNTR_CODE"
    cntr_df = cntr_df.reset_index()
    size_clip = cntr_df["asset_size"].clip(lower=1.0)
    for h in ("heat", "wildfire"):
        for b in ("mean", "min", "max"):
            cntr_df[f"{h}_{b}_pct"] = cntr_df[f"{h}_{b}"] / size_clip * 100

    N = min(30, len(cntr_df))
    top = cntr_df.nlargest(N, "heat_mean_pct").reset_index(drop=True)
    y_labels = top["CNTR_CODE"].tolist()
    y_pos = np.arange(len(y_labels))
    OFFSET = 0.18

    fig, ax = plt.subplots(figsize=(10, max(6, N * 0.42)))

    for i, row in top.iterrows():
        for h, sign in [("heat", +OFFSET), ("wildfire", -OFFSET)]:
            yy = y_pos[i] + sign
            lo = row[f"{h}_min_pct"]
            hi = row[f"{h}_max_pct"]
            mid = row[f"{h}_mean_pct"]
            ax.plot([lo, hi], [yy, yy],
                    color=HCOLS[h], linewidth=2.5, alpha=0.7,
                    solid_capstyle="round", zorder=2)
            ax.scatter(mid, yy, color=HCOLS[h], s=50, zorder=4,
                       edgecolors="white", linewidths=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("% of CI assets exposed, 2100 SSP5-8.5", fontsize=12)
    ax.set_title("Climate model range by country\nbar = min→max, dot = mean  |  sorted by heat exposure",
                 fontsize=12, pad=8)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    heat_patch = mpatches.Patch(color=HCOLS["heat"],     label=HAZARD_LABELS["heat"])
    wild_patch = mpatches.Patch(color=HCOLS["wildfire"], label=HAZARD_LABELS["wildfire"])
    ax.legend(handles=[heat_patch, wild_patch], loc="lower right", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  Fig S6 saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for figures (default: repo_root/figures/)")
    parser.add_argument("--fig", type=str, default=None,
                        help="Generate only this figure: 1-4 (main), s1-s9 (supplementary)")
    parser.add_argument("--miraca-dir", default=None)
    parser.add_argument("--agg-dir", default=None)
    parser.add_argument("--presentation", action="store_true",
                        help="Generate presentation variants (with subtitles) instead of paper")
    parser.add_argument("--skip-fig2", action="store_true",
                        help="Skip Fig 2 (LAU map, slow)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    miraca_dir = Path(args.miraca_dir) if args.miraca_dir else repo_root / "MIRACA_OUTPUT_FULL"
    agg_dir = Path(args.agg_dir) if args.agg_dir else repo_root / "MIRACA_AGGREGATED_FULL"

    base_dir = Path(args.output_dir) if args.output_dir else repo_root / "figures"
    out_dir = base_dir / ("presentation" if args.presentation else "paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    pres = args.presentation
    print(f"\nMIRACA_OUTPUT: {miraca_dir}")
    print(f"MIRACA_AGGREGATED: {agg_dir}")
    print(f"Output dir: {out_dir}  ({'presentation' if pres else 'paper'})\n")

    fig_arg = args.fig.lower() if args.fig else None
    run_all = fig_arg is None

    def _want(key: str) -> bool:
        return run_all or fig_arg == key

    # Figures needing summary_statistics.csv (the rest read the aggregates)
    if run_all or fig_arg in ("1", "3", "4", "s7", "s8"):
        print("Loading summary_statistics.csv...")
        df = load_summary(miraca_dir)
        df = df[~df["system"].isin(EXCLUDED_SYSTEMS)]

    # ---------------- Main text ----------------

    if _want("1"):
        print("Generating Fig 1 (exposure and risk overview)...")
        fig1_overview(df, out_dir / "fig1_overview.png", presentation=pres)

    if _want("2") and not args.skip_fig2:
        print("Generating Fig 2 (LAU dominant hazard / system map — slow)...")
        fig2_lau_map(agg_dir, out_dir / "fig2_lau_map.png", presentation=pres)
    elif args.skip_fig2 and fig_arg == "2":
        print("Fig 2 skipped (--skip-fig2)")

    if _want("3"):
        print("Generating Fig 3 (heat + wildfire exposure trajectories and maps)...")
        fig3_heat_wildfire(df, agg_dir, out_dir / "fig3_heat_wildfire.png",
                           presentation=pres)

    if _want("4"):
        print("Generating Fig 4 (river + coastal flood risk)...")
        fig4_flood_risk(df, agg_dir, out_dir / "fig4_flood_risk.png",
                        presentation=pres)

    # ---------------- Supplementary ----------------

    if _want("s1"):
        print("Generating Fig S1 (NUTS2 dominant hazard + system map)...")
        figS1_nuts2_map(agg_dir, out_dir / "figS1_nuts2_map.png", presentation=pres)

    if _want("s2"):
        print("Generating Fig S2 (multi-hazard exposure map)...")
        figS2_multihazard_map(agg_dir, out_dir / "figS2_multihazard_map.png",
                              presentation=pres)

    if _want("s3"):
        print("Generating Fig S3 (uncertainty map per NUTS2)...")
        figS3_uncertainty_map(agg_dir, out_dir / "figS3_uncertainty_map.png",
                              presentation=pres)

    if _want("s4") or _want("s5"):
        sub_path = miraca_dir / "subsystem_summary.csv"
        sub_df = pd.read_csv(sub_path) if sub_path.exists() else None
        if sub_df is None:
            print(f"  WARNING: {sub_path} not found — run compute_subsystem_summary.py first")

    if _want("s4"):
        print("Generating Fig S4 (uncertainty heatmap)...")
        figS4_uncertainty_heatmap(sub_df, out_dir / "figS4_uncertainty_heatmap.png",
                                  presentation=pres)

    if _want("s5"):
        print("Generating Fig S5 (subsystem drivers)...")
        figS5_subsystem(sub_df, out_dir / "figS5_subsystem.png", presentation=pres)

    if _want("s6"):
        print("Generating Fig S6 (dumbbell: climate model range by country)...")
        figS6_dumbbell(agg_dir, out_dir / "figS6_dumbbell.png", presentation=pres)

    if _want("s7"):
        print("Generating Fig S7 (future changes, all hazards)...")
        figS7_future_changes(df, out_dir / "figS7_future_changes.png", presentation=pres)

    if _want("s8"):
        print("Generating Fig S8 (sector profiles)...")
        figS8_sector_profiles(df, out_dir / "figS8_sector_profiles.png", presentation=pres)

    if _want("s9"):
        print("Generating Fig S9 (2nd/3rd dominant CI)...")
        figS9_dominant_ci_rank(agg_dir, out_dir / "figS9_dominant_ci_rank.png",
                               presentation=pres)

    print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
