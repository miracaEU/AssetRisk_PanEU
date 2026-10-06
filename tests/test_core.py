"""Unit tests on tiny synthetic inputs (no data files or config.yml needed).

Covers the trapezoidal EAD integration, protection standards, the heat/wildfire
exposure threshold and the country x system summary statistics.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

import constants as C
import merge_outputs as M
import risk_integration as RI


def _d(curve):
    return {rp: {"mean": v, "min": v, "max": v} for rp, v in curve.items()}


# --------------------------------------------------------------------- EAD
def test_ead_trapezoid_no_protection():
    ead = RI.integrate_ead(_d({10: 1.0, 100: 5.0, 500: 9.0}))[0]
    expected = 0.5 * (1 + 5) * (1 / 10 - 1 / 100) + 0.5 * (5 + 9) * (1 / 100 - 1 / 500)
    assert ead == pytest.approx(expected)


def test_ead_protection_between_rps_interpolates():
    # protection at RP 50: damage at 50 interpolated linearly in RP between 10 and 100
    ead = RI.integrate_ead(_d({10: 1.0, 100: 5.0, 500: 9.0}), protection_standard=50)[0]
    d50 = 1.0 + (50 - 10) * (5.0 - 1.0) / (100 - 10)
    expected = 0.5 * (d50 + 5) * (1 / 50 - 1 / 100) + 0.5 * (5 + 9) * (1 / 100 - 1 / 500)
    assert ead == pytest.approx(expected)


def test_ead_protection_equal_to_max_rp_is_single_point():
    # pipeline behaviour: only the largest RP survives, EAD = D(RP) / RP
    ead = RI.integrate_ead(_d({1: 0.0, 100: 2.0, 1000: 7.0}), protection_standard=1000)[0]
    assert ead == pytest.approx(7.0 / 1000)


def test_ead_min_mean_max_and_empty():
    d = {10: {"mean": 2.0, "min": 1.0, "max": 4.0}, 100: {"mean": 4.0, "min": 2.0, "max": 8.0}}
    mean, lo, hi = RI.integrate_ead(d)
    assert lo < mean < hi
    assert RI.integrate_ead({}) == (0.0, 0.0, 0.0)


def test_collect_ead_per_asset_applies_protection_per_asset():
    feats = gpd.GeoDataFrame({"osm_id": [1, 2]}, geometry=[Point(0, 0)] * 2, crs=3035)
    rp = {}
    for r, f in [(10, 1.0), (100, 3.0)]:
        g = gpd.GeoDataFrame(index=feats.index)
        g["damage_mean"] = g["damage_min"] = g["damage_max"] = [f, f]
        rp[r] = g
    out = RI.collect_ead_per_asset(rp, feats, pd.Series([0, 100], index=feats.index))
    assert out.loc[0, "EAD_mid"] == pytest.approx(0.5 * (1 + 3) * (0.1 - 0.01))
    assert out.loc[1, "EAD_mid"] == pytest.approx(3.0 / 100)


# --------------------------------------------------------------------- thresholds
def test_country_threshold_p90_of_nonzero_values_with_floor(tmp_path):
    n = 20
    days = np.r_[np.zeros(10), np.arange(1, 11, dtype=float)]           # positives 1..10
    north = np.r_[np.zeros(18), [0.2, 0.4]]                              # p90 < 1 -> floor
    for iso, vals in [("ESP", days), ("NOR", north)]:
        gdf = gpd.GeoDataFrame({"osm_id": range(n), "exposure_abs_heat_35C_current": vals,
                                "exposure_abs_wildfire_current": vals},
                               geometry=[Point(i, 0) for i in range(n)], crs=3035)
        gdf.to_parquet(tmp_path / f"{iso}_rail_exposure.parquet")
    pairs = [{"iso3": iso, "system": "rail", "hazard_path": None,
              "exposure_path": tmp_path / f"{iso}_rail_exposure.parquet"} for iso in ["ESP", "NOR"]]
    t = M.compute_country_thresholds(pairs, heat_threshold="35C")
    assert t["ESP"]["heat"] == pytest.approx(np.percentile(np.arange(1, 11), 90))
    assert t["NOR"]["heat"] == M.FALLBACK_THRESHOLD == 1.0


# --------------------------------------------------------------------- summary
def test_compute_summary_counts_and_sums():
    gdf = pd.DataFrame({"EAD_mid_river_current": [0.0, 2.0, 3.0, 0.0],
                        "exposure_abs_river_current": [0.0, 5.0, 0.0, 1.0],
                        "exposure_rel_river_current": [0.0, 1.0, 0.0, 1.0]})
    rows = pd.DataFrame(M.compute_summary(gdf, "NLD", "rail"))
    ead = rows[(rows.metric_type == "EAD") & (rows.metric_bound == "mid")].iloc[0]
    assert ead["sum"] == 5.0 and ead["n_exposed"] == 2 and ead["pct_exposed"] == 50.0
    rel = rows[rows.metric_type == "exposure_rel"].iloc[0]
    assert rel["n_exposed"] == 2 and rel["count"] == 4


# --------------------------------------------------------------------- shared settings
def test_shared_settings():
    assert len(C.PAPER_SYSTEMS) == 8 and "ports" in C.PAPER_SYSTEMS
    assert C.EXCLUDED_SYSTEMS == {"oil", "gas"} and not C.EXCLUDED_SYSTEMS & set(C.PAPER_SYSTEMS)
    assert C.WINDOW_TO_PERIOD["mid_future"] == "2050" and C.WINDOW_TO_PERIOD["far_future"] == "2100"
    assert C.RIVER_FUTURE_TEMP_LABELS == tuple(C.FUTURE_PERIODS)
    assert set(C.EAD_HAZARDS) < set(C.EXPOSURE_HAZARDS)
