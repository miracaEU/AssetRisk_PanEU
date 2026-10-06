# Pan-European Asset-Level Exposure and Risk to Critical Infrastructure

Code for the paper *"Pan-European natural hazard exposure and risk to critical infrastructure"* (Koks et al., submitted to Environmental Research Letters).

The pipeline intersects about 58 million OpenStreetMap infrastructure assets in 8 systems (roads, rail, airports, ports, power, telecommunications, education, healthcare) and 36 European countries with 7 hazards:
- **Expected annual damage (EAD):** river flooding, coastal flooding, windstorm and earthquake (exposure is computed too).
- **Exposure only:** extreme heat, wildfire and landslide.

> Input data and pipeline outputs are **not** part of this repository. See [Input data](#input-data).

---

## Repository layout

```
├── src/                          # Paper pipeline (everything needed for the figures and tables)
│   ├── constants.py              # Single config module: curves, max damages, ISO codes, periods,
│   │                             #   systems, road classes, load_config()
│   ├── data_loader.py            # Exposure file discovery and loading
│   ├── run_exposure_pipeline.py  # Step 1: heat / wildfire / landslide exposure
│   ├── exposure_heat.py, exposure_wildfire.py, exposure_landslide.py, exposure_utils.py
│   ├── run_pipeline.py           # Step 2: risk (EAD + exposure) for river, coastal, windstorm, earthquake
│   ├── hazard_river.py, hazard_coastal.py, hazard_windstorm.py, hazard_earthquake.py
│   ├── risk_integration.py       # Damage per return period, protection standards, trapezoidal EAD
│   ├── merge_outputs.py          # Step 3: merge, heat/wildfire p90 thresholds, summary_statistics.csv
│   ├── river_exposure_interp.py  # Step 4: future river exposure (RP-shift interpolation)
│   ├── aggregate_outputs.py      # Step 5: LAU / NUTS2 / country aggregates
│   ├── compute_subsystem_summary.py  # Step 6: per object type summary (Figs S4, S5)
│   ├── figures.py                # Step 7: all main and supplementary figures
│   ├── supplementary_tables.py   # Step 8: Supplementary Tables S1–S4
│   └── tools/build_basins_rp_shift.py  # input prep: basin RP-shift factors for future river
├── src_tent/                     # Separate TEN-T corridor workflow (not used by the paper);
│                                 #   reuses the hazard/risk modules in src/
├── scripts/hpc/                  # SLURM launchers for the full run (and scripts/hpc/tent/)
├── book/                         # Exploratory and data-preparation notebooks
├── tests/                        # Unit tests (pytest)
├── config.template.yml           # Copy to config.yml and fill in paths
├── pyproject.toml, uv.lock       # Reference environment (uv)
├── requirements.txt              # pip alternative (pinned)
└── environment.yaml              # conda alternative
```

---

## Installation

With [uv](https://docs.astral.sh/uv/) (reference environment, Python 3.13):

```bash
git clone https://github.com/miracaEU/AssetRisk_PanEU.git
cd AssetRisk_PanEU
uv sync
```

Or with pip (`pip install -r requirements.txt`) or conda (`conda env create -f environment.yaml`).

`cmcrameri` is optional. The paper figures were produced **without** it, using the fallback colour maps in `src/figures.py`; installing it changes the colours. `python-docx` is optional and is only needed for the Word export of the supplementary tables.

## Configuration

```bash
cp config.template.yml config.yml   # gitignored; holds machine-specific paths
```

All scripts read `config.yml` from the repository root. `heat_threshold: "35C"` selects the heat indicator used in the paper.

---

## Input data

| Input | Source | Used by |
|---|---|---|
| Exposure database: OSM assets per country and system, split at LAU boundaries (`Exposure_files/{System}/{System}_{ISO2}.parquet`) | Extracted from OpenStreetMap by the MIRACA project; the extraction code is not part of this repository. Contact the authors | all steps |
| River flood depth maps, RP 10 to 500 (`Europe_RP{rp}_filled_depth.tif`) | Baugh et al. (2026), JRC: https://data.jrc.ec.europa.eu/dataset/1d128b6c-a4ee-4858-9e34-6210707f3c81 | `hazard_river.py` |
| River flood protection standards (`floodProtection_v2019_paper3.tif`) | Dottori et al. (2023), JRC | `hazard_river.py` |
| River discharge change per warming level (`disEnsemble_highExtremes_{10,100,500}.nc`) and HydroBASINS lev08 | Mentaschi et al. (2020), JRC PESETA IV; HydroBASINS (hydrosheds.org) | `tools/build_basins_rp_shift.py` → `data/basins_abs_shift_return_periods.parquet` |
| Coastal flood maps (2010, 2050, 2100; RP 1, 100, 1000) | CoCliCo STAC catalogue, read online: https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json | `hazard_coastal.py` |
| Coastal protection standards (`COASTPROS-EU.xlsx`) | van Maanen et al. (2025) | `hazard_coastal.py` |
| Windstorm 3 s gust return-period footprints | Priestley et al. (2023) | `hazard_windstorm.py` |
| Earthquake PGA (`PGA_1_{50,101,476,976,2500,5000}_vs30.tif`) | ESHM20, Danciu et al. (2024), EFEHR | `hazard_earthquake.py` |
| Hot days ≥30/35/40 °C and days with high fire danger, monthly reanalysis + projections | Copernicus CDS *Climate indicators for Europe* (`sis-ecde-climate-indicators`), C3S (2024) | `exposure_heat.py`, `exposure_wildfire.py` |
| Landslide susceptibility (`elsus_v2.asc`) | ELSUS v2, Wilde et al. (2018), ESDAC | `exposure_landslide.py` |
| Vulnerability curves (`Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx`) and maximum damages | Nirandjan et al. (2024), NHESS 24, 4341–68 | `hazard_*.py`, `constants.py` |
| Earthquake fragility curves (`EQ_fragility.xlsx`, including the `-C` building curves) | MIRACA project (see manuscript Methods) | `hazard_earthquake.py` |
| NUTS 2024 / 2016 and LAU 2024 / 2016 boundaries | Eurostat GISCO | `aggregate_outputs.py`, `hazard_coastal.py` |
| Natural Earth admin-0 countries, 1:10m | naturalearthdata.com | map backgrounds in `figures.py` |

Files expected in `data/`:
- `COASTPROS-EU.xlsx`
- `EQ_fragility.xlsx`
- `Table_D2_…_conversions.xlsx`
- `NUTS_RG_20M_2024_3035.geojson`
- `LAU_RG_01M_2024_3035.parquet`
- `basins_abs_shift_return_periods.parquet`
- `floodProtection_v2019_paper3.tif`
- `ne_10m_admin_0_countries.*`

---

## Reproducing the paper

The full run (steps 1–2) takes days on an HPC cluster. The SLURM launchers are in `scripts/hpc/`. They contain the absolute repository path `/scistor/ivm/eks510/projects/AssetRisk_PanEU`; edit it for your system.

| Step | Command | Output |
|---|---|---|
| 1 Exposure (heat, wildfire, landslide) | `uv run python src/run_exposure_pipeline.py` (`scripts/hpc/run_exposure_batch.sh`) | `MIRACA_EXPOSURE/{ISO3}_{system}_exposure.parquet` |
| 2 Risk (river, coastal, windstorm, earthquake) | `uv run python src/run_pipeline.py` (`scripts/hpc/run_risk_batch.sh`) | `MIRACA_RISK/{ISO3}_{system}_hazards.parquet` |
| 3 Merge + thresholds + summary | `uv run python src/merge_outputs.py --overwrite` (`scripts/hpc/submit_merge_full.sh`) | `MIRACA_OUTPUT_FULL/*.parquet`, `summary_statistics.csv`, `country_thresholds.csv` |
| 4 Future river exposure | `uv run python src/river_exposure_interp.py`, then rename `summary_statistics_river_interp.csv` to `summary_statistics.csv` | patched summary |
| 5 Spatial aggregates | `uv run python src/aggregate_outputs.py` (`scripts/hpc/submit_aggregate_full.sh`) | `MIRACA_AGGREGATED_FULL/{LAU,NUTS2,NUTS0}_*_hazards.parquet` |
| 6 Subsystem summary | `uv run python src/compute_subsystem_summary.py --input-dir MIRACA_OUTPUT_FULL --output MIRACA_OUTPUT_FULL/subsystem_summary.csv` | `subsystem_summary.csv` |
| 7 Figures | `uv run python src/figures.py [--fig N] [--output-dir DIR]` | `figures/paper/*.png` |
| 8 Supplementary tables | `uv run python src/supplementary_tables.py [--exposure-dir DIR]` | `tables/S1–S4*.csv` (+ `.docx` with python-docx) |

Steps 7 and 8 only read stored results and run in minutes.

| Paper item | Command / function | Reads |
|---|---|---|
| Figure 1 | `figures.py --fig 1` → `fig1_overview` | `summary_statistics.csv` |
| Figure 2 | `figures.py --fig 2` → `fig2_lau_map` | LAU aggregates |
| Figure 3 | `figures.py --fig 3` → `fig3_heat_wildfire` | `summary_statistics.csv`, LAU aggregates |
| Figure 4 | `figures.py --fig 4` → `fig4_flood_risk` | `summary_statistics.csv`, NUTS2 aggregates |
| Figures S1 to S9 | `figures.py --fig s1` … `--fig s9` | aggregates / summaries (S4, S5: `subsystem_summary.csv`) |
| Tables S1 to S4 | `supplementary_tables.py` | exposure inputs, `constants.py`, curve files |

### Conventions to keep in mind

- **Time windows (heat, wildfire):**
  - baseline = **1990–2016** reanalysis mean;
  - "2050" = **2041–2060**;
  - the output label "2100" = **2061–2080**. Figures display it as 2070.
  - Scenarios are EURO-CORDEX **RCP4.5 / RCP8.5**, labelled SSP2-4.5 / SSP5-8.5.
- **River futures** are global warming levels, mapped to periods by convention: 1.5 °C → 2050 SSP2-4.5, 2 °C → 2050 SSP5-8.5, 3 °C → 2100 SSP2-4.5, 4 °C → 2100 SSP5-8.5.
- **Coastal futures** are the CoCliCo 2050 and 2100 horizons, relative to 2010.
- **Systems:** oil and gas are computed but excluded from every paper figure and table (`constants.EXCLUDED_SYSTEMS`).
- **Andorra duplicate:** Andorra appears twice in the current outputs (`AD` from the risk pipeline, `AND` from the exposure pipeline) because the ISO maps differ. Paper numbers drop `AD`.

## Tests

```bash
uv run --with pytest pytest tests
```

## Acknowledgements

This work received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101093854 (MIRACA, [miraca-project.eu](https://miraca-project.eu)).
