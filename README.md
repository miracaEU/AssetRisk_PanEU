# Pan-European Asset Level Risk Assessment

## Overview

This repository contains the codebase for a pan-European, asset-level multi-hazard risk assessment pipeline. For each country and infrastructure asset type, the pipeline:

1. Loads exposure data from a harmonised exposure database
2. Runs hazard assessments for river flooding, coastal flooding, windstorms, and earthquakes
3. Outputs enriched GeoDataFrames as parquet files with risk metrics per asset

> **Note:** Hazard data, exposure databases, vulnerability curves, and other input files are **not included in this repository** and must be obtained and configured separately. See the [Configuration](#configuration) section below.

---

## Repository Structure

```
├── src/
│   ├── run_pipeline.py         # Top-level pipeline entry point
│   ├── data_loader.py          # Exposure data loading and country/asset discovery
│   ├── constants.py            # Shared constants (asset types, ISO codes, etc.)
│   ├── hazard_river.py         # River flood hazard assessment
│   ├── hazard_coastal.py       # Coastal flood hazard assessment
│   ├── hazard_windstorm.py     # Windstorm hazard assessment
│   ├── hazard_earthquake.py    # Earthquake hazard assessment
│   └── risk_integration.py    # EAD integration and risk aggregation
├── config.template.yml         # Template for machine-specific paths (see below)
├── config.yml                  # Your local config — gitignored, never committed
└── README.md
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart your shell
```

### 2. Clone the repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

### 3. Create environment and install dependencies

```bash
uv sync
```

To run scripts directly:

```bash
uv run python src/run_pipeline.py --help
```

---

## Configuration

All machine-specific paths (to hazard data, exposure files, output directories, etc.) are managed through a local `config.yml` file. This file is **gitignored** and must be created separately on each machine or HPC environment.

### Setup

Copy the template and fill in the paths for your environment:

```bash
cp config.template.yml config.yml
```

Then edit `config.yml`:

```yaml
exposure_dir: /path/to/exposure/files
output_dir: /path/to/output
river_hazard_dir: /path/to/hazard/river
wind_hazard_dir: /path/to/hazard/wind
eq_hazard_dir: /path/to/hazard/earthquakes
vulnerability_path: /path/to/repo/data/vulnerability_curves.xlsx
fragility_path: /path/to/repo/data/EQ_fragility.xlsx
protection_standard_path: /path/to/hazard/floodProtection_v2019_paper3.tif
basin_data_path: /path/to/repo/data/basins_abs_shift_return_periods.parquet
coastal_stac_url: "https://storage.googleapis.com/coclico-data-public/coclico/coclico-stac/catalog.json"
```

### External data requirements

The following data are **not part of this repository** and must be connected separately:

| Data | Description |
|------|-------------|
| Exposure database | Harmonised asset-level exposure files per country and asset type (parquet) |
| River flood hazard | Return-period flood depth rasters |
| Windstorm hazard | Return-period wind speed rasters |
| Earthquake hazard | Ground motion rasters |
| Flood protection standard | `floodProtection_v2019_paper3.tif` raster |
| Vulnerability curves | Excel file with fragility/vulnerability curves per hazard and asset type |
| EQ fragility curves | Excel file with earthquake fragility curves |
| Basin climate data | Parquet file with basin-level RP shifts for future river flood scenarios |

Contact the MIRACA project team for access to these datasets.

---

## Usage

```bash
# Run full pipeline (all countries, all assets, all hazards)
uv run python src/run_pipeline.py

# Specific countries
uv run python src/run_pipeline.py --countries PRT ESP ITA

# Specific asset types
uv run python src/run_pipeline.py --assets power roads

# Specific hazards only
uv run python src/run_pipeline.py --hazards river coastal

# Limit parallel workers
uv run python src/run_pipeline.py --workers 4

# Skip already-processed output files
uv run python src/run_pipeline.py --skip-existing
```

### Parallelism note

When `--workers` is greater than 1, inner return-period-level parallelism is automatically disabled to avoid nested process pools and memory exhaustion. Set `--workers 1` to run country/asset combinations sequentially with full inner parallelism enabled.

---

## Acknowledgements

[MIRACA](https://miraca-project.eu) (Multi-hazard Infrastructure Risk Assessment for Climate Adaptation) is a research project building an evidence-based decision support toolkit that meets real world demands.

This project has received funding from the European Union's Horizon Europe research programme under grant agreement No 101004174.

---