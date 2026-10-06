#!/bin/bash
# Rerun energy network TENT outputs after vuln_ratio normalization fix
# (risk_integration_tent.py compute_ratio_and_exposure_per_rp).
#
# No corridor/country filter (not supported for energy assets). Existing
# files were run as a single combined-hazard job per asset --
# TENT_energy_{lines,buses}_coastal_earthquake_river_windstorm.parquet --
# so this omits --hazards to match that filename and overwrite it.
#
# energy_buses skips windstorm entirely (substations not wind-sensitive,
# hardcoded EAD=0) -- no vuln_ratio_windstorm_* columns there, expected.

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/MIRACA_RISK/TENT/logs/slurm

mkdir -p "$LOG_DIR"

submit() {
    local asset=$1
    local mem=${2:-10000}
    local name="tent_${asset}_vulnfix"
    cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=${mem}
#SBATCH --output=${LOG_DIR}/out_${name}
#SBATCH --error=${LOG_DIR}/err_${name}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
uv run python ${SCRIPT} --assets ${asset} --no-skip-existing --workers 1
SLURM
}

submit energy_lines 10000
submit energy_buses 10000
