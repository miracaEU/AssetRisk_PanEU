#!/bin/bash
# Rerun coastal hazard for non-corridor TENT assets (full Europe) to add
# future scenario vuln_ratio columns (vuln_ratio_coastal_2050_SSP245_RP* etc.).
# Uses --no-skip-existing to overwrite existing coastal parquets.
# 5 jobs: airports, ports, iww, energy_lines, energy_buses.

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/projects/AssetRisk_PanEU/logs

mkdir -p "$LOG_DIR"

ASSETS=(airports ports iww energy_lines energy_buses)

for ASSET in "${ASSETS[@]}"; do
    JOB_NAME="tent_${ASSET}_coastal_future"
    cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=8000
#SBATCH --output=${LOG_DIR}/out_${JOB_NAME}
#SBATCH --error=${LOG_DIR}/err_${JOB_NAME}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
uv run python ${SCRIPT} --assets ${ASSET} --hazards coastal --workers 2 --no-skip-existing
SLURM
    echo "Submitted: ${ASSET}"
done
