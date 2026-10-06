#!/bin/bash
# Rerun coastal hazard for all TEN-T corridors x roads/rail to add future
# scenario vuln_ratio columns (vuln_ratio_coastal_2050_SSP245_RP* etc.).
# Uses --no-skip-existing to overwrite existing coastal parquets.
# 18 jobs: 9 corridors x 2 assets.

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/projects/AssetRisk_PanEU/logs

mkdir -p "$LOG_DIR"

CORRIDORS=(A B C E G I J K L)
ASSETS=(roads rail)

for CORRIDOR in "${CORRIDORS[@]}"; do
    for ASSET in "${ASSETS[@]}"; do
        JOB_NAME="tent_${ASSET}_cor${CORRIDOR}_coastal_future"
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
uv run python ${SCRIPT} --assets ${ASSET} --corridor ${CORRIDOR} --hazards coastal --workers 2 --no-skip-existing
SLURM
        echo "Submitted: ${JOB_NAME}"
    done
done
