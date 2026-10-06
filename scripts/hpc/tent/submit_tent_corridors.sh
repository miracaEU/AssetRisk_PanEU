#!/bin/bash
# Submit TENT corridor risk assessment for all TEN-T corridors × transport assets × hazards.
# One SLURM job per corridor × asset × hazard (72 jobs: 9 corridors × roads/rail × 4 hazards).
# Corridor L already done -- --skip-existing will skip completed outputs.
#
# Corridor letter codes (from CORRIDORS column in TENT parquets):
#   A=Atlantic  B=NorthSea-Baltic  C=NorthSea-Med  E=Scandinavian-Med
#   G=Mediterranean  I=Rhine-Danube  J=Orient/EastMed  K=Baltic-Adriatic  L=Rhine-Alpine
#
# Output per job: TENT_{asset}_corridor{X}_{hazard}.parquet in MIRACA_RISK/TENT/

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/projects/AssetRisk_PanEU/logs

mkdir -p "$LOG_DIR"

CORRIDORS=(A B C E G I J K L)
ASSETS=(roads rail)
HAZARDS=(river coastal windstorm earthquake)

for CORRIDOR in "${CORRIDORS[@]}"; do
    for ASSET in "${ASSETS[@]}"; do
        for HAZARD in "${HAZARDS[@]}"; do
            JOB_NAME="tent_${ASSET}_cor${CORRIDOR}_${HAZARD}"
            cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=8000
#SBATCH --output=${LOG_DIR}/out_${JOB_NAME}
#SBATCH --error=${LOG_DIR}/err_${JOB_NAME}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
uv run python ${SCRIPT} --assets ${ASSET} --corridor ${CORRIDOR} --hazards ${HAZARD} --workers 4 --skip-existing
SLURM
            echo "Submitted: ${JOB_NAME}"
        done
    done
done
