#!/bin/bash
SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/MIRACA_RISK/TENT/logs/slurm

mkdir -p "$LOG_DIR"

cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_airports_coastal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=${LOG_DIR}/out_tent_airports_coastal
#SBATCH --error=${LOG_DIR}/err_tent_airports_coastal
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
uv run python ${SCRIPT} --assets airports --hazards coastal --no-skip-existing --workers 1
SLURM
