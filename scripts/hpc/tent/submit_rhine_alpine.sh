#!/bin/bash
SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/MIRACA_RISK/TENT/logs/slurm

mkdir -p "$LOG_DIR"

# Rail — Rhine-Alpine corridor (L)
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_rail_corridorL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=${LOG_DIR}/out_tent_rail_corridorL
#SBATCH --error=${LOG_DIR}/err_tent_rail_corridorL
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
uv run python ${SCRIPT} --assets rail --corridor L --no-skip-existing --workers 1
SLURM

# Roads — Rhine-Alpine corridor (L)
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_roads_corridorL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=${LOG_DIR}/out_tent_roads_corridorL
#SBATCH --error=${LOG_DIR}/err_tent_roads_corridorL
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
uv run python ${SCRIPT} --assets roads --corridor L --no-skip-existing --workers 1
SLURM
