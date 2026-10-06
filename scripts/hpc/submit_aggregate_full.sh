#!/bin/bash
# submit_aggregate_full.sh
#
# Full re-aggregate (NUTS0/NUTS2/LAU, all systems) after the landslide
# point-cap and coastal sum-then-clip fixes. No --systems filter, so it
# picks up roads too (was stale at June 12 last check). Run only after
# submit_merge_full.sh has completed.

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src/aggregate_outputs.py
LOG_DIR=/scistor/ivm/eks510/projects/AssetRisk_PanEU/logs
mkdir -p "$LOG_DIR"

cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=aggregate_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=30000
#SBATCH --output=${LOG_DIR}/out_aggregate_full
#SBATCH --error=${LOG_DIR}/err_aggregate_full
cd /scistor/ivm/eks510/projects/AssetRisk_PanEU
uv run python ${SCRIPT}
SLURM
