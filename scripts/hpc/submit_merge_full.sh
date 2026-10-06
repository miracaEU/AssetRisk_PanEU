#!/bin/bash
# submit_merge_full.sh
#
# Full re-merge of MIRACA_OUTPUT_FULL after the landslide point-cap and
# coastal sum-then-clip fixes. Uses cached country_thresholds.csv (heat/
# wildfire exposure data unchanged by these fixes) to skip the slowest step.
# For a first run from scratch, remove --use-cached-thresholds so the
# country p90 thresholds are computed (written to country_thresholds.csv).

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src/merge_outputs.py
LOG_DIR=/scistor/ivm/eks510/projects/AssetRisk_PanEU/logs
mkdir -p "$LOG_DIR"

cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=merge_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=30000
#SBATCH --output=${LOG_DIR}/out_merge_full
#SBATCH --error=${LOG_DIR}/err_merge_full
cd /scistor/ivm/eks510/projects/AssetRisk_PanEU
uv run python ${SCRIPT} --overwrite --use-cached-thresholds
SLURM
