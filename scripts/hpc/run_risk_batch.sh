#!/bin/bash

job_script="MIRACA_RUN_RISK.slurm"
cat <<SLURM > $job_script
#!/bin/bash
#SBATCH --job-name=miraca_risk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=35000
#SBATCH --output=out_miraca_risk
#SBATCH --error=err_miraca_risk
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src/run_pipeline.py \
    --assets rail telecom power ports \
    --workers 4
    
SLURM
sbatch $job_script