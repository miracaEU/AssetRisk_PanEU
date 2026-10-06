#!/bin/bash
job_script="MIRACA_EXPOSURE_ALL.slurm"
cat <<SLURM > $job_script
#!/bin/bash
#SBATCH --job-name=miraca_exposure
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=25000
#SBATCH --output=out_miraca_exposure
#SBATCH --error=err_miraca_exposure
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src/run_exposure_pipeline.py \
    --workers 4
SLURM
sbatch $job_script
rm $job_script