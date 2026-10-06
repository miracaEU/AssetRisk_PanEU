#!/bin/bash

# River flood — airports
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_airports_river
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_airports_river
#SBATCH --error=err_tent_airports_river
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets airports --hazards river --no-skip-existing --workers 1
SLURM

# River flood — ports
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_ports_river
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_ports_river
#SBATCH --error=err_tent_ports_river
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets ports --hazards river --no-skip-existing --workers 1
SLURM

# River flood — iww
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_iww_river
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_iww_river
#SBATCH --error=err_tent_iww_river
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets iww --hazards river --no-skip-existing --workers 1
SLURM

