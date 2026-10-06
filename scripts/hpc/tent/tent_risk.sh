#!/bin/bash

# River flood — roads
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_roads_river
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_roads_river
#SBATCH --error=err_tent_roads_river
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets roads --hazards river --workers 1 --country BEL
SLURM

# Earthquake — roads
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_roads_eq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_roads_eq
#SBATCH --error=err_tent_roads_eq
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets roads --hazards earthquake --workers 1 --country BEL
SLURM

# Windstorm — roads
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_roads_wind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_roads_wind
#SBATCH --error=err_tent_roads_wind
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets roads --hazards windstorm --workers 1 --country BEL
SLURM

# Coastal — roads
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_roads_coast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_roads_coast
#SBATCH --error=err_tent_roads_coast

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets roads --hazards coastal --workers 1 --country BEL
SLURM

# River flood — rail
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_rail_river
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_rail_river
#SBATCH --error=err_tent_rail_river
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets rail --hazards river --workers 1 --country BEL
SLURM

# Earthquake — rail
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_rail_eq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_rail_eq
#SBATCH --error=err_tent_rail_eq
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets rail --hazards earthquake --workers 1 --country BEL
SLURM

# Windstorm — rail
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_rail_wind
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_rail_wind
#SBATCH --error=err_tent_rail_wind
uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets rail --hazards windstorm --workers 1 --country BEL
SLURM

# Coastal — rail
cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=tent_rail_coast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=10000
#SBATCH --output=out_tent_rail_coast
#SBATCH --error=err_tent_rail_coast

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

uv run python /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets rail --hazards coastal --workers 1 --country BEL
SLURM