#!/bin/bash
#SBATCH --job-name=tent_roads_coast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=75000
#SBATCH --output=out_tent_roads_coast
#SBATCH --error=err_tent_roads_coast

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

/scistor/ivm/eks510/projects/AssetRisk_PanEU/.venv/bin/python \
    /scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py \
    --assets rail --hazards coastal --workers 1