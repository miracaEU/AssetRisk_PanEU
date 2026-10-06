#!/bin/bash
# run_aggregate_all.sh
# Submit one aggregate job per system in parallel.
# Submit with: bash run_aggregate_all.sh

INPUT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/MIRACA_OUTPUT_FULL
OUTPUT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/MIRACA_AGGREGATED_FULL
LOG=/scistor/ivm/eks510/MIRACA_RISK/logs
SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src/aggregate_outputs.py

mkdir -p "$LOG"

submit() {
    local system="$1"
    local mem="$2"   # MB per cpu
    local cpus="$3"
    cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=agg_${system}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=${mem}
#SBATCH --time=02:00:00
#SBATCH --output=${LOG}/out_agg_${system}
#SBATCH --error=${LOG}/err_agg_${system}

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export UV_PROJECT_ENVIRONMENT=/scistor/ivm/eks510/.venv/assetrisk-paneu

cd /scistor/ivm/eks510/projects/AssetRisk_PanEU

uv run python ${SCRIPT} --systems ${system} \
    --input-dir ${INPUT} \
    --output-dir ${OUTPUT}
SLURM
}

# Large
submit roads  16000 4
submit rail   16000 4

# Medium
submit power  12000 2
submit telecom 8000 2

# Small
submit airports  8000 1
submit ports     8000 1
submit education 8000 1
submit healthcare 8000 1
submit gas       8000 1
submit oil       8000 1
