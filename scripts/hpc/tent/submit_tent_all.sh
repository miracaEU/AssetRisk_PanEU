#!/bin/bash
# Submit all TENT risk pipeline jobs to SLURM
# Assets: airports, ports, iww, rail, roads
# Hazards: river, coastal, windstorm, earthquake

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/MIRACA_RISK/TENT/logs/slurm

mkdir -p "$LOG_DIR"

submit() {
    local asset=$1
    local hazard=$2
    local mem=${3:-10000}   # default 10GB
    local country=${4:-}    # optional country filter
    local name="tent_${asset}_${hazard}"
    local country_flag=""
    if [ -n "$country" ]; then
        name="${name}_${country}"
        country_flag="--country ${country}"
    fi
    cat <<SLURM | sbatch
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=ivm
#SBATCH --mem-per-cpu=${mem}
#SBATCH --output=${LOG_DIR}/out_${name}
#SBATCH --error=${LOG_DIR}/err_${name}
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
SKIP_FLAG="--skip-existing"
[ "${hazard}" = "coastal" ] && SKIP_FLAG="--no-skip-existing"
uv run python ${SCRIPT} \\
    --assets ${asset} --hazards ${hazard} \${SKIP_FLAG} --workers 1 ${country_flag}
SLURM
}

# ── River flood ──────────────────────────────────────────────────────────────
submit airports  river
submit ports     river  20000
submit iww       river
submit rail      river  10000  BEL
submit roads     river  10000  BEL

# ── Coastal flood ────────────────────────────────────────────────────────────
submit airports  coastal
submit ports     coastal 20000
submit iww       coastal
submit rail      coastal 10000 BEL
submit roads     coastal 10000 BEL

# ── Windstorm ────────────────────────────────────────────────────────────────
submit airports  windstorm
submit ports     windstorm 20000
submit iww       windstorm
submit rail      windstorm 10000 BEL
submit roads     windstorm 10000 BEL

# ── Earthquake ───────────────────────────────────────────────────────────────
submit airports  earthquake
submit ports     earthquake 20000
submit iww       earthquake
submit rail      earthquake 10000 BEL
submit roads     earthquake 10000 BEL
