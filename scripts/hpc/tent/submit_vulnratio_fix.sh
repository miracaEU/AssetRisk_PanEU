#!/bin/bash
# Rerun TENT assets after vuln_ratio normalization fix (risk_integration_tent.py
# compute_ratio_and_exposure_per_rp: ratio now divides by maxdam_per_unit x
# feature_size instead of bare maxdam_per_unit, which was saturating
# vuln_ratio to 0/1 for ports/airports/roads/rail).
#
# EAD columns are unaffected by the fix -- only vuln_ratio_*/exposure_*_RP*
# columns change. is_complete() doesn't check those columns, so a normal
# --skip-existing run would skip everything below; --no-skip-existing forces
# a real recompute.
#
# One job per asset x hazard (matches original submit_tent_all.sh layout)
# so output filenames match exactly and overwrite the existing parquet
# for that asset/hazard, instead of writing a new combined-hazard file.
#
# Scope:
#   ports, airports, iww : full Europe, all hazards
#   roads, rail           : Rhine-Alpine corridor (L) only, all hazards

SCRIPT=/scistor/ivm/eks510/projects/AssetRisk_PanEU/src_tent/run_pipeline_tent.py
LOG_DIR=/scistor/ivm/eks510/MIRACA_RISK/TENT/logs/slurm

mkdir -p "$LOG_DIR"

submit() {
    local asset=$1
    local hazard=$2
    local mem=${3:-10000}
    local corridor=${4:-}
    local name="tent_${asset}_${hazard}_vulnfix"
    local corridor_flag=""
    if [ -n "$corridor" ]; then
        name="${name}_corridor${corridor}"
        corridor_flag="--corridor ${corridor}"
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
uv run python ${SCRIPT} --assets ${asset} --hazards ${hazard} ${corridor_flag} --no-skip-existing --workers 1
SLURM
}

for hazard in river coastal windstorm earthquake; do
    # ── Full Europe (no corridor filter) ────────────────────────────────────
    submit ports     "$hazard" 20000
    submit airports  "$hazard" 10000
    submit iww       "$hazard" 10000

    # ── Rhine-Alpine corridor only (L) ──────────────────────────────────────
    submit roads     "$hazard" 10000 L
    submit rail      "$hazard" 10000 L
done

# ── Energy network (no corridor/country filter; originally run as a single
#    combined-hazard job per asset -- existing files are named
#    TENT_energy_{lines,buses}_coastal_earthquake_river_windstorm.parquet,
#    so rerun must also omit --hazards to match that filename and overwrite) ──
submit_combined() {
    local asset=$1
    local mem=${2:-10000}
    local name="tent_${asset}_vulnfix"
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
uv run python ${SCRIPT} --assets ${asset} --no-skip-existing --workers 1
SLURM
}

submit_combined energy_lines 10000
submit_combined energy_buses 10000
