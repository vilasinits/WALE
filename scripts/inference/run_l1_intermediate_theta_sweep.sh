#!/usr/bin/env bash
# Intermediate-θ L1 NPE sweep. For each θ ∈ {40, 50, 60} runs two NPE jobs
# (sim-trained vs theory-trained) per κ-cut. Cuts are read from a CSV produced
# by `scripts/inference/design_l1_kappa_cuts.py`, but can also be passed via
# env vars (CUTS_THETA40="full -0.01:0.01 ..." etc.) to override.
#
# Sequential — GPU 1 only.

set -e
cd /mnt/home/tersenov/software/WALE

THETAS=(40.0 50.0 60.0)
CSV="outputs/l1_cuts_design.csv"

# Default cuts per θ: filled in from the cut-design CSV at runtime. If the env
# var is set, it overrides; otherwise we parse the CSV.
default_cuts_for_theta() {
    local theta=$1
    # Take symmetric cuts at the {1, 2, 3}σ thresholds, plus "full".
    if [ -f "$CSV" ]; then
        # Print "kmin:kmax" rows for this theta where usable_sym=1.
        awk -F, -v t="$theta" '
            NR == 1 { for (i=1;i<=NF;i++) col[$i]=i; next }
            $col["theta"] == t && $col["usable_sym"] == 1 {
                printf "%s:%s\n", $col["sym_min"], $col["sym_max"]
            }
        ' "$CSV" | sort -u
    fi
}

run_pair() {
    local theta=$1
    local cut=$2
    local sim_reals="data/l1/simulations/sim_doth_l1_bin4_realizations_fidcov_theta${theta}_ratio2.0.npz"
    local theory_reals="data/l1/theory/theory_doth_l1_bin4_realizations_fidcov_theta${theta}_ratio2.0.npz"
    local sim_fid="data/l1/simulations/sim_doth_l1_bin4_theta${theta}_ratio2.0_fiducial.npz"

    local args=""
    local tag=""
    if [ "$cut" = "full" ]; then
        args=""
        tag="full"
    else
        local kmin="${cut%%:*}"
        local kmax="${cut##*:}"
        args="--axis-min ${kmin} --axis-max ${kmax}"
        tag="amin${kmin}_amax${kmax}"
    fi
    echo "=================================================="
    echo "[θ=${theta} ${tag}]"
    echo "  sim-trained ..."
    conda run -n jaxili python scripts/inference/run_npe_inference.py \
        --data-key l1_norms --axis-key kappa_bins \
        --sim-cls-file "$sim_reals" --fiducial-cls-file "$sim_fid" \
        $args --train 2>&1 | tail -2
    echo "  theory-trained ..."
    conda run -n jaxili python scripts/inference/run_npe_inference.py \
        --data-key l1_norms --axis-key kappa_bins \
        --sim-cls-file "$theory_reals" --fiducial-cls-file "$sim_fid" \
        $args --train 2>&1 | tail -2
}

for THETA in "${THETAS[@]}"; do
    TINT="${THETA%.*}"
    CUT_ENV_VAR="CUTS_THETA${TINT}"
    CUTS_STR="${!CUT_ENV_VAR}"
    if [ -n "$CUTS_STR" ]; then
        # Whitespace-separated cuts from env var
        read -r -a CUTS <<< "$CUTS_STR"
    else
        # Default: full + the symmetric cuts at all thresholds in the CSV.
        CUTS=("full")
        while IFS= read -r line; do
            [ -n "$line" ] && CUTS+=("$line")
        done < <(default_cuts_for_theta "${THETA}")
    fi
    if [ "${#CUTS[@]}" -le 1 ] && [ -z "$CUTS_STR" ]; then
        echo "[warn] No CSV cuts for θ=${THETA} — running only 'full'. Run design_l1_kappa_cuts.py first."
    fi
    echo "θ=${THETA} cuts: ${CUTS[*]}"
    for CUT in "${CUTS[@]}"; do
        run_pair "${THETA}" "${CUT}"
    done
done

echo "=================================================="
echo "Intermediate-θ L1 NPE sweep complete."
