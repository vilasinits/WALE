#!/usr/bin/env bash
# Single-θ BNT-bin-4 L1 NPE sweep with chosen κ-cuts.
# Args: $1=theta (e.g. 30.0); cuts read from env CUTS (whitespace-separated),
# or from the BNT cut-design CSV when no override is given.
set -e
cd /mnt/home/tersenov/software/WALE

THETA="$1"
if [ -z "$THETA" ]; then
    echo "usage: $0 <theta> ; cuts via env CUTS (e.g. 'full -0.001:0.001')"
    exit 2
fi
TINT="${THETA%.*}"

if [ -z "$CUTS" ]; then
    CSV="outputs/l1_cuts_design_bnt.csv"
    if [ -f "$CSV" ]; then
        sym_cuts=$(awk -F, -v t="$THETA" '
            NR == 1 { for (i=1;i<=NF;i++) col[$i]=i; next }
            $col["theta"] == t && $col["usable_sym"] == 1 {
                printf "%s:%s\n", $col["sym_min"], $col["sym_max"]
            }
        ' "$CSV" | sort -u)
        CUTS=("full")
        while IFS= read -r line; do
            [ -n "$line" ] && CUTS+=("$line")
        done <<< "$sym_cuts"
    else
        CUTS=("full")
    fi
else
    # shellcheck disable=SC2206
    CUTS=($CUTS)
fi

SIM_REALS="data/l1_bnt/simulations/sim_doth_l1_bnt4_bin4_realizations_fidcov_theta${THETA}_ratio2.0.npz"
THEORY_REALS="data/l1_bnt/theory/theory_doth_l1_bnt4_bin4_realizations_fidcov_theta${THETA}_ratio2.0.npz"
SIM_FID="data/l1_bnt/simulations/sim_doth_l1_bnt4_bin4_theta${THETA}_ratio2.0_fiducial.npz"

for need in "$SIM_REALS" "$THEORY_REALS" "$SIM_FID"; do
    [ -f "$need" ] || { echo "[err] missing $need"; exit 1; }
done

echo "BNT θ=${THETA}, cuts: ${CUTS[*]}"
for CUT in "${CUTS[@]}"; do
    if [ "$CUT" = "full" ]; then
        AXIS_ARGS=""
        TAG="full"
    else
        KMIN="${CUT%%:*}"
        KMAX="${CUT##*:}"
        AXIS_ARGS="--axis-min ${KMIN} --axis-max ${KMAX}"
        TAG="${CUT}"
    fi
    echo "[BNT θ=${THETA} ${TAG}] sim-trained"
    conda run -n jaxili python scripts/inference/run_npe_inference.py \
        --data-key l1_norms --axis-key kappa_bins \
        --sim-cls-file "$SIM_REALS" --fiducial-cls-file "$SIM_FID" \
        $AXIS_ARGS --train 2>&1 | tail -2
    echo "[BNT θ=${THETA} ${TAG}] theory-trained"
    conda run -n jaxili python scripts/inference/run_npe_inference.py \
        --data-key l1_norms --axis-key kappa_bins \
        --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$SIM_FID" \
        $AXIS_ARGS --train 2>&1 | tail -2
done

echo "[done] BNT θ=${THETA} sweep complete."
