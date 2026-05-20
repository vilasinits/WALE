#!/usr/bin/env bash
# Sequential L1 NPE sweep at θ=30: each kappa-cut runs two NPE jobs
# (sim-trained vs theory-trained). Fiducial observation is the mean of the 200
# sim fiducial L1 realizations.
#
# GPU 1 only (sequential by default; parallel would oversubscribe).

set -e
cd /mnt/home/tersenov/software/WALE

SIM_REALS="data/l1/simulations/sim_doth_l1_bin4_realizations_fidcov_theta30.0_ratio2.0.npz"
THEORY_REALS="data/l1/theory/theory_doth_l1_bin4_realizations_fidcov_theta30.0_ratio2.0.npz"
SIM_FID="data/l1/simulations/sim_doth_l1_bin4_theta30.0_ratio2.0_fiducial.npz"

# κ-cut grid (positional override allowed). Format: kmin:kmax or "full".
CUTS=(
    "full"
    "-0.010:0.010"
    "-0.005:0.005"
    "-0.012:0.012"
)
if [ "$#" -gt 0 ]; then
    CUTS=("$@")
fi

for CUT in "${CUTS[@]}"; do
    echo "=================================================="
    if [ "$CUT" = "full" ]; then
        AXIS_ARGS=""
        TAG="full"
    else
        KMIN="${CUT%%:*}"
        KMAX="${CUT##*:}"
        AXIS_ARGS="--axis-min ${KMIN} --axis-max ${KMAX}"
        TAG="amin${KMIN}_amax${KMAX}"
    fi
    echo "[sim-trained, ${TAG}]"
    conda run -n jaxili python scripts/inference/run_npe_inference.py \
        --data-key l1_norms --axis-key kappa_bins \
        --sim-cls-file "$SIM_REALS" --fiducial-cls-file "$SIM_FID" \
        $AXIS_ARGS --train 2>&1 | tail -3

    echo "[theory-trained, ${TAG}]"
    conda run -n jaxili python scripts/inference/run_npe_inference.py \
        --data-key l1_norms --axis-key kappa_bins \
        --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$SIM_FID" \
        $AXIS_ARGS --train 2>&1 | tail -3
done

echo "=================================================="
echo "L1 sweep complete."
echo "=================================================="
