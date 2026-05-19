#!/usr/bin/env bash
# Sequential lmax sweep across (theta, lmax) cells for the safe-regime mapping.
# Each cell runs 4 NPE jobs: S1, S2, T1, T2 (sim/theory training x sim/theory fid).
# Sequential because parallel NPE jobs collide on the GPU.

set -e
cd /mnt/home/tersenov/software/WALE

# Each entry: "theta lmax_list" where lmax can be "full" for no cap.
declare -a CELLS=(
    "10.0  1000 1300 1500 full"
    "20.0  400 600"
    "30.0  300 400 500 full"
)

for cell in "${CELLS[@]}"; do
    read -ra parts <<< "$cell"
    THETA="${parts[0]}"
    LMAX_LIST=("${parts[@]:1}")

    THETA_TAG=$(printf "%s_ratio2.0" "theta${THETA}")
    SIM_NB="data/cls/simulations/sim_doth_cls_bin4_${THETA_TAG}_nobaryons.npz"
    SIM_FID="data/cls/simulations/sim_doth_cls_bin4_${THETA_TAG}_fiducial.npz"
    THEORY_REALS="data/cls/theory/theory_doth_cls_bin4_realizations_fidcov_${THETA_TAG}.npz"
    THEORY_FID="data/cls/theory/theory_doth_cls_bin4_fiducial_${THETA_TAG}_nside512_pixwin.npz"

    for LMAX in "${LMAX_LIST[@]}"; do
        echo "=================================================="
        echo "theta=${THETA}  lmax=${LMAX}"
        echo "=================================================="

        # Build the lmax flag — empty string means full ell
        if [ "$LMAX" = "full" ]; then
            LMAX_FLAG=""
        else
            LMAX_FLAG="--lmax $LMAX"
        fi

        echo "[S1 theta=${THETA} lmax=${LMAX}] sim-train, sim-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$SIM_NB" --fiducial-cls-file "$SIM_FID" \
            $LMAX_FLAG --train 2>&1 | tail -3

        echo "[S2 theta=${THETA} lmax=${LMAX}] sim-train, theory-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$SIM_NB" --fiducial-cls-file "$THEORY_FID" \
            $LMAX_FLAG --train 2>&1 | tail -3

        echo "[T1 theta=${THETA} lmax=${LMAX}] theory-train, sim-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$SIM_FID" \
            $LMAX_FLAG --train 2>&1 | tail -3

        echo "[T2 theta=${THETA} lmax=${LMAX}] theory-train, theory-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$THEORY_FID" \
            $LMAX_FLAG --train 2>&1 | tail -3
    done
done

echo "=================================================="
echo "lmax sweep complete."
echo "=================================================="
