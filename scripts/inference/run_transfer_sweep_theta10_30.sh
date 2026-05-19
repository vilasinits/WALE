#!/usr/bin/env bash
# Sequential transfer-threshold sweep at theta in {10, 30}, thresholds {0.1, 0.05, 0.01}.
# 2 theta x 3 threshold x 4 NPE jobs = 24 runs. Sequential (GPU contention).

set -e
cd /mnt/home/tersenov/software/WALE

for THETA in 10.0 30.0; do
    THETA_TAG="theta${THETA}_ratio2.0"
    SIM_NB="data/cls/simulations/sim_doth_cls_bin4_${THETA_TAG}_nobaryons.npz"
    SIM_FID="data/cls/simulations/sim_doth_cls_bin4_${THETA_TAG}_fiducial.npz"
    THEORY_REALS="data/cls/theory/theory_doth_cls_bin4_realizations_fidcov_${THETA_TAG}.npz"
    THEORY_FID="data/cls/theory/theory_doth_cls_bin4_fiducial_${THETA_TAG}_nside512_pixwin.npz"
    TRANSFER_FILE="data/cls/theory/theory_doth_cls_bin4_simbin_${THETA_TAG}_nside512_pixwin.npz"

    for THRESH in 0.1 0.05 0.01; do
        echo "=================================================="
        echo "theta=${THETA}  tf=${THRESH}"
        echo "=================================================="

        echo "[S1 theta=${THETA} tf=${THRESH}] sim-train, sim-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$SIM_NB" --fiducial-cls-file "$SIM_FID" \
            --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
            --train 2>&1 | tail -3

        echo "[S2 theta=${THETA} tf=${THRESH}] sim-train, theory-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$SIM_NB" --fiducial-cls-file "$THEORY_FID" \
            --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
            --train 2>&1 | tail -3

        echo "[T1 theta=${THETA} tf=${THRESH}] theory-train, sim-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$SIM_FID" \
            --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
            --train 2>&1 | tail -3

        echo "[T2 theta=${THETA} tf=${THRESH}] theory-train, theory-fid"
        conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
            --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$THEORY_FID" \
            --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
            --train 2>&1 | tail -3
    done
done

echo "=================================================="
echo "Transfer sweep at theta in {10, 30} complete."
echo "=================================================="
