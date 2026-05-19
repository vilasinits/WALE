#!/usr/bin/env bash
# Sequential transfer-threshold sweep at theta=20, no lmax cut.
# thresholds: 0.1 (main lobe only), 0.05 (main + part of secondary), 0.01 (mostly all secondary).
# Each threshold: 4 NPE jobs (S1, S2, T1, T2).

set -e
cd /mnt/home/tersenov/software/WALE

SIM_NB="data/cls/simulations/sim_doth_cls_bin4_theta20.0_ratio2.0_nobaryons.npz"
SIM_FID="data/cls/simulations/sim_doth_cls_bin4_theta20.0_ratio2.0_fiducial.npz"
THEORY_REALS="data/cls/theory/theory_doth_cls_bin4_realizations_fidcov_theta20.0_ratio2.0.npz"
THEORY_FID="data/cls/theory/theory_doth_cls_bin4_fiducial_theta20.0_ratio2.0_nside512_pixwin.npz"
# Transfer file must have the 'doth_transfer' field; we use the theory simbin file (it has it).
TRANSFER_FILE="data/cls/theory/theory_doth_cls_bin4_simbin_theta20.0_ratio2.0_nside512_pixwin.npz"

for THRESH in 0.1 0.05 0.01; do
    echo "=============================================="
    echo "transfer threshold = $THRESH"
    echo "=============================================="

    echo "[S1 tf=$THRESH] sim-trained, sim fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$SIM_NB" --fiducial-cls-file "$SIM_FID" \
        --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
        --train 2>&1 | tail -3

    echo "[S2 tf=$THRESH] sim-trained, theory fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$SIM_NB" --fiducial-cls-file "$THEORY_FID" \
        --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
        --train 2>&1 | tail -3

    echo "[T1 tf=$THRESH] theory-trained, sim fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$SIM_FID" \
        --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
        --train 2>&1 | tail -3

    echo "[T2 tf=$THRESH] theory-trained, theory fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$THEORY_REALS" --fiducial-cls-file "$THEORY_FID" \
        --transfer-file "$TRANSFER_FILE" --transfer-threshold "$THRESH" \
        --train 2>&1 | tail -3
done

echo "=============================================="
echo "Transfer sweep complete."
echo "=============================================="
