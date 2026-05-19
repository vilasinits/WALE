#!/usr/bin/env bash
# Sequential lmax sweep at theta=20 arcmin: lmax in {700, 1000, 1300}.
# Each lmax does 4 NPE jobs: {sim-train, theory-train} x {sim-fid, theory-fid}.
# Sequential because parallel runs collide on the GPU (cuSolver internal error).

set -e
cd /mnt/home/tersenov/software/WALE

SIM_NB="data/cls/simulations/sim_doth_cls_bin4_theta20.0_ratio2.0_nobaryons.npz"
SIM_FID="data/cls/simulations/sim_doth_cls_bin4_theta20.0_ratio2.0_fiducial.npz"
THEORY_REALS="data/cls/theory/theory_doth_cls_bin4_realizations_fidcov_theta20.0_ratio2.0.npz"
THEORY_FID="data/cls/theory/theory_doth_cls_bin4_fiducial_theta20.0_ratio2.0_nside512_pixwin.npz"

for LMAX in 700 1000 1300; do
    echo "=============================================="
    echo "lmax = $LMAX"
    echo "=============================================="

    # S1: sim-trained, sim fid
    echo "[S1 lmax=$LMAX] sim-trained, sim fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$SIM_NB" \
        --fiducial-cls-file "$SIM_FID" \
        --lmax "$LMAX" --train 2>&1 | tail -3

    # S2: sim-trained, theory fid
    echo "[S2 lmax=$LMAX] sim-trained, theory fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$SIM_NB" \
        --fiducial-cls-file "$THEORY_FID" \
        --lmax "$LMAX" --train 2>&1 | tail -3

    # T1: theory-trained, sim fid
    echo "[T1 lmax=$LMAX] theory-trained, sim fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$THEORY_REALS" \
        --fiducial-cls-file "$SIM_FID" \
        --lmax "$LMAX" --train 2>&1 | tail -3

    # T2: theory-trained, theory fid
    echo "[T2 lmax=$LMAX] theory-trained, theory fid"
    conda run -n jaxili python scripts/inference/run_npe_inference_cls.py \
        --sim-cls-file "$THEORY_REALS" \
        --fiducial-cls-file "$THEORY_FID" \
        --lmax "$LMAX" --train 2>&1 | tail -3
done

echo "=============================================="
echo "Sweep complete."
echo "=============================================="
