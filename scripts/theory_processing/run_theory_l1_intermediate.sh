#!/usr/bin/env bash
# Compute LDT theory L1 at θ ∈ {40, 50, 60} arcmin, on the κ grid from the
# packed sim NPZs. Sequential because each invocation uses 50 cpu workers.
set -e
cd /mnt/home/tersenov/software/WALE

mkdir -p logs/l1_intermediate

THETAS=(40.0 50.0 60.0)

for THETA in "${THETAS[@]}"; do
    SIM_NPZ="data/l1/simulations/sim_doth_l1_bin4_theta${THETA}_ratio2.0_nobaryons.npz"
    OUT_NPZ="data/l1/theory/theory_doth_l1_bin4_theta${THETA}_ratio2.0_simbin.npz"
    LOG="logs/l1_intermediate/theory_theta${THETA%.*}.log"

    echo "[theory θ=${THETA}] → ${OUT_NPZ}"
    conda run -n wale python scripts/theory_processing/compute_theory_l1_halofit.py \
        --sim-npz "$SIM_NPZ" \
        --nz-file data/nz/nz_stage3_4_GRID.txt \
        --output "$OUT_NPZ" --overwrite \
        --n-jobs 50 --backend loky \
        > "$LOG" 2>&1
    tail -5 "$LOG"
done

echo "[done] Intermediate-θ theory L1 complete."
