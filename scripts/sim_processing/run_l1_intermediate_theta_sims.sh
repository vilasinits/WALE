#!/usr/bin/env bash
# Generate sim L1 (halofit grid + fiducial) at θ ∈ {40, 50, 60} arcmin.
# Sequential because each invocation uses 50 workers (multiprocessing.Pool).
#
# Note: pass `.npy` in --combined-output so os.path.splitext picks the right ext.
set -e
cd /mnt/home/tersenov/software/WALE

mkdir -p logs/l1_intermediate

run_sim() {
    local theta=$1
    local kmax=$2
    local mode=$3   # "halofit" or "fiducial"
    local stem="data/l1/simulations/raw_sim_doth_l1_bin4_theta${theta}_ratio2.0_${mode}.npy"
    local log="logs/l1_intermediate/sim_theta${theta%.*}_${mode}.log"
    local extra=""
    if [ "$mode" = "fiducial" ]; then extra="--fiducial"; fi
    local kmin=$(python -c "print(-${kmax})")

    echo "[sim ${mode} θ=${theta}] kappa=[${kmin}, ${kmax}] → ${stem}"
    conda run -n wale python scripts/sim_processing/l1_norm_processing_halofit.py \
        --bin-number 4 --no-noise --theta "${theta}" --theta-ratio 2.0 \
        --nbins 200 --kappa-min "${kmin}" --kappa-max "${kmax}" \
        --num-workers 50 --save-combined --force-overwrite ${extra} \
        --combined-output "${stem}" \
        > "${log}" 2>&1
    tail -3 "${log}"
}

# θ=40 halofit already ran (filenames were renamed to match convention).
# Remaining work: θ=40 fiducial, θ=50 halofit+fiducial, θ=60 halofit+fiducial.

run_sim 40.0 0.014 fiducial
run_sim 50.0 0.013 halofit
run_sim 50.0 0.013 fiducial
run_sim 60.0 0.012 halofit
run_sim 60.0 0.012 fiducial

echo "[done] All intermediate-θ sim L1 jobs complete."
