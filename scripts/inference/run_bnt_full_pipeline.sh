#!/usr/bin/env bash
# Run all post-theory steps for one BNT theta: diagnostics, cut design,
# pseudo-realizations, NPE sweep, contour plots.
#
# Usage:  bash scripts/inference/run_bnt_full_pipeline.sh <theta>
# Example: bash scripts/inference/run_bnt_full_pipeline.sh 20.0

set -e
cd /mnt/home/tersenov/software/WALE

THETA="$1"
if [ -z "$THETA" ]; then
    echo "usage: $0 <theta>"
    exit 2
fi
TINT="${THETA%.*}"

SIM_NPZ="data/l1_bnt/simulations/sim_doth_l1_bnt4_bin4_theta${THETA}_ratio2.0_nobaryons.npz"
SIM_FID="data/l1_bnt/simulations/sim_doth_l1_bnt4_bin4_theta${THETA}_ratio2.0_fiducial.npz"
TH_NPZ="data/l1_bnt/theory/theory_doth_l1_bnt4_bin4_theta${THETA}_ratio2.0_simbin.npz"
FID_COV="data/l1_bnt/simulations/fid_cov_bnt4_theta${TINT}.npz"
SIM_REALS="data/l1_bnt/simulations/sim_doth_l1_bnt4_bin4_realizations_fidcov_theta${THETA}_ratio2.0.npz"
TH_REALS="data/l1_bnt/theory/theory_doth_l1_bnt4_bin4_realizations_fidcov_theta${THETA}_ratio2.0.npz"

DIAG_DIR="outputs/plots/l1_bnt/theta${TINT}/diagnostics"
OVERLAY_DIR="outputs/plots/l1_bnt/theta${TINT}/overlays"
mkdir -p "$DIAG_DIR" "$OVERLAY_DIR"

for f in "$SIM_NPZ" "$SIM_FID" "$TH_NPZ"; do
    [ -f "$f" ] || { echo "[err] missing $f"; exit 1; }
done

echo "=== Step 1: moments diagnostic ==="
conda run -n wale python scripts/diagnostics/compare_theory_vs_sim_moments.py \
    --sim-npz "$SIM_NPZ" --theory-npz "$TH_NPZ" \
    --outdir "$DIAG_DIR" --tag "bnt4_theta${TINT}"

echo
echo "=== Step 2: 3-panel L1 residual + fid_cov ==="
conda run -n wale python scripts/diagnostics/compare_theory_vs_sim_l1.py \
    --sim-npz "$SIM_NPZ" --theory-npz "$TH_NPZ" --fiducial-sim-npz "$SIM_FID" \
    --outdir "$DIAG_DIR" --tag "bnt4_theta${TINT}" \
    --cov-out "$FID_COV"

echo
echo "=== Step 3: recal distribution ==="
conda run -n wale python scripts/diagnostics/compute_recal_distribution.py \
    --theory-npz "$TH_NPZ" --tag "bnt4_theta${TINT}" \
    --outdir "$DIAG_DIR"

echo
echo "=== Step 4: l1 datavectors overview ==="
conda run -n wale python scripts/diagnostics/plot_l1_datavectors_overview.py \
    --sim-npz "$SIM_NPZ" --theory-npz "$TH_NPZ" \
    --outdir "$DIAG_DIR" --tag "bnt4_theta${TINT}"

echo
echo "=== Step 5: single-cosmo debug ==="
conda run -n wale python scripts/diagnostics/plot_l1_single_cosmo_debug.py \
    --sim-npz "$SIM_NPZ" --theory-npz "$TH_NPZ" \
    --out "${DIAG_DIR}/l1_single_cosmo_debug_bnt4_theta${TINT}.pdf"

echo
echo "=== Step 6: data-driven cut design ==="
conda run -n wale python scripts/inference/design_l1_kappa_cuts.py \
    --sim-npz "$SIM_NPZ" --theory-npz "$TH_NPZ" --fid-cov-npz "$FID_COV" \
    --csv outputs/l1_cuts_design_bnt.csv \
    --plot-outdir "$DIAG_DIR"

echo
echo "=== Step 7: pseudo-realizations (sim + theory) ==="
conda run -n wale python scripts/inference/generate_l1_pseudorealizations.py \
    --mean-source "$SIM_NPZ" --cov-source "$SIM_FID" \
    --mean-mode average-perms --n-draws 7 \
    --output "$SIM_REALS" --overwrite
conda run -n wale python scripts/inference/generate_l1_pseudorealizations.py \
    --mean-source "$TH_NPZ" --cov-source "$SIM_FID" \
    --mean-mode as-is --n-draws 7 \
    --output "$TH_REALS" --overwrite

echo
echo "=== Done. NPE sweep is the next step; run e.g.:"
echo "  CUTS='full -0.0003:0.0003 -0.0009:0.0009' bash scripts/inference/run_l1_bnt_theta_npe.sh ${THETA}"
echo "  conda run -n jaxili python scripts/inference/plot_l1_kappa_cut_comparison.py --mode bnt4 --theta-tag theta${TINT} --outdir ${OVERLAY_DIR}"
