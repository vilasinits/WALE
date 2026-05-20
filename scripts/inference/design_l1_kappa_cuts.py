#!/usr/bin/env python3
"""
Data-driven κ-cut design for the L1 NPE sweep.

Loads the theory L1 NPZ, the sim L1 NPZ, and the fiducial covariance NPZ
(saved by `scripts/diagnostics/compare_theory_vs_sim_l1.py`). Per κ bin,
computes

    R(k) = median_cosmos | <L1_th> - <L1_sim> |(k) / sigma_fid(k)

where the mean over cosmos is taken across the 7 perms for each unique
`cosmo_id = selected_indices // 7`. This R(k) is the "typical residual in
fiducial-1σ units" — the quantity the NPE actually has to absorb when reading
the theory training set against the sim fiducial observation.

For each threshold T in {1, 2, 3}, walks **outward from κ=0** and reports:
- the symmetric inner range [-κ*, +κ*] such that all bins in [-κ*, +κ*] satisfy
  R(k) < T;
- the (potentially asymmetric) contiguous inner range [κ_-, κ_+] containing
  κ=0 such that every bin inside satisfies R(k) < T.

Outputs a CSV row per (theta, threshold) to `outputs/l1_cuts_design.csv`,
prints ready-to-paste sweep tags, and saves a residual+cut diagnostic plot to
`outputs/plots/l1/theta<T>/diagnostics/kappa_cut_design_theta<T>.pdf`.

Usage:
    conda run -n wale python scripts/inference/design_l1_kappa_cuts.py \
        --sim-npz    data/l1/simulations/sim_doth_l1_bin4_theta40.0_ratio2.0_nobaryons.npz \
        --theory-npz data/l1/theory/theory_doth_l1_bin4_theta40.0_ratio2.0_simbin.npz \
        --fid-cov-npz data/l1/simulations/fid_cov_theta40.npz \
        --thresholds 1 2 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-npz", type=Path, required=True)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--fid-cov-npz", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                   help="R(k) thresholds (default: 1, 2, 3).")
    p.add_argument("--csv", type=Path,
                   default=REPO_ROOT / "outputs" / "l1_cuts_design.csv")
    p.add_argument("--plot-outdir", type=Path, default=None)
    p.add_argument("--min-bins", type=int, default=5,
                   help="Refuse to suggest cuts with fewer than this many bins.")
    return p.parse_args()


def per_cosmo_mean(sel_idx: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Mean L1 across 7-perm group for each unique cosmo_id = sel_idx // 7."""
    cosmo_ids = sel_idx // 7
    uniq = np.unique(cosmo_ids)
    out = np.zeros((uniq.size, data.shape[1]))
    for j, cid in enumerate(uniq):
        out[j] = data[cosmo_ids == cid].mean(axis=0)
    return out


def find_inner_range(kappa: np.ndarray, R: np.ndarray, threshold: float) -> tuple[float, float, int]:
    """Walk outward from κ=0; return (κ_-, κ_+, n_bins) of the largest
    contiguous range containing the central bin where R < threshold."""
    if R.min() > threshold:
        return float("nan"), float("nan"), 0
    center = int(np.argmin(np.abs(kappa)))
    if R[center] >= threshold:
        # central bin already fails — no inner range
        return float("nan"), float("nan"), 0
    lo = center
    while lo > 0 and R[lo - 1] < threshold:
        lo -= 1
    hi = center
    while hi < R.size - 1 and R[hi + 1] < threshold:
        hi += 1
    return float(kappa[lo]), float(kappa[hi]), int(hi - lo + 1)


def symmetric_inner_range(kappa: np.ndarray, R: np.ndarray, threshold: float) -> tuple[float, float, int]:
    """Largest symmetric range [-κ*, κ*] containing κ=0 with all bins below
    threshold. Equals min(|κ_-|, |κ_+|) of `find_inner_range`."""
    km, kp, _ = find_inner_range(kappa, R, threshold)
    if not np.isfinite(km):
        return float("nan"), float("nan"), 0
    half = min(abs(km), abs(kp))
    inside = (kappa >= -half) & (kappa <= +half)
    return -half, +half, int(inside.sum())


def fmt_cut(km: float, kp: float) -> str:
    if not np.isfinite(km):
        return "FAIL"
    return f"{km:+.4f}:{kp:+.4f}"


def main() -> None:
    args = parse_args()

    sim = dict(np.load(args.sim_npz, allow_pickle=True))
    th = dict(np.load(args.theory_npz, allow_pickle=True))
    fc = dict(np.load(args.fid_cov_npz, allow_pickle=True))

    for name, ref, x in [("theory", sim["kappa_bins"], th["kappa_bins"]),
                          ("fid_cov", sim["kappa_bins"], fc["kappa_bins"])]:
        if not np.allclose(ref, x):
            raise ValueError(f"kappa_bins mismatch between sim and {name}.")
    kappa = sim["kappa_bins"]
    theta = float(sim["theta"])
    sigma_fid = fc["sigma_diag"]
    # Bins with sigma_fid == 0 are outside the populated κ range (P(κ) = 0
    # everywhere in the 200 fid realizations). Mark them as "off-range" — the
    # cut walker treats off-range bins as failures so the suggested ranges
    # never include them.
    off_range = sigma_fid <= 0

    th_mean = per_cosmo_mean(th["selected_indices"], th["l1_norms"])
    sim_mean = per_cosmo_mean(sim["selected_indices"], sim["l1_norms"])
    if th_mean.shape != sim_mean.shape:
        raise ValueError(f"per-cosmo theory shape {th_mean.shape} != sim {sim_mean.shape}")

    safe_sigma = np.where(off_range, np.inf, sigma_fid)
    abs_resid = np.abs(th_mean - sim_mean) / safe_sigma[None, :]   # (n_cosmo, nbins)
    R_median = np.median(abs_resid, axis=0)
    R_p84 = np.quantile(abs_resid, 0.84, axis=0)
    R_max = np.max(abs_resid, axis=0)
    # Force off-range bins to look like failures so the walker doesn't extend through them.
    R_median[off_range] = np.inf
    R_p84[off_range] = np.inf
    R_max[off_range] = np.inf

    print(f"theta = {theta}")
    print(f"central R(0) = median {R_median[np.argmin(np.abs(kappa))]:.2f}, "
          f"p84 {R_p84[np.argmin(np.abs(kappa))]:.2f}, "
          f"max {R_max[np.argmin(np.abs(kappa))]:.2f}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.csv.exists()
    csv_rows = []
    sweep_lines = []
    for thr in args.thresholds:
        sm, sp, n_sym = symmetric_inner_range(kappa, R_median, thr)
        am, ap, n_asym = find_inner_range(kappa, R_median, thr)
        usable_sym = n_sym >= args.min_bins
        usable_asym = n_asym >= args.min_bins
        csv_rows.append(dict(theta=theta, threshold=thr,
                             sym_min=sm, sym_max=sp, n_sym=n_sym,
                             asym_min=am, asym_max=ap, n_asym=n_asym,
                             usable_sym=int(usable_sym), usable_asym=int(usable_asym)))
        print(f"  T={thr:.1f}σ: sym {fmt_cut(sm, sp)}  ({n_sym} bins, "
              f"{'OK' if usable_sym else 'too narrow'}); "
              f"asym {fmt_cut(am, ap)}  ({n_asym} bins, "
              f"{'OK' if usable_asym else 'too narrow'})")
        if usable_sym:
            sweep_lines.append(f"{sm:.4f}:{sp:.4f}")
        if usable_asym and (abs(am) != abs(ap) or not usable_sym):
            # Add asymmetric cut only if it differs materially from the symmetric one.
            sweep_lines.append(f"{am:.4f}:{ap:.4f}")

    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"Appended {len(csv_rows)} rows to {args.csv}")

    # Ready-to-paste sweep tags
    if sweep_lines:
        print("\nSuggested sweep cuts (for `run_l1_intermediate_theta_sweep.sh`):")
        print("  full")
        for c in sweep_lines:
            print(f"  {c}")

    # Diagnostic plot
    if args.plot_outdir is None:
        args.plot_outdir = REPO_ROOT / "outputs" / "plots" / "l1" / f"theta{theta:.0f}" / "diagnostics"
    args.plot_outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(kappa, np.where(np.isfinite(R_median), R_median, np.nan),
            label="median |Δ| / σ_fid", color="C0")
    ax.fill_between(kappa,
                    np.where(np.isfinite(R_median), R_median, np.nan),
                    np.where(np.isfinite(R_p84), R_p84, np.nan),
                    alpha=0.2, color="C0", label="median → p84")
    ax.plot(kappa, np.where(np.isfinite(R_max), R_max, np.nan),
            color="C0", ls=":", alpha=0.5, label="max")
    for thr in args.thresholds:
        ax.axhline(thr, color="grey", ls="--", lw=0.7)
        ax.text(kappa.max(), thr, f" T={thr:.0f}", color="grey", va="center", fontsize=8)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$|L_1^{\rm th}-L_1^{\rm sim}| / \sigma_{\rm fid}$")
    ax.set_yscale("log")
    ax.set_title(f"L1 residual per-cosmo distribution, θ={theta:.0f}'")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_pdf = args.plot_outdir / f"kappa_cut_design_theta{theta:.0f}.pdf"
    plt.savefig(out_pdf, transparent=True); plt.close()
    print(f"Saved residual plot: {out_pdf}")


if __name__ == "__main__":
    main()
