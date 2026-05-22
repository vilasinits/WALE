#!/usr/bin/env python3
"""
Three diagnostic plots comparing the theory ℓ₁-norm to the simulated ℓ₁-norm
on the cosmoGRID grid.

Inputs (all packed NPZ files):
    --sim-npz           per-perm sim L1 (e.g. 1299 rows)
    --theory-npz        per-perm theory L1 (same row count as sim)
    --fiducial-sim-npz  200-realization fiducial sim L1 (for fid covariance)

Per unique cosmology (selected_indices // 7), means are computed across the 7
permutations of both sim and theory. Then three panels are produced:

  Panel 1: fractional difference (L1_th − L1_sim) / L1_sim
           horizontal dashed lines at ±2 %, ±5 %.
  Panel 2: (L1_th − L1_sim) / σ_sim(κ; per-cosmo)
           σ from the std over the 7 perms (computed once per cosmo).
           horizontal dashed lines at ±2σ, ±3σ.
  Panel 3: (L1_th − L1_sim) / σ_fid(κ)
           σ_fid from the diagonal of the 200-realization fiducial covariance.
           horizontal dashed lines at ±2σ, ±3σ.

Each cosmology gets one line; the lines are coloured by Ω_m (default).
Outputs PDF figures to outputs/plots/l1_theory_vs_sim_{frac,sigma_percosmo,sigma_fidcov}_thetaT.pdf
and a combined 3-panel summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-npz", type=Path, required=True)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--fiducial-sim-npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default: outputs/plots/l1/theta<T>/diagnostics).")
    p.add_argument("--tag", type=str, default=None,
                   help="Filename tag (default: derived from theta).")
    p.add_argument("--color-by", type=int, default=0,
                   help="Parameter index to colour cosmologies by (default 0: Omega_m).")
    p.add_argument("--ymax-frac", type=float, default=0.30)
    p.add_argument("--ymax-sigma", type=float, default=6.0)
    p.add_argument("--cov-out", type=Path, default=None,
                   help="Override path for the saved fid-cov NPZ "
                        "(default: data/l1/simulations/fid_cov_{tag}.npz).")
    return p.parse_args()


def load_packed(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        out = {k: d[k] for k in d.files}
    return out


def per_cosmo_stats(sel_idx: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group rows by cosmo_id = sel_idx // 7. Return (cosmo_ids, mean, std, first_row_idx)."""
    cosmo_ids = sel_idx // 7
    uniq = np.unique(cosmo_ids)
    means = np.zeros((uniq.size, data.shape[1]))
    stds = np.zeros((uniq.size, data.shape[1]))
    first = np.zeros(uniq.size, dtype=int)
    for j, cid in enumerate(uniq):
        rows = np.where(cosmo_ids == cid)[0]
        means[j] = data[rows].mean(axis=0)
        stds[j] = data[rows].std(axis=0, ddof=1) if rows.size > 1 else np.zeros(data.shape[1])
        first[j] = rows[0]
    return uniq, means, stds, first


def plot_panel(ax, kappa, lines, color_vals, ymax, ylines, ylabel, title):
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    for i in range(lines.shape[0]):
        ax.plot(kappa, lines[i], color=cmap(norm(color_vals[i])), lw=0.5, alpha=0.55)
    for y in ylines:
        ax.axhline(y, color="black", ls="--", lw=0.7)
        ax.axhline(-y, color="black", ls="--", lw=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-ymax, ymax)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    return cmap, norm


def main() -> None:
    args = parse_args()
    # Defer outdir resolution to after we read theta from the sim file.

    sim = load_packed(args.sim_npz)
    th = load_packed(args.theory_npz)
    fid = load_packed(args.fiducial_sim_npz)

    if not np.allclose(sim["kappa_bins"], th["kappa_bins"]):
        raise ValueError("Sim and theory kappa_bins differ.")
    if not np.allclose(sim["kappa_bins"], fid["kappa_bins"]):
        raise ValueError("Sim and fiducial kappa_bins differ.")
    kappa = sim["kappa_bins"]
    theta = float(sim["theta"])
    if args.outdir is None:
        args.outdir = REPO_ROOT / "outputs" / "plots" / "l1" / f"theta{theta:.0f}" / "diagnostics"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Per-cosmo means and stds
    cids_sim, sim_mean, sim_std, first_sim = per_cosmo_stats(
        sim["selected_indices"], sim["l1_norms"]
    )
    cids_th, th_mean, th_std, first_th = per_cosmo_stats(
        th["selected_indices"], th["l1_norms"]
    )
    if not np.array_equal(cids_sim, cids_th):
        raise ValueError("Sim and theory cosmo_id sets differ.")
    cids = cids_sim
    print(f"Compared {cids.size} unique cosmologies at theta={theta}.")

    # Colouring by a parameter
    color_vals = sim["params"][first_sim, args.color_by]

    # Fiducial covariance diagonal
    fid_l1 = fid["l1_norms"]
    sig_fid = fid_l1.std(axis=0, ddof=1)        # (nbins,)
    cov_fid = np.cov(fid_l1.T)                   # (nbins, nbins) -- saved for downstream use

    # Differences
    diff = th_mean - sim_mean                                    # (n_cosmo, nbins)

    # Panel 1: fractional
    safe_denom = np.where(np.abs(sim_mean) > 1e-30, sim_mean, np.nan)
    frac = diff / safe_denom                                     # (n_cosmo, nbins)

    # Panel 2: sigma per-cosmo (from 7-perm std)
    safe_sig_pc = np.where(sim_std > 0, sim_std, np.nan)
    sig_pc = diff / safe_sig_pc                                  # (n_cosmo, nbins)

    # Panel 3: sigma from fiducial cov diagonal
    safe_sig_fid = np.where(sig_fid > 0, sig_fid, np.nan)[None, :]
    sig_fid_diff = diff / safe_sig_fid                           # (n_cosmo, nbins)

    tag = args.tag or f"theta{theta:.0f}"

    # --- Combined 3-panel figure (the per-panel singles were dropped as redundant) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, lines, ymax, ylines, ylabel, title in [
        (axes[0], frac,         args.ymax_frac,  [0.02, 0.05],
         r"$(L_1^{\rm th} - L_1^{\rm sim}) / L_1^{\rm sim}$",            "Fractional"),
        (axes[1], sig_pc,       args.ymax_sigma, [2, 3],
         r"$(L_1^{\rm th} - L_1^{\rm sim}) / \sigma_{\rm sim}^{\rm pc}$", "σ per-cosmo"),
        (axes[2], sig_fid_diff, args.ymax_sigma, [2, 3],
         r"$(L_1^{\rm th} - L_1^{\rm sim}) / \sigma_{\rm fid}$",         "σ fiducial cov"),
    ]:
        plot_panel(ax, kappa, lines, color_vals, ymax, ylines, ylabel, f"{title}, θ={theta}'")
    out_combined = args.outdir / f"l1_theory_vs_sim_combined_{tag}.pdf"
    plt.savefig(out_combined, transparent=True); plt.close()
    print(f"Saved {out_combined}")

    # Optional: save the fiducial covariance for downstream consumers
    cov_npz = args.cov_out or (
        REPO_ROOT / "data" / "l1" / "simulations" / f"fid_cov_{tag}.npz"
    )
    cov_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cov_npz, kappa_bins=kappa, cov=cov_fid, sigma_diag=sig_fid, theta=theta)
    print(f"Saved fiducial cov: {cov_npz}")


if __name__ == "__main__":
    main()
