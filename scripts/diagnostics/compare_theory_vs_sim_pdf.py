#!/usr/bin/env python3
"""
Side-by-side overview of the theory and sim PDFs (P(kappa), no |kappa|
weighting).

NOTE: do NOT use this script to plot fractional/sigma residuals — those are
mathematically identical to the L1 versions because the |kappa| weighting
cancels in any ratio. For shape diagnostics, see
`compare_theory_vs_sim_moments.py` (per-cosmology variance / skewness /
kurtosis comparison).

The sim PDF is recovered from L1 = pdf*|kappa| as pdf = L1 / |kappa| (kappa=0
bin handled with NaN).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def per_cosmo_stats(sel_idx: np.ndarray, data: np.ndarray):
    cids = sel_idx // 7
    uniq = np.unique(cids)
    means = np.array([data[cids == c].mean(0) for c in uniq])
    stds = np.array([data[cids == c].std(0, ddof=1) for c in uniq])
    first = np.array([np.where(cids == c)[0][0] for c in uniq])
    return uniq, means, stds, first


def recover_pdf(l1: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """pdf = L1 / |kappa|, with kappa=0 bins masked."""
    out = np.zeros_like(l1)
    mask = np.abs(kappa) > 1e-30
    out[:, mask] = l1[:, mask] / np.abs(kappa)[None, mask]
    out[:, ~mask] = np.nan
    return out


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
    p = argparse.ArgumentParser()
    p.add_argument("--sim-npz", type=Path, required=True)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default: outputs/plots/l1/theta<T>/diagnostics).")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    sim = dict(np.load(args.sim_npz, allow_pickle=True))
    th = dict(np.load(args.theory_npz, allow_pickle=True))
    if not np.allclose(sim["kappa_bins"], th["kappa_bins"]):
        raise ValueError("sim and theory kappa_bins differ")
    kappa = sim["kappa_bins"]
    theta = float(sim["theta"])

    sim_pdf = recover_pdf(np.asarray(sim["l1_norms"], dtype=float), kappa)
    if "pdf_theory" not in th:
        raise ValueError("theory NPZ has no 'pdf_theory' field.")
    th_pdf = np.asarray(th["pdf_theory"], dtype=float)

    cids_sim, sim_mean, _, first_sim = per_cosmo_stats(sim["selected_indices"], sim_pdf)
    cids_th, th_mean, _, _ = per_cosmo_stats(th["selected_indices"], th_pdf)
    if not np.array_equal(cids_sim, cids_th):
        raise ValueError("cosmo_id sets differ")

    color_vals = sim["params"][first_sim, 0]   # Ωm

    tag = args.tag or f"theta{theta:.0f}"
    if args.outdir is None:
        args.outdir = REPO_ROOT / "outputs" / "plots" / "l1" / f"theta{theta:.0f}" / "diagnostics"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Side-by-side overview of theory PDF (left) vs sim mean PDF (right)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    for ax, data, title in zip(axes, [th_mean, sim_mean],
                               ["Theory PDF (per cosmology)",
                                "Sim mean PDF (per cosmology)"]):
        for i in range(data.shape[0]):
            ax.plot(kappa, data[i], color=cmap(norm(color_vals[i])), lw=0.6, alpha=0.6)
        ax.set_xlabel(r"$\kappa$")
        ax.set_title(f"{title}, θ={theta}'")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$P(\kappa)$")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        label=r"$\Omega_m$", shrink=0.9, pad=0.02)
    out_overview = args.outdir / f"pdf_datavectors_overview_{tag}.pdf"
    plt.savefig(out_overview, transparent=True); plt.close()
    print(f"Saved {out_overview}")


if __name__ == "__main__":
    main()
