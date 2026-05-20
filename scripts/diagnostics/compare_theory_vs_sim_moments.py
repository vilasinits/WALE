#!/usr/bin/env python3
"""
Per-cosmology PDF moments: theory vs simulation.

For each unique cosmology compute (mean, variance, skewness, excess kurtosis)
of both theory and sim PDFs, and produce 4 scatter panels: theory_moment vs
sim_moment, colour-coded by Omega_m, with the y=x diagonal for reference.

This is the proper "(B) PDF shape comparison" — replaces the misleading
fractional-residual plot whose |kappa| weighting cancels exactly.

Why this matters: the LDT prediction is calibrated via `recal_value` to match
the variance, so variance should sit on the diagonal. The skew and kurtosis
panels show where the LDT shape model deviates from the simulation truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def moments(pdf: np.ndarray, k: np.ndarray, dk: float):
    """(norm, mean, variance, skewness, excess kurtosis) per row."""
    norm = np.sum(pdf * dk, axis=1)
    mean = np.sum(k[None, :] * pdf * dk, axis=1) / norm
    centered = k[None, :] - mean[:, None]
    var = np.sum(centered ** 2 * pdf * dk, axis=1) / norm
    m3 = np.sum(centered ** 3 * pdf * dk, axis=1) / norm
    m4 = np.sum(centered ** 4 * pdf * dk, axis=1) / norm
    return norm, mean, var, m3 / var ** 1.5, m4 / var ** 2 - 3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sim-npz", type=Path, required=True)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None,
                   help="Default: outputs/plots/l1/theta<T>/diagnostics.")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    sim = dict(np.load(args.sim_npz, allow_pickle=True))
    th = dict(np.load(args.theory_npz, allow_pickle=True))
    if not np.allclose(sim["kappa_bins"], th["kappa_bins"]):
        raise ValueError("kappa_bins differ")
    k = sim["kappa_bins"]; dk = k[1] - k[0]
    theta = float(sim["theta"])

    cids = sim["selected_indices"] // 7
    uniq = np.unique(cids)

    ak = np.abs(k)[None, :]
    sim_pdf_pp = np.where(ak > 1e-30, sim["l1_norms"] / np.where(ak > 1e-30, ak, 1), 0.0)
    sim_pdf = np.array([np.nanmean(sim_pdf_pp[cids == c], axis=0) for c in uniq])

    th_cids = th["selected_indices"] // 7
    th_pdf = np.array([th["pdf_theory"][np.where(th_cids == c)[0][0]] for c in uniq])

    _, mu_s, var_s, skew_s, kurt_s = moments(sim_pdf, k, dk)
    _, mu_t, var_t, skew_t, kurt_t = moments(th_pdf, k, dk)

    first = np.array([np.where(cids == c)[0][0] for c in uniq])
    om = sim["params"][first, 0]

    tag = args.tag or f"theta{theta:.0f}"
    outdir = args.outdir or (REPO_ROOT / "outputs" / "plots" / "l1" / f"theta{theta:.0f}" / "diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=om.min(), vmax=om.max())

    pairs = [
        (mu_s,   mu_t,   r"$\langle \kappa \rangle$",      "Mean"),
        (var_s,  var_t,  r"$\sigma^2_\kappa$",              "Variance"),
        (skew_s, skew_t, r"$S_3(\kappa)$",                  "Skewness"),
        (kurt_s, kurt_t, r"$K_4(\kappa)$ (excess)",         "Excess kurtosis"),
    ]
    for ax, (xv, yv, label, title) in zip(axes, pairs):
        for i in range(len(uniq)):
            ax.scatter(xv[i], yv[i], color=cmap(norm(om[i])), s=18, alpha=0.85, edgecolor="none")
        lo = min(xv.min(), yv.min()); hi = max(xv.max(), yv.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1e-30
        rng = (lo - pad, hi + pad)
        ax.plot(rng, rng, "k-", lw=0.8, alpha=0.8)
        ax.set_xlim(rng); ax.set_ylim(rng)
        ax.set_xlabel(f"sim  {label}"); ax.set_ylabel(f"theory  {label}")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label=r"$\Omega_m$", shrink=0.85)

    out = outdir / f"moments_theory_vs_sim_{tag}.pdf"
    plt.savefig(out, transparent=True)
    plt.close()
    print(f"Saved {out}")

    # Print stats
    print(f"\nAcross {len(uniq)} cosmologies (sim PDFs averaged over the 7 perms):")
    print(f"  {'moment':>14} {'sim mean':>12} {'theory mean':>13} {'th/sim ratio':>13}")
    for (xv, yv, _, name) in pairs:
        ratio = (yv / xv).mean()
        print(f"  {name:>14} {xv.mean():>12.4e} {yv.mean():>13.4e} {ratio:>13.4f}")


if __name__ == "__main__":
    main()
