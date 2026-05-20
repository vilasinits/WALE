#!/usr/bin/env python3
"""
Side-by-side overview of the L1 datavectors:
  Left  panel: all theory L1 vectors (one row per cosmology).
  Right panel: per-cosmology sim L1 means (averaged over the 7 perms).

Both panels share x and y limits and the same colour scheme (default: Omega_m),
to allow a direct visual comparison of theory-vs-sim shape across the
cosmology grid.

Usage:
    conda run -n wale python scripts/diagnostics/plot_l1_datavectors_overview.py \\
        --sim-npz data/l1/simulations/sim_doth_l1_bin4_theta30.0_ratio2.0_nobaryons.npz \\
        --theory-npz data/l1/theory/theory_doth_l1_bin4_theta30.0_ratio2.0_simbin.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-npz", type=Path, required=True)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default: outputs/plots/l1/theta<T>/diagnostics).")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--color-by", type=int, default=0,
                   help="Parameter index for colour (0=Om).")
    p.add_argument("--yscale", choices=["linear", "log"], default="linear")
    args = p.parse_args()

    with np.load(args.sim_npz, allow_pickle=True) as d:
        sim_params = np.asarray(d["params"], dtype=float)
        sim_sel = np.asarray(d["selected_indices"], dtype=int)
        sim_kappa = np.asarray(d["kappa_bins"], dtype=float)
        sim_l1 = np.asarray(d["l1_norms"], dtype=float)
        theta = float(d["theta"])
    with np.load(args.theory_npz, allow_pickle=True) as d:
        th_params = np.asarray(d["params"], dtype=float)
        th_sel = np.asarray(d["selected_indices"], dtype=int)
        th_kappa = np.asarray(d["kappa_bins"], dtype=float)
        th_l1 = np.asarray(d["l1_norms"], dtype=float)

    if not np.allclose(sim_kappa, th_kappa):
        raise ValueError("sim/theory kappa_bins differ")
    kappa = sim_kappa

    # Per-cosmo means
    cids = sim_sel // 7
    uniq = np.unique(cids)
    sim_mean = np.array([sim_l1[cids == c].mean(0) for c in uniq])
    # Same grouping/order for theory (per-cosmo identical rows)
    th_cids = th_sel // 7
    if not np.array_equal(uniq, np.unique(th_cids)):
        raise ValueError("sim/theory cosmo_id sets differ")
    th_mean = np.array([th_l1[th_cids == c].mean(0) for c in uniq])
    first_sim = np.array([np.where(cids == c)[0][0] for c in uniq])
    color_vals = sim_params[first_sim, args.color_by]

    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=color_vals.min(), vmax=color_vals.max())

    # Shared y limits — use the union range over all rows of both panels
    lo = min(np.nanmin(th_mean), np.nanmin(sim_mean))
    hi = max(np.nanmax(th_mean), np.nanmax(sim_mean))
    pad = 0.05 * (hi - lo) if args.yscale == "linear" else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharex=True, sharey=True)
    titles = ["Theory $L_1$ (per cosmology)", "Sim $L_1$ mean (per cosmology, over 7 perms)"]
    for ax, data, title in zip(axes, [th_mean, sim_mean], titles):
        for i in range(data.shape[0]):
            ax.plot(kappa, data[i], color=cmap(norm(color_vals[i])), lw=0.6, alpha=0.6)
        ax.set_xlabel(r"$\kappa$")
        ax.set_title(f"{title}, θ={theta}'")
        ax.set_yscale(args.yscale)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$L_1(\kappa)$")

    # shared y-limits
    if args.yscale == "linear":
        axes[0].set_ylim(lo - pad, hi + pad)
    else:
        axes[0].set_ylim(max(lo, 1e-12), hi * 1.2)

    # colorbar (single, shared)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, label=r"$\Omega_m$" if args.color_by == 0 else f"param[{args.color_by}]",
                         shrink=0.9, pad=0.02)

    tag = args.tag or f"theta{theta:.0f}"
    outdir = args.outdir or (REPO_ROOT / "outputs" / "plots" / "l1" / f"theta{theta:.0f}" / "diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"l1_datavectors_overview_{tag}.pdf"
    plt.savefig(out, transparent=True)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
