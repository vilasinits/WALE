#!/usr/bin/env python3
"""
Side-by-side debug plot for a handful of representative cosmologies: theory L1
vs sim L1 (mean over the 7 perms) overlaid, on both linear and log y-axes.

Picks: closest-to-fiducial, highest Om, lowest Om, and a random middle cosmo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FID = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])


def pick_cosmos(params_per_cosmo: np.ndarray) -> list[tuple[int, str]]:
    om = params_per_cosmo[:, 0]
    s8 = params_per_cosmo[:, 1]
    fid_dist = np.linalg.norm((params_per_cosmo - FID[None, :]) /
                              (np.std(params_per_cosmo, axis=0) + 1e-30), axis=1)
    picks = [
        (int(np.argmin(fid_dist)), "closest-to-fid"),
        (int(np.argmax(om)),       "highest Om"),
        (int(np.argmin(om)),       "lowest Om"),
        (int(np.argmax(s8)),       "highest sigma8"),
        (int(np.argmin(s8)),       "lowest sigma8"),
    ]
    # de-dupe by index
    seen = set(); uniq = []
    for idx, lbl in picks:
        if idx not in seen:
            seen.add(idx); uniq.append((idx, lbl))
    return uniq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sim-npz", type=Path, required=True)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: outputs/plots/l1/theta<T>/diagnostics/l1_single_cosmo_debug_theta<T>.pdf).")
    args = p.parse_args()

    with np.load(args.sim_npz, allow_pickle=True) as d:
        sim_params = np.asarray(d["params"], dtype=float)
        sim_sel = np.asarray(d["selected_indices"], dtype=int)
        kappa = np.asarray(d["kappa_bins"], dtype=float)
        sim_l1 = np.asarray(d["l1_norms"], dtype=float)
        theta = float(d["theta"])
    with np.load(args.theory_npz, allow_pickle=True) as d:
        th_params = np.asarray(d["params"], dtype=float)
        th_sel = np.asarray(d["selected_indices"], dtype=int)
        th_l1 = np.asarray(d["l1_norms"], dtype=float)
        recal = np.asarray(d["recal_value"], dtype=float)

    cids = sim_sel // 7
    uniq = np.unique(cids)
    sim_mean = np.array([sim_l1[cids == c].mean(0) for c in uniq])
    sim_std = np.array([sim_l1[cids == c].std(0, ddof=1) for c in uniq])
    th_cids = th_sel // 7
    th_first = np.array([np.where(th_cids == c)[0][0] for c in uniq])
    th_mean = th_l1[th_first]                     # per-cosmo theory (rows identical anyway)
    recal_per_cosmo = recal[th_first]
    params_per_cosmo = sim_params[np.array([np.where(cids == c)[0][0] for c in uniq])]

    picks = pick_cosmos(params_per_cosmo)

    n_picks = len(picks)
    fig, axes = plt.subplots(n_picks, 2, figsize=(13, 3.0 * n_picks), constrained_layout=True)
    if n_picks == 1:
        axes = axes[None, :]

    for row, (idx, label) in enumerate(picks):
        p = params_per_cosmo[idx]
        title = (f"{label}: Om={p[0]:.3f} σ8={p[1]:.3f} w0={p[2]:.2f} "
                 f"H0={p[3]:.1f} ns={p[4]:.3f} Ob={p[5]:.4f}  "
                 f"recal={recal_per_cosmo[idx]:.4f}")

        ax = axes[row, 0]
        ax.fill_between(kappa, sim_mean[idx] - sim_std[idx], sim_mean[idx] + sim_std[idx],
                        alpha=0.3, color="C0", label=r"sim mean $\pm$ 1σ (per-cosmo)")
        ax.plot(kappa, sim_mean[idx], color="C0", lw=1.4, label="sim mean")
        ax.plot(kappa, th_mean[idx], color="C3", lw=1.4, ls="--", label="theory")
        ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$L_1(\kappa)$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        # Fractional residual + ±1σ band (per-cosmo from 7 perms)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (th_mean[idx] - sim_mean[idx]) / sim_mean[idx]
            sig_pc = sim_std[idx] / np.abs(sim_mean[idx])
        ax.fill_between(kappa, -sig_pc, sig_pc, alpha=0.3, color="C0",
                        label=r"$\sigma_{\rm sim}^{\rm pc}/|L_1^{\rm sim}|$")
        ax.plot(kappa, frac, color="C3", lw=1.2, label=r"$(L_1^{\rm th}-L_1^{\rm sim})/L_1^{\rm sim}$")
        for y in (0.02, 0.05):
            ax.axhline(y, color="black", ls=":", lw=0.7)
            ax.axhline(-y, color="black", ls=":", lw=0.7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("fractional diff")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = args.out or (REPO_ROOT / "outputs" / "plots" / "l1" /
                       f"theta{theta:.0f}" / "diagnostics" /
                       f"l1_single_cosmo_debug_theta{theta:.0f}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, transparent=True)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
