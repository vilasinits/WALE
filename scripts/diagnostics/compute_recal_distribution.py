#!/usr/bin/env python3
"""
Plot the distribution of recal_value = σ²_LDT / σ²_sim across the cosmology grid.

A sanity check that the LDT variance prediction is close to the sim variance at
the per-cosmology level — recal_value should sit close to 1 with modest scatter.
Values < 0.5 or > 2 indicate that the kernel construction or P(k) call is
mis-set; values systematically offset from 1 indicate a uniform calibration
issue (e.g., pixel-window correction missing on either side).

Usage:
    python scripts/diagnostics/compute_recal_distribution.py \
        --theory-npz data/l1_bnt/theory/theory_doth_l1_bnt4_bin4_theta30.0_ratio2.0_simbin.npz \
        --tag bnt4_theta30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theory-npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    d = dict(np.load(args.theory_npz, allow_pickle=True))
    recal = np.asarray(d["recal_value"], dtype=float)
    var_ldt = np.asarray(d["variance_ldt"], dtype=float)
    var_sim = np.asarray(d["variances"], dtype=float)
    params = np.asarray(d["params"], dtype=float)
    selected_indices = np.asarray(d["selected_indices"], dtype=int)
    theta = float(d["theta"])
    tomo_bin = int(d["tomo_bin"])

    # Collapse perms → 186 unique cosmologies (the per-cosmo recal is identical
    # across perms because variance is the only per-row dependence).
    cids = selected_indices // 7
    uniq = np.unique(cids)
    recal_unique = np.array([recal[cids == c].mean() for c in uniq])
    om_unique = np.array([params[cids == c, 0].mean() for c in uniq])

    outdir = args.outdir or (REPO_ROOT / "outputs" / "plots" / "l1_bnt" /
                             f"theta{int(theta)}" / "diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"bnt{tomo_bin}_theta{int(theta)}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.hist(recal_unique, bins=30, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="#d62728", ls="--", lw=1.0, label="recal = 1")
    ax.axvline(recal_unique.mean(), color="#2ca02c", ls=":", lw=1.0,
               label=f"mean = {recal_unique.mean():.3f}")
    ax.set_xlabel(r"$\mathrm{recal\_value} = \sigma^2_{\rm LDT} / \sigma^2_{\rm sim}$")
    ax.set_ylabel("number of unique cosmologies")
    ax.set_title(f"recal_value distribution — {tag}")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(om_unique, recal_unique, c=om_unique, cmap="viridis", s=20, alpha=0.85)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.0)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"recal_value")
    ax.set_title(r"recal_value vs $\Omega_m$")
    ax.grid(alpha=0.3)

    outpath = outdir / f"recal_distribution_{tag}.pdf"
    plt.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close()

    print(f"BNT bin {tomo_bin}, theta={theta} arcmin")
    print(f"  unique cosmos: {len(uniq)}")
    print(f"  recal_value: mean={recal_unique.mean():.3f}  std={recal_unique.std():.3f}  "
          f"min={recal_unique.min():.3f}  max={recal_unique.max():.3f}")
    print(f"  σ²_LDT (mean over 186 cosmos):   {var_ldt.mean():.3e}")
    print(f"  σ²_sim (mean over 186 cosmos):   {var_sim.mean():.3e}")
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
