#!/usr/bin/env python3
"""
Calibrate the kappa-histogram range for the BNT-bin-4 DoTH field at a given theta.

Loads one fiducial cosmoGRID perm, applies the BNT matrix, computes the
DoTH-filtered map, and reports σ_κ. Recommends a kappa range that brackets
~7σ on either side and plots the histogram.

Run once per smoothing scale before launching the full sim grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PROC = REPO_ROOT / "scripts" / "sim_processing"
if str(SIM_PROC) not in sys.path:
    sys.path.insert(0, str(SIM_PROC))

from l1_norm_processing_bnt import BNT_MATRIX, smooth_map_dual  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theta", type=float, default=30.0)
    p.add_argument("--theta-ratio", type=float, default=2.0)
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--bnt-bin", type=int, default=4, choices=(1, 2, 3, 4))
    p.add_argument("--perm", type=int, default=0)
    p.add_argument("--fiducial-dir", type=Path,
                   default=Path("/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    path = args.fiducial_dir / f"perm_{args.perm:04d}" / "projected_probes_maps_nobaryons512.h5"
    print(f"Loading {path}")
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        kgs = np.stack([np.array(f[f"kg/stage3_lensing{i}"]) for i in (1, 2, 3, 4)], axis=0)

    kgs_bnt = BNT_MATRIX @ kgs
    bnt_row = args.bnt_bin - 1
    kg = kgs_bnt[bnt_row]

    k1, k2 = smooth_map_dual(
        kg, args.theta, args.theta * args.theta_ratio,
        nside=args.nside, lmax_factor=3.0, fast_mode=False,
    )
    doth = k2 - k1
    sigma = float(np.std(doth))
    rng = float(np.max(np.abs(doth)))

    suggested = 7 * sigma
    # Round to a clean decimal of 1e-3.
    nice = max(1e-3, round(suggested * 1000) / 1000.0)

    print(f"\nBNT bin {args.bnt_bin}, theta={args.theta} arcmin (ratio={args.theta_ratio}):")
    print(f"  σ_κ (DoTH)           = {sigma:.4e}")
    print(f"  max |κ| on full sky  = {rng:.4e}")
    print(f"  7 σ                  = {suggested:.4e}")
    print(f"  recommended (rounded to nearest 1e-3): ±{nice:.3f}")
    print(f"  CLI:  --kappa-min {-nice:.4f} --kappa-max {nice:.4f}")

    outdir = (
        args.out.parent if args.out is not None
        else REPO_ROOT / "outputs" / "plots" / "l1_bnt" / f"theta{int(args.theta)}" / "diagnostics"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = args.out or outdir / "kappa_range_calibration.pdf"

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    edges = np.linspace(-rng, rng, 401)
    ax.hist(doth, bins=edges, density=True, color="#1f77b4", alpha=0.85,
            label=fr"DoTH κ, BNT bin {args.bnt_bin}, $\theta_1={args.theta}'$")
    for x, lbl, color in [(-suggested, "−7σ", "#d62728"), (suggested, "+7σ", "#d62728"),
                          (-nice, f"−{nice:.3f}", "#2ca02c"), (nice, f"+{nice:.3f}", "#2ca02c")]:
        ax.axvline(x, color=color, ls="--", lw=1.0, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel(r"DoTH $\kappa$")
    ax.set_ylabel(r"PDF (density)")
    ax.set_title(f"BNT bin {args.bnt_bin} DoTH PDF — σ={sigma:.3e}, ±{nice:.3f} envelope")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
