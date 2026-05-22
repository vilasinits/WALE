#!/usr/bin/env python3
"""
BNT-vs-standard contour comparison plot at a single smoothing scale.

The figure answers the central question of the BNT investigation:
"does BNT recover unbiased LDT contours at θ=30 where the standard analysis
cannot?"

It overlays four sets of contours on (Ωm, σ₈) at a fixed θ:

  * sim-trained NPE on standard bin 4   (the SBI reference for the standard analysis)
  * theory-trained NPE on standard bin 4 (the biased contour we want to fix)
  * sim-trained NPE on BNT bin 4         (the SBI reference for the BNT analysis)
  * theory-trained NPE on BNT bin 4      (the contour we expect to be unbiased)

Each entry can be selected via --std-cut and --bnt-cut; pass "full" for the
unrestricted κ range or a literal cut tag like "_aminm0p001_amaxp0p001".

Output: outputs/plots/l1_bnt/<theta-tag>/overlays/bnt_vs_standard_contour.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots


REPO_ROOT = Path(__file__).resolve().parents[2]

PARAM_NAMES = ["Om", "sigma8", "w0", "H0", "ns", "Ob"]
PARAM_LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"H_0", r"n_s", r"\Omega_b"]
TRUE_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])


def sample_path(samples_dir: Path, training: str, theta: float, mode: str,
                cut_tag: str) -> Path:
    """
    Build the samples file path produced by run_npe_inference.py.

    training : "sim" or "theory"
    mode     : "standard" or "bnt4"
    cut_tag  : "full" or e.g. "_aminm0p001_amaxp0p001"
    """
    if mode == "standard":
        train_stem = (
            f"{training}_doth_l1_bin4_realizations_fidcov_theta{theta}_ratio2.0"
        )
        fid_stem = f"sim_doth_l1_bin4_theta{theta}_ratio2.0_fiducial"
    elif mode == "bnt4":
        train_stem = (
            f"{training}_doth_l1_bnt4_bin4_realizations_fidcov_theta{theta}_ratio2.0"
        )
        fid_stem = f"sim_doth_l1_bnt4_bin4_theta{theta}_ratio2.0_fiducial"
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    cut_part = "" if cut_tag == "full" else cut_tag
    return samples_dir / (
        f"samples_l1_norms_{train_stem}_fid_{fid_stem}{cut_part}_npe.npy"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "outputs" / "samples")
    p.add_argument("--theta", type=float, default=30.0)
    p.add_argument("--std-cut", type=str, default="full",
                   help="Cut tag for the standard bin 4 entries (default: full).")
    p.add_argument("--bnt-cut", type=str, default="full",
                   help="Cut tag for the BNT bin 4 entries (default: full).")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--params", nargs="+", default=["Om", "sigma8"])
    p.add_argument("--include-sim", action="store_true", default=True,
                   help="Include sim-trained references (default on).")
    p.add_argument("--no-include-sim", dest="include_sim", action="store_false")
    args = p.parse_args()

    entries: list[tuple[str, Path, str, str]] = []

    if args.include_sim:
        entries.append((
            f"standard bin 4 sim ({args.std_cut})",
            sample_path(args.samples_dir, "sim", args.theta, "standard", args.std_cut),
            "#7a7a7a", "-",
        ))
    entries.append((
        f"standard bin 4 theory ({args.std_cut})",
        sample_path(args.samples_dir, "theory", args.theta, "standard", args.std_cut),
        "#ff7f0e", "--",
    ))
    if args.include_sim:
        entries.append((
            f"BNT bin 4 sim ({args.bnt_cut})",
            sample_path(args.samples_dir, "sim", args.theta, "bnt4", args.bnt_cut),
            "#1f77b4", "-",
        ))
    entries.append((
        f"BNT bin 4 theory ({args.bnt_cut})",
        sample_path(args.samples_dir, "theory", args.theta, "bnt4", args.bnt_cut),
        "#2ca02c", "--",
    ))

    mc_list = []
    contour_colors = []
    contour_ls = []
    for label, path, color, ls in entries:
        if not path.exists():
            raise FileNotFoundError(f"Missing samples file: {path}")
        s = np.load(path)
        mc = MCSamples(samples=s, names=PARAM_NAMES, labels=PARAM_LABELS, label=label)
        mc_list.append(mc)
        contour_colors.append(color)
        contour_ls.append(ls)
        print(f"loaded  {path.name}   N={s.shape[0]}")

    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.55
    g.settings.legend_fontsize = 10
    g.settings.lab_fontsize = 11
    g.settings.linewidth_contour = 1.2

    g.triangle_plot(
        mc_list,
        params=args.params,
        filled=True,
        contour_colors=contour_colors,
        contour_ls=contour_ls,
        markers={n: v for n, v in zip(PARAM_NAMES, TRUE_PARAMS)},
    )

    g.fig.suptitle(
        rf"BNT vs standard LDT contours at $\theta_1={args.theta:.0f}'$ "
        f"(std cut: {args.std_cut}; BNT cut: {args.bnt_cut})",
        fontsize=12, y=1.005,
    )

    if args.out is None:
        args.out = (
            REPO_ROOT / "outputs" / "plots" / "l1_bnt"
            / f"theta{int(args.theta)}" / "overlays" / "bnt_vs_standard_contour.pdf"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
