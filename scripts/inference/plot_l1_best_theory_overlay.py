#!/usr/bin/env python3
"""
Overlay the *best unbiased* LDT-based contours at each smoothing scale.

For each θ in the configured set, picks the theory-trained NPE samples at
the κ-cut that produced the smallest sim-vs-theory bias on (Ωm, σ₈), and
overplots them all on a single getdist triangle figure. This is the
"what's the best the LDT can do at each scale" view — every contour is
already validated against its sim-trained twin.

Default scales and cuts match the data-driven catalogue in
`outputs/l1_cuts_design.csv` and the analysis of §9 of the methodology
note. Pass `--cuts` to override.

Output: `outputs/plots/l1/l1_best_theory_overlay.pdf`
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

# Best κ-cut per θ as identified in §9 of the methodology note.
DEFAULT_CUTS = {
    40.0: "aminm0p0013_amaxp0p0006",   # asymmetric T=2σ
    50.0: "aminm0p0008_amaxp0p0005",   # asymmetric T=1σ
    60.0: "aminm0p001_amaxp0p001",     # symmetric T=1σ
    100.0: "aminm0p003_amaxp0p003",    # symmetric tight
}


def sample_file(samples_dir: Path, theta: float, cut_tag: str, training: str) -> Path:
    stem_train = f"{training}_doth_l1_bin4_realizations_fidcov_theta{theta}_ratio2.0"
    stem_fid = f"sim_doth_l1_bin4_theta{theta}_ratio2.0_fiducial"
    return samples_dir / f"samples_l1_norms_{stem_train}_fid_{stem_fid}_{cut_tag}_npe.npy"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "samples")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "plots" / "l1" /
                           "l1_best_theory_overlay.pdf")
    p.add_argument("--cuts", nargs="+", default=None,
                   help="Overrides as 'theta:cut_tag', e.g. 40:aminm0p0013_amaxp0p0006")
    p.add_argument("--params", nargs="+", default=PARAM_NAMES,
                   help="Subset of parameters to include in the triangle.")
    p.add_argument("--with-sim-ref", action="store_true",
                   help="Also overlay the sim-trained contour at the largest θ as a SBI ground-truth reference.")
    args = p.parse_args()

    cuts = dict(DEFAULT_CUTS)
    if args.cuts:
        cuts = {}
        for tok in args.cuts:
            t_str, tag = tok.split(":", 1)
            cuts[float(t_str)] = tag

    mc_list = []
    for theta in sorted(cuts):
        path = sample_file(args.samples_dir, theta, cuts[theta], "theory")
        if not path.exists():
            raise FileNotFoundError(path)
        s = np.load(path)
        mc = MCSamples(samples=s, names=PARAM_NAMES, labels=PARAM_LABELS,
                       label=fr"$\theta={theta:.0f}'$ (theory)")
        mc_list.append(mc)
        print(f"θ={theta}: {path.name} — N={s.shape[0]}")

    if args.with_sim_ref:
        ref_theta = max(cuts)
        ref_path = sample_file(args.samples_dir, ref_theta, cuts[ref_theta], "sim")
        if ref_path.exists():
            s_ref = np.load(ref_path)
            mc_ref = MCSamples(samples=s_ref, names=PARAM_NAMES, labels=PARAM_LABELS,
                               label=fr"$\theta={ref_theta:.0f}'$ (sim, ref)")
            mc_list.append(mc_ref)
            print(f"reference: {ref_path.name} — N={s_ref.shape[0]}")

    # Triangle plot
    g = plots.get_subplot_plotter(width_inch=9)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.6
    g.settings.legend_fontsize = 11
    g.settings.axes_fontsize = 9
    g.settings.lab_fontsize = 11
    g.settings.linewidth_contour = 1.2
    g.triangle_plot(mc_list, params=args.params, filled=True,
                    markers={n: v for n, v in zip(PARAM_NAMES, TRUE_PARAMS)})

    title = "Best unbiased LDT-based posteriors across smoothing scales"
    g.fig.suptitle(title, fontsize=13, y=1.005)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
