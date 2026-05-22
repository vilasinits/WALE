#!/usr/bin/env python3
"""
Per-cut overlay contour plot for the L1 SBI comparison: sim-trained vs
theory-trained NPE posteriors at the same κ-cut. The fiducial observation is
always the mean of the 200 sim fiducial L1 realizations.

For each cut tag found in `--samples-dir`, identify the matching sim- and
theory-trained sample files (by stem prefix), build a triangle plot with two
contours, and save to outputs/plots/l1_contours_theta30_cut_<tag>.pdf.

Optionally writes a small CSV summary of the per-cut posterior-mean shift
between sim-trained and theory-trained NPE.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots


REPO_ROOT = Path(__file__).resolve().parents[2]
PARAM_NAMES = ["Om", "sigma8", "w0", "H0", "ns", "Ob"]
PARAM_LABELS = [r"\Omega_m", r"\sigma_8", r"w_0", r"H_0", r"n_s", r"\Omega_b"]
TRUE_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])


# Note: '_norecal' lives in the *training* stem and selects the theory variant;
# it is NOT a data-vector cut and must not appear in cut_tag.
CUT_RE = re.compile(r"(_amin[^_]+|_amax[^_]+|_tf[^_]+|_dcrm)")


def parse_filename(name: str) -> dict | None:
    """Return (training, theta, cut_tag) for L1 sample files. None for others."""
    if not name.endswith("_npe.npy") or "_l1_" not in name:
        return None
    # Training source: was it the sim or theory cov-injected pseudo-realizations file?
    # The training stem is the substring between "samples_l1_norms_" and "_fid_".
    if "_fid_" not in name:
        return None
    train_stem = name.split("_fid_", 1)[0]
    # Match both standard (`_doth_l1_bin<N>_realizations`) and BNT
    # (`_doth_l1_bnt<N>_bin<N>_realizations`) training stems and capture the
    # mode (None for standard, "bnt<N>" for BNT) so callers can group separately.
    th_m = re.search(r"theory_doth_l1(_bnt\d+)?_bin\d+_realizations", train_stem)
    sim_m = re.search(r"sim_doth_l1(_bnt\d+)?_bin\d+_realizations", train_stem)
    if th_m:
        training = "theory_norecal" if "_norecal" in train_stem else "theory"
        mode = (th_m.group(1) or "").lstrip("_") or "std"
    elif sim_m:
        training = "sim"
        mode = (sim_m.group(1) or "").lstrip("_") or "std"
    else:
        return None

    # Theta: the *training* file (left of "_fid_") carries the relevant smoothing scale.
    m = re.search(r"theta(\d+\.\d+)", train_stem)
    if m is None:
        return None
    theta = float(m.group(1))

    body = name[: -len("_npe.npy")]
    tags = "".join(CUT_RE.findall(body))
    return dict(training=training, theta=theta, cut_tag=tags, mode=mode, name=name)


def load_samples(path: Path) -> np.ndarray:
    s = np.load(path)
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "outputs" / "samples")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default: outputs/plots/l1/<theta-tag>/overlays).")
    p.add_argument("--theta-tag", type=str, default="theta30",
                   help="Tag used in output filenames (default theta30).")
    p.add_argument("--mode", type=str, default=None,
                   help="Filter to a single training mode tag, e.g. 'std' or 'bnt4'. "
                        "Default: include all modes (each gets its own plot).")
    p.add_argument("--theta", type=float, default=None,
                   help="Filter to a single smoothing scale (in arcmin). "
                        "Default: include every θ found in --samples-dir. "
                        "Pass this when writing into a θ-specific outdir to "
                        "avoid mixing scales.")
    args = p.parse_args()

    if args.outdir is None:
        args.outdir = REPO_ROOT / "outputs" / "plots" / "l1" / args.theta_tag / "overlays"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Find matching pairs of (sim, theory*) samples per (mode, theta, cut_tag).
    by_cut: dict[tuple[str, float, str], dict[str, Path]] = {}
    for f in sorted(args.samples_dir.glob("samples_l1_*_npe.npy")):
        info = parse_filename(f.name)
        if info is None or info["training"] not in ("sim", "theory", "theory_norecal"):
            continue
        if args.mode is not None and info["mode"] != args.mode:
            continue
        if args.theta is not None and not np.isclose(info["theta"], args.theta):
            continue
        by_cut.setdefault(
            (info["mode"], info["theta"], info["cut_tag"]), {}
        )[info["training"]] = f

    if not by_cut:
        print(f"No L1 sample files found in {args.samples_dir}")
        return

    print(f"Found {len(by_cut)} (mode, theta, cut) cells in {args.samples_dir}.")
    summary_rows = []
    for (mode, theta, cut_tag), files in sorted(by_cut.items()):
        if "sim" not in files:
            print(f"  [skip] mode={mode} theta={theta} cut='{cut_tag}' missing sim.")
            continue
        s_sim = load_samples(files["sim"])
        theory_variants = [k for k in files if k.startswith("theory")]
        if not theory_variants:
            print(f"  [skip] mode={mode} theta={theta} cut='{cut_tag}' missing theory.")
            continue
        for tname in theory_variants:
            s_th = load_samples(files[tname])
            tag_suffix = "" if tname == "theory" else f"_{tname.split('_',1)[1]}"
            nice_tag = (cut_tag.lstrip("_") or "full") + tag_suffix
            mode_tag = "" if mode == "std" else f"_{mode}"
            print(f"  mode={mode} theta={theta} cut={nice_tag}: sim N={s_sim.shape[0]}, "
                  f"{tname} N={s_th.shape[0]}")
            mc_sim = MCSamples(samples=s_sim, names=PARAM_NAMES, labels=PARAM_LABELS,
                               label="sim-trained")
            mc_th = MCSamples(samples=s_th, names=PARAM_NAMES, labels=PARAM_LABELS,
                              label=tname + "-trained")
            g = plots.get_subplot_plotter()
            g.settings.figure_legend_frame = False
            g.settings.alpha_filled_add = 1.0
            g.settings.legend_fontsize = 10
            g.triangle_plot([mc_sim, mc_th], filled=True,
                            markers={n: v for n, v in zip(PARAM_NAMES, TRUE_PARAMS)})
            theta_tag = f"theta{theta:.0f}"
            g.fig.suptitle(
                f"L1 NPE contours [{mode}]: sim vs {tname}, {theta_tag}, cut={nice_tag}",
                fontsize=12, y=1.01,
            )
            outdir = args.outdir if args.outdir is not None else (
                REPO_ROOT / "outputs" / "plots" / "l1" / theta_tag / "overlays"
            )
            outdir.mkdir(parents=True, exist_ok=True)
            out_pdf = outdir / f"l1_contours_{theta_tag}{mode_tag}_cut_{nice_tag}.pdf"
            plt.savefig(out_pdf, transparent=True, bbox_inches="tight")
            plt.close()
            print(f"    saved {out_pdf}")

            mu_sim, sig_sim = s_sim.mean(0), s_sim.std(0, ddof=1)
            mu_th = s_th.mean(0)
            shift = (mu_th - mu_sim) / sig_sim
            summary_rows.append(dict(mode=mode, theta=theta, cut_tag=nice_tag,
                                     shifts=shift,
                                     max_abs=float(np.max(np.abs(shift)))))

    if summary_rows:
        print("\nPer-(mode, theta, cut) posterior shift (theory − sim) / sigma_sim:")
        print(f"  {'mode':>5} {'theta':>6} {'cut':>30} {'Om':>7} {'sigma8':>8} {'w0':>7} {'H0':>7} {'ns':>7} {'Ob':>7}    max")
        for r in summary_rows:
            shifts = "  ".join(f"{x:+.2f}" for x in r["shifts"])
            print(f"  {r['mode']:>5} {r['theta']:>6.1f} {r['cut_tag']:>30}  {shifts}   {r['max_abs']:+.2f}")


if __name__ == "__main__":
    main()
