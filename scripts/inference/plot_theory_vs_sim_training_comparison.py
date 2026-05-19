#!/usr/bin/env python3
"""
Overlay triangle plots comparing sim-trained vs theory-trained posteriors.

For each covariance approach (percosmo_cov and fidcov), produces one triangle
plot with four contours:

  1. Theory-trained, theory fiducial  — red,     solid
  2. Theory-trained, sim fiducial     — orange,   solid
  3. Sim-trained,    theory fiducial  — darkred,  dashed
  4. Sim-trained,    sim fiducial     — blue,     dashed

Contours 3 and 4 are the same across both plots (existing full-ell sample files).

Usage
-----
# Auto-detect from samples-dir (default)
python plot_theory_vs_sim_training_comparison.py

# Override samples directory
python plot_theory_vs_sim_training_comparison.py --samples-dir /path/to/samples

# Only produce the percosmo-covariance plot
python plot_theory_vs_sim_training_comparison.py --approaches percosmo_cov
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots


REPO_ROOT = Path(__file__).resolve().parents[2]

TRUE_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
PARAM_NAMES = ["Om", "sigma8", "w0", "H0", "ns", "Ob"]
PARAM_LABELS = [
    r"\Omega_m", r"\sigma_8", r"w_0", r"H_0", r"n_s", r"\Omega_b"
]

# Consistent style per (training, fiducial) combination
STYLES = {
    "theory_theory": dict(color="red",     ls="solid",  lw=1.5, label="Theory-trained, theory fid"),
    "theory_sim":    dict(color="orange",   ls="solid",  lw=1.5, label="Theory-trained, sim fid"),
    "sim_theory":    dict(color="darkred",  ls="dashed", lw=1.5, label="Sim-trained, theory fid"),
    "sim_sim":       dict(color="blue",     ls="dashed", lw=1.5, label="Sim-trained, sim fid"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay sim-trained vs theory-trained posterior contours.",
    )
    parser.add_argument(
        "--samples-dir", type=Path,
        default=REPO_ROOT / "outputs" / "samples",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs" / "plots",
    )
    parser.add_argument(
        "--approaches", nargs="+",
        default=["percosmo_cov", "fidcov"],
        choices=["percosmo_cov", "fidcov"],
        help="Which covariance approaches to plot.",
    )
    parser.add_argument(
        "--sim-trained-sim-fid-file", type=Path, default=None,
        help="Explicit path for the sim-trained, sim-fiducial sample file.",
    )
    parser.add_argument(
        "--sim-trained-theory-fid-file", type=Path, default=None,
        help="Explicit path for the sim-trained, theory-fiducial sample file.",
    )
    parser.add_argument(
        "--theory-trained-sim-fid-file", type=Path, default=None,
        help="Explicit path for the theory-trained, sim-fiducial sample file.",
    )
    parser.add_argument(
        "--theory-trained-theory-fid-file", type=Path, default=None,
        help="Explicit path for the theory-trained, theory-fiducial sample file.",
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix appended to the output PDF filename (e.g. '_doth').",
    )
    parser.add_argument(
        "--filled", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-filled", dest="filled", action="store_false",
    )
    return parser.parse_args()


def find_sample_file(samples_dir: Path, pattern: str) -> Path | None:
    """Return the first file in samples_dir whose name contains all words in pattern."""
    words = pattern.split()
    for f in sorted(samples_dir.glob("*.npy")):
        if all(w in f.name for w in words):
            return f
    return None


def load_samples(path: Path) -> np.ndarray:
    s = np.load(path)
    print(f"  Loaded {s.shape[0]} samples from {path.name}")
    return s


def make_mcsample(samples: np.ndarray, label: str) -> MCSamples:
    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS,
        label=label,
    )


def make_comparison_plot(
    approach: str,
    samples_dir: Path,
    output_dir: Path,
    filled: bool,
    sim_trained_sim_fid_file: Path | None = None,
    sim_trained_theory_fid_file: Path | None = None,
    theory_trained_sim_fid_file: Path | None = None,
    theory_trained_theory_fid_file: Path | None = None,
    output_suffix: str = "",
) -> Path | None:
    """Build a 4-contour triangle plot for the given covariance approach."""

    approach_tag = "percosmo" if approach == "percosmo_cov" else "fidcov"

    # Locate sample files — explicit overrides take priority over auto-detection
    auto_specs = {
        "theory_theory": ["realizations", approach_tag, "fid", "theory", "pixwin"],
        "theory_sim":    ["realizations", approach_tag, "fid", "sim", "fiducial"],
        "sim_theory":    ["sim_cls_bin4_nobaryons", "fid", "theory", "pixwin"],
        "sim_sim":       ["sim_cls_bin4_nobaryons", "fid", "sim", "fiducial"],
    }
    explicit_overrides = {
        "theory_theory": theory_trained_theory_fid_file,
        "theory_sim":    theory_trained_sim_fid_file,
        "sim_theory":    sim_trained_theory_fid_file,
        "sim_sim":       sim_trained_sim_fid_file,
    }

    mc_list = []
    missing = []

    for key, words in auto_specs.items():
        # Use explicit override if provided
        if key in explicit_overrides and explicit_overrides[key] is not None:
            p = explicit_overrides[key]
            if not p.exists():
                print(f"  [WARN] Explicit file not found: {p}")
                missing.append(key)
                continue
        else:
            p = find_sample_file(samples_dir, " ".join(words))
            # For sim-trained files exclude lmax runs
            if p is not None and key.startswith("sim_") and "lmax" in p.name:
                candidates = [
                    f for f in sorted(samples_dir.glob("*.npy"))
                    if all(w in f.name for w in words) and "lmax" not in f.name
                ]
                p = candidates[0] if candidates else None

        if p is None:
            print(f"  [WARN] Could not find sample file for key='{key}' words={words}")
            missing.append(key)
            continue

        sty = STYLES[key]
        mc_list.append(make_mcsample(load_samples(p), sty["label"]))

    if len(mc_list) < 2:
        print(f"  [SKIP] Not enough sample files found for approach '{approach}'.")
        return None

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 1.0
    g.settings.legend_fontsize = 10

    g.triangle_plot(
        mc_list,
        filled=filled,
        markers={name: val for name, val in zip(PARAM_NAMES, TRUE_PARAMS)},
    )

    title_map = {
        "percosmo_cov": "Per-cosmology covariance (Approach A)",
        "fidcov":       "Fiducial covariance (Approach B)",
    }
    g.fig.suptitle(title_map[approach], fontsize=13, y=1.01)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"comparison_theory_vs_sim_training_{approach}{output_suffix}.pdf"
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()

    for approach in args.approaches:
        print(f"\n=== {approach} ===")
        make_comparison_plot(
            approach=approach,
            samples_dir=args.samples_dir,
            output_dir=args.output_dir,
            filled=args.filled,
            sim_trained_sim_fid_file=args.sim_trained_sim_fid_file,
            sim_trained_theory_fid_file=args.sim_trained_theory_fid_file,
            theory_trained_sim_fid_file=args.theory_trained_sim_fid_file,
            theory_trained_theory_fid_file=args.theory_trained_theory_fid_file,
            output_suffix=args.output_suffix,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
