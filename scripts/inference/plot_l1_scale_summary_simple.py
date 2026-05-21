#!/usr/bin/env python3
"""
Single-panel, paper-ready version of the LDT ℓ₁-norm scale-summary plot.

Plots only what the reader needs:

  * the full κ-range bias vs θ      — what the LDT does if you do nothing
  * the best-κ-cut bias vs θ        — what the LDT can do with our
                                       data-driven cut
  * a 1σ usability threshold + a shaded "LDT usable" region

The information equivalent of `l1_scale_summary.pdf` but with no
per-cut spaghetti.

Output: `outputs/plots/l1/l1_scale_summary_simple.pdf`
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PARAM_NAMES = ["Om", "sigma8", "w0", "H0", "ns", "Ob"]

CUT_RE = re.compile(r"(_amin[^_]+|_amax[^_]+|_tf[^_]+|_dcrm)")


def parse_filename(name: str) -> dict | None:
    if not name.endswith("_npe.npy") or "_l1_" not in name or "_fid_" not in name:
        return None
    train_stem = name.split("_fid_", 1)[0]
    if "theory_doth_l1_bin4_realizations" in train_stem:
        training = "theory_norecal" if "_norecal" in train_stem else "theory"
    elif "sim_doth_l1_bin4_realizations" in train_stem:
        training = "sim"
    else:
        return None
    m = re.search(r"theta(\d+\.\d+)", train_stem)
    if m is None:
        return None
    theta = float(m.group(1))
    body = name[: -len("_npe.npy")]
    cut_tag = "".join(CUT_RE.findall(body)) or "_full"
    return dict(training=training, theta=theta, cut_tag=cut_tag)


def format_cut_label(cut_tag: str) -> str:
    if cut_tag == "_full":
        return "full"
    m_min = re.search(r"amin([m]?[0-9p]+)", cut_tag)
    m_max = re.search(r"amaxp?([0-9p]+)", cut_tag)
    parts = []
    if m_min:
        parts.append(m_min.group(1).replace("m", "-").replace("p", "."))
    if m_max:
        parts.append(m_max.group(1).replace("p", "."))
    if not parts:
        return cut_tag
    return f"κ ∈ [{parts[0]}, {parts[1]}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "samples")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "plots" / "l1" /
                           "l1_scale_summary_simple.pdf")
    p.add_argument("--params", nargs="+", default=["Om", "sigma8"])
    p.add_argument("--exclude-trainings", nargs="*", default=["theory_norecal"])
    p.add_argument("--annotate-cuts", action="store_true", default=True,
                   help="Annotate the best-cut points with the κ-range used.")
    args = p.parse_args()

    idx = [PARAM_NAMES.index(n) for n in args.params]

    # Collect samples
    pairs: dict[tuple[float, str], dict[str, Path]] = defaultdict(dict)
    for f in sorted(args.samples_dir.glob("samples_l1_*_npe.npy")):
        info = parse_filename(f.name)
        if info is None or info["training"] in args.exclude_trainings:
            continue
        if "_dcrm" in info["cut_tag"]:           # DC-removal is a separate experiment
            continue
        pairs[(info["theta"], info["cut_tag"])][info["training"]] = f

    rows = []
    for (theta, cut), files in pairs.items():
        if "sim" not in files or "theory" not in files:
            continue
        s_sim = np.load(files["sim"])
        s_th = np.load(files["theory"])
        shift = (s_th.mean(0) - s_sim.mean(0)) / s_sim.std(0, ddof=1)
        rows.append(dict(theta=theta, cut=cut,
                         max_abs_shift=float(np.max(np.abs(shift[idx])))))

    if not rows:
        print("No data found.")
        return

    # Full and best per θ
    by_theta: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_theta[r["theta"]].append(r)

    thetas = sorted(by_theta)
    fulls = []
    bests = []
    best_cuts = []
    for t in thetas:
        rs = by_theta[t]
        full = next((r for r in rs if r["cut"] == "_full"), None)
        fulls.append(full["max_abs_shift"] if full else np.nan)
        best = min(rs, key=lambda r: r["max_abs_shift"])
        bests.append(best["max_abs_shift"])
        best_cuts.append(best["cut"])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    # Shaded "usable" band below 1σ
    ax.axhspan(1e-3, 1.0, color="#cbe9d4", alpha=0.45, zorder=0)
    ax.text(thetas[-1] - 1, 0.5, "LDT usable",
            fontsize=10, color="#2e7d3e", ha="right", va="center",
            fontweight="bold", zorder=1)

    # Curves
    ax.plot(thetas, fulls, "s--", color="#7a7a7a", lw=1.8, ms=8,
            label="full κ-range (no cut)", zorder=3)
    ax.plot(thetas, bests, "o-", color="#1f77b4", lw=2.2, ms=9,
            label="best data-driven κ-cut", zorder=4)

    # 1σ reference
    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.0, zorder=2)
    ax.text(thetas[0] + 0.5, 1.08, r"1 σ (usability threshold)",
            color="#c0392b", fontsize=9, zorder=2)

    # Cut annotations — above each best-cut point
    if args.annotate_cuts:
        for t, b, ct in zip(thetas, bests, best_cuts):
            label = format_cut_label(ct)
            ax.annotate(label, xy=(t, b), xytext=(8, 8),
                        textcoords="offset points", fontsize=8,
                        color="#1f77b4")

    ax.set_xlabel(r"smoothing scale $\theta_1$  [arcmin]", fontsize=11)
    ax.set_ylabel(rf"max posterior shift on ({', '.join(args.params)})   [σ]",
                  fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 12)
    ax.set_xlim(min(thetas) - 3, max(thetas) + 8)
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_title("LDT ℓ₁-norm posterior bias vs smoothing scale",
                 fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")
    print("\nUnderlying table:")
    print(f"  {'θ':>5}  {'full':>8}  {'best':>8}   best cut")
    for t, f, b, ct in zip(thetas, fulls, bests, best_cuts):
        print(f"  {t:>5.1f}  {f:>8.2f}  {b:>8.2f}   {format_cut_label(ct)}")


if __name__ == "__main__":
    main()
