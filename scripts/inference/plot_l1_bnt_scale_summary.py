#!/usr/bin/env python3
"""
BNT bin 4 scale-summary plot — bias vs smoothing scale, for full and best-cut.

For each θ ∈ {20, 25, 30}, reads the BNT NPE samples for the "full" κ range and
the tight ±0.0003 cut, and plots the max posterior shift on (Ωm, σ₈) as a
function of θ.

This is the BNT analogue of `plot_l1_scale_summary_simple.py`. The headline
message: BNT + a Gaussian-core κ-cut reaches <0.5σ at θ ∈ {20, 25, 30},
extending the LDT validity floor 20' below the standard analysis.
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
    th = re.search(r"theory_doth_l1(_bnt(\d+))?_bin\d+_realizations", train_stem)
    si = re.search(r"sim_doth_l1(_bnt(\d+))?_bin\d+_realizations", train_stem)
    if th:
        training = "theory"
        mode = f"bnt{th.group(2)}" if th.group(1) else "std"
    elif si:
        training = "sim"
        mode = f"bnt{si.group(2)}" if si.group(1) else "std"
    else:
        return None
    m = re.search(r"theta(\d+\.\d+)", train_stem)
    if m is None:
        return None
    theta = float(m.group(1))
    body = name[: -len("_npe.npy")]
    cut_tag = "".join(CUT_RE.findall(body)) or "_full"
    return dict(training=training, theta=theta, cut_tag=cut_tag, mode=mode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "samples")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "plots" / "l1_bnt" /
                           "bnt_scale_summary.pdf")
    p.add_argument("--params", nargs="+", default=["Om", "sigma8"])
    p.add_argument("--mode", type=str, default="bnt4")
    args = p.parse_args()

    idx = [PARAM_NAMES.index(n) for n in args.params]

    # Collect (mode, theta, cut) -> {training: path}
    pairs: dict[tuple[float, str], dict[str, Path]] = defaultdict(dict)
    for f in sorted(args.samples_dir.glob("samples_l1_*_npe.npy")):
        info = parse_filename(f.name)
        if info is None or info["mode"] != args.mode:
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
        print(f"No {args.mode} samples found in {args.samples_dir}")
        return

    by_theta: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_theta[r["theta"]].append(r)

    thetas = sorted(by_theta)
    fulls, bests, best_cuts = [], [], []
    for t in thetas:
        rs = by_theta[t]
        full = next((r for r in rs if r["cut"] == "_full"), None)
        fulls.append(full["max_abs_shift"] if full else np.nan)
        best = min(rs, key=lambda r: r["max_abs_shift"])
        bests.append(best["max_abs_shift"])
        best_cuts.append(best["cut"])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.axhspan(1e-3, 1.0, color="#cbe9d4", alpha=0.45, zorder=0)
    ax.text(thetas[-1] - 1, 0.5, "LDT usable",
            fontsize=10, color="#2e7d3e", ha="right", va="center",
            fontweight="bold", zorder=1)

    ax.plot(thetas, fulls, "s--", color="#7a7a7a", lw=1.8, ms=8,
            label="full κ-range (no cut)", zorder=3)
    ax.plot(thetas, bests, "o-", color="#1f77b4", lw=2.2, ms=9,
            label="best κ-cut (±0.0003 region)", zorder=4)

    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.0, zorder=2)
    ax.text(thetas[0] + 0.5, 1.08, r"1 σ (usability threshold)",
            color="#c0392b", fontsize=9, zorder=2)

    for t, b, ct in zip(thetas, bests, best_cuts):
        m_min = re.search(r"amin([m]?[0-9p]+)", ct)
        m_max = re.search(r"amaxp?([0-9p]+)", ct)
        if m_min and m_max:
            lo = m_min.group(1).replace("m", "-").replace("p", ".")
            hi = m_max.group(1).replace("p", ".")
            label = f"κ ∈ [{lo}, {hi}]"
        else:
            label = ct
        ax.annotate(label, xy=(t, b), xytext=(8, 8),
                    textcoords="offset points", fontsize=8,
                    color="#1f77b4")

    ax.set_xlabel(r"smoothing scale $\theta_1$  [arcmin]", fontsize=11)
    ax.set_ylabel(rf"max posterior shift on ({', '.join(args.params)})   [σ]",
                  fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 12)
    ax.set_xlim(min(thetas) - 3, max(thetas) + 4)
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_title(f"LDT ℓ₁-norm posterior bias vs θ — BNT bin 4 ({args.mode})",
                 fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")
    print("\nUnderlying table:")
    print(f"  {'θ':>5}  {'full':>8}  {'best':>8}   best cut")
    for t, f, b, ct in zip(thetas, fulls, bests, best_cuts):
        print(f"  {t:>5.1f}  {f:>8.2f}  {b:>8.2f}   {ct.lstrip('_')}")


if __name__ == "__main__":
    main()
