#!/usr/bin/env python3
"""
Summary plot: at what θ does the LDT ℓ₁-norm theory yield unbiased contours?

For each θ ∈ {discovered from filenames} and each cut tag, compute the
posterior-mean shift (theory − sim) / σ_sim on (Ωm, σ8), then plot the **max**
of those two shifts vs θ, one curve per cut family.

A θ counts as "LDT-usable" if any cut gives max-shift < 1σ. The plot includes a
1σ dashed reference line. We also annotate the smallest such θ per cut family.

Usage:
    conda run -n wale python scripts/inference/plot_l1_scale_summary.py \
        [--samples-dir outputs/samples] \
        [--params Om sigma8]
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
    if not name.endswith("_npe.npy") or "_l1_" not in name:
        return None
    if "_fid_" not in name:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "outputs" / "samples")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "plots" / "l1" / "l1_scale_summary.pdf")
    p.add_argument("--params", nargs="+", default=["Om", "sigma8"],
                   help="Parameters to include in the max-shift over (default: Om sigma8).")
    p.add_argument("--exclude-trainings", nargs="*", default=["theory_norecal"],
                   help="Training variants to skip when looking for sim/theory pairs.")
    args = p.parse_args()

    idx = [PARAM_NAMES.index(n) for n in args.params]

    # Collect (theta, cut_tag, training) → path
    by_key: dict[tuple[float, str, str], Path] = {}
    for f in sorted(args.samples_dir.glob("samples_l1_*_npe.npy")):
        info = parse_filename(f.name)
        if info is None:
            continue
        if info["training"] in args.exclude_trainings:
            continue
        by_key[(info["theta"], info["cut_tag"], info["training"])] = f

    # Group by (theta, cut_tag); need both sim and theory.
    pair_data: dict[tuple[float, str], dict[str, Path]] = defaultdict(dict)
    for (theta, cut, train), path in by_key.items():
        pair_data[(theta, cut)][train] = path

    rows = []
    for (theta, cut), files in pair_data.items():
        if "sim" not in files or "theory" not in files:
            continue
        s_sim = np.load(files["sim"])
        s_th = np.load(files["theory"])
        mu_sim = s_sim.mean(axis=0)
        sig_sim = s_sim.std(axis=0, ddof=1)
        mu_th = s_th.mean(axis=0)
        shift = (mu_th - mu_sim) / sig_sim
        rows.append(dict(theta=theta, cut=cut,
                         max_abs_shift=float(np.max(np.abs(shift[idx]))),
                         shift_per_param=shift))

    if not rows:
        print("No (theta, cut) pairs found with both sim and theory samples.")
        return

    # Group by cut tag for plotting; sort each curve by theta.
    by_cut: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cut[r["cut"]].append(r)
    for c in by_cut:
        by_cut[c].sort(key=lambda r: r["theta"])

    # Plot
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True,
                             sharey=True)
    cmap = plt.get_cmap("viridis")
    cuts_sorted = sorted(
        by_cut.keys(),
        key=lambda c: (0 if c == "_full" else 1,
                       _cut_inner_kmax(c) if c != "_full" else -1.0),
    )
    # Panel 1: per-cut curves (each cut is a separate curve over θ)
    ax = axes[0]
    for i, cut in enumerate(cuts_sorted):
        thetas = [r["theta"] for r in by_cut[cut]]
        shifts = [r["max_abs_shift"] for r in by_cut[cut]]
        label = "full κ-range" if cut == "_full" else _format_cut_label(cut)
        color = cmap(i / max(1, len(cuts_sorted) - 1))
        ax.plot(thetas, shifts, "o-", color=color, label=label, lw=1.5, ms=5)
    ax.axhline(1.0, color="black", ls="--", lw=0.8, label="1σ")
    ax.set_xlabel(r"$\theta_1$ [arcmin]")
    ax.set_ylabel(rf"max |Δμ| / σ_sim over ({', '.join(args.params)})")
    ax.set_yscale("log")
    ax.set_title("per-cut bias vs θ")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # Panel 2: best (min) cut per θ — the "with the right κ-cut, how good can LDT be?"
    by_theta: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_theta[r["theta"]].append(r)
    best = []
    for theta in sorted(by_theta):
        best_row = min(by_theta[theta], key=lambda r: r["max_abs_shift"])
        full_row = next((r for r in by_theta[theta] if r["cut"] == "_full"), None)
        best.append((theta, best_row["max_abs_shift"], best_row["cut"],
                     full_row["max_abs_shift"] if full_row else None))
    ax = axes[1]
    thetas = [b[0] for b in best]
    bests = [b[1] for b in best]
    fulls = [b[3] if b[3] is not None else np.nan for b in best]
    ax.plot(thetas, fulls, "s--", color="grey", label="full κ-range",
            lw=1.2, ms=6, alpha=0.7)
    ax.plot(thetas, bests, "o-", color="C0", label="best κ-cut", lw=2, ms=7)
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    # Annotate the best cut at each θ
    for theta, shift, cut, _ in best:
        label = "full" if cut == "_full" else _format_cut_label(cut)
        ax.annotate(label, xy=(theta, shift), xytext=(5, 4),
                    textcoords="offset points", fontsize=7, color="C0",
                    rotation=12)
    ax.set_xlabel(r"$\theta_1$ [arcmin]")
    ax.set_yscale("log")
    ax.set_title("best vs uncut bias")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("LDT ℓ₁-norm posterior bias vs smoothing scale", fontsize=12)
    plt.savefig(args.out, transparent=True); plt.close()
    print(f"Saved {args.out}")

    # Text summary
    print("\nPer-(theta, cut) max |Δμ|/σ_sim:")
    for cut in cuts_sorted:
        label = "full" if cut == "_full" else _format_cut_label(cut)
        for r in by_cut[cut]:
            print(f"  θ={r['theta']:>5.1f}  cut={label:>25}  max_shift={r['max_abs_shift']:.2f}")


def _cut_inner_kmax(cut_tag: str) -> float:
    """Sort key: extract the |κ_max| from the cut tag (smaller = tighter)."""
    m = re.search(r"amaxp?([0-9p]+)", cut_tag)
    if not m:
        return 0.0
    return float(m.group(1).replace("p", "."))


def _format_cut_label(cut_tag: str) -> str:
    parts = []
    m_min = re.search(r"amin([m]?[0-9p]+)", cut_tag)
    m_max = re.search(r"amaxp?([0-9p]+)", cut_tag)
    if m_min:
        s = m_min.group(1).replace("m", "-").replace("p", ".")
        parts.append(f"κmin={s}")
    if m_max:
        parts.append(f"κmax={m_max.group(1).replace('p', '.')}")
    if "_dcrm" in cut_tag:
        parts.append("DC-rm")
    return ", ".join(parts) if parts else cut_tag


if __name__ == "__main__":
    main()
