#!/usr/bin/env python3
"""
Build the empirical safe-lmax(theta_1) curve for DoTH-filtered C_ell SBI.

For each available (theta, lmax) cell, compute the max |off-diagonal shift|
relative to the reference contour A = (sim training, sim fiducial). Mark a cell
as "passing" if that shift is <= --pass-threshold (default 0.5 sigma).

The empirical safe-lmax(theta) is the largest passing lmax per theta. We also
plot the analytic prediction l_cut = 3.83 / theta_1 (with theta in radians) for
comparison; this corresponds to the first zero of the inner top-hat beam.

Usage
-----
    python scripts/inference/plot_safe_lmax_vs_theta.py
    python scripts/inference/plot_safe_lmax_vs_theta.py --pass-threshold 0.5

Output
------
- Prints a per-(theta, lmax) shift table to stdout.
- Saves outputs/plots/safe_lmax_vs_theta.pdf.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))

# Re-use the filename parser from compute_posterior_shifts.py
from compute_posterior_shifts import parse_filename, summarize


def gather_shift_table(samples_dir: Path, pass_threshold: float, kind: str = "lmax") -> dict:
    """For each cell, find max |off-diag shift| and pass/fail.

    kind='lmax': group by (theta, lmax), skip transfer-threshold runs.
    kind='tf':   group by (theta, tf_thresh), use only transfer-threshold runs.
    """
    assert kind in ("lmax", "tf")
    cells: dict[tuple, dict] = {}

    for p in sorted(samples_dir.glob("*_npe.npy")):
        info = parse_filename(p.name)
        if info is None:
            continue
        if kind == "lmax" and info["tf_thresh"] is not None:
            continue
        if kind == "tf" and info["tf_thresh"] is None:
            continue
        key = (info["theta"], info["lmax" if kind == "lmax" else "tf_thresh"])
        cells.setdefault(key, {})[(info["training"], info["fiducial"])] = p

    table = {}
    for cell, files in cells.items():
        if ("sim", "sim") not in files:
            continue
        mu_A, sig_A = summarize(np.load(files[("sim", "sim")]))
        off_diag = []
        for k in [("sim", "theory"), ("theory", "sim")]:
            if k not in files:
                continue
            mu_k, _ = summarize(np.load(files[k]))
            off_diag.append(np.max(np.abs((mu_k - mu_A) / sig_A)))
        if not off_diag:
            continue
        max_shift = max(off_diag)
        table[cell] = dict(max_off_diag=max_shift, passes=max_shift <= pass_threshold)
    return table


def safe_lmax_per_theta(table: dict, lmax_max: int = 1535) -> dict[float, int]:
    """Largest passing lmax per theta. None-lmax entries are mapped to lmax_max."""
    by_theta: dict[float, list[tuple[int, bool]]] = {}
    for (theta, lmax), info in table.items():
        lm = lmax_max if lmax is None else lmax
        by_theta.setdefault(theta, []).append((lm, info["passes"]))

    result = {}
    for theta, runs in by_theta.items():
        passing = sorted([lm for lm, ok in runs if ok])
        result[theta] = passing[-1] if passing else None
    return result


def analytic_lcut(theta_arcmin: float) -> float:
    """Analytic main-lobe cutoff: l ~ 3.83 / theta_rad."""
    theta_rad = theta_arcmin * np.pi / (60 * 180)
    return 3.83 / theta_rad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path,
                        default=REPO_ROOT / "outputs" / "samples")
    parser.add_argument("--pass-threshold", type=float, default=0.5,
                        help="Max |off-diag shift| in sigma_A for a passing cell (default 0.5).")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "outputs" / "plots" / "safe_lmax_vs_theta.pdf")
    parser.add_argument("--lmax-max", type=int, default=1535,
                        help="ell cap for 'full' runs.")
    args = parser.parse_args()

    table = gather_shift_table(args.samples_dir, args.pass_threshold, kind="lmax")
    tf_table = gather_shift_table(args.samples_dir, args.pass_threshold, kind="tf")
    if not table:
        print("No (theta, lmax) cells found.")
        return

    print(f"Pass threshold: max |off-diag shift| <= {args.pass_threshold} sigma_A")
    print(f"\n{'theta':>6} {'lmax':>8} {'max |off-diag|':>16} {'verdict':>10}")
    def key_fn(item):
        (th, lm), _ = item
        return (th, 99999 if lm is None else lm)
    for (theta, lmax), info in sorted(table.items(), key=key_fn):
        lm = "full" if lmax is None else str(lmax)
        verdict = "PASS" if info["passes"] else "fail"
        print(f"{theta:>6.1f} {lm:>8} {info['max_off_diag']:>16.2f} {verdict:>10}")

    safe = safe_lmax_per_theta(table, lmax_max=args.lmax_max)
    print(f"\nEmpirical safe lmax per theta (max passing lmax):")
    for theta, lm in sorted(safe.items()):
        analytic = analytic_lcut(theta)
        if lm is not None:
            print(f"  theta={theta:.1f}': safe_lmax={lm}  "
                  f"(analytic l_cut ~ {analytic:.0f}, ratio empirical/analytic = {lm/analytic:.2f})")
        else:
            print(f"  theta={theta:.1f}': NO passing lmax found.")

    # Transfer-threshold table per theta: lowest passing threshold (most info kept)
    if tf_table:
        print(f"\nTransfer-threshold table:")
        print(f"  {'theta':>6} {'tf':>6} {'max |off-diag|':>16} {'verdict':>10}")
        for (theta, tf), info in sorted(tf_table.items()):
            verdict = "PASS" if info["passes"] else "fail"
            print(f"  {theta:>6.1f} {tf:>6.2f} {info['max_off_diag']:>16.2f} {verdict:>10}")

    # Plot: top panel = max |off-diag| vs lmax per theta; bottom panel = safe lmax vs theta.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: max |off-diag| vs lmax for each theta, with tf=0.1 horizontal lines
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    thetas_lmax = sorted({theta for theta, _ in table.keys()})
    for i, theta in enumerate(thetas_lmax):
        color = cmap(i / max(1, len(thetas_lmax) - 1))
        cells = [(lm if lm is not None else args.lmax_max, table[(theta, lm)]["max_off_diag"])
                 for (th, lm) in table.keys() if th == theta]
        cells.sort()
        lmaxes, shifts = zip(*cells)
        ax.plot(lmaxes, shifts, "-o", color=color, label=f"θ={theta:.0f}'", lw=1.8)

        # vertical line at the analytic cutoff
        ax.axvline(analytic_lcut(theta), color=color, ls=":", alpha=0.7)

        # mark tf=0.1 result as a star at the corresponding analytic_cut location
        tf01 = tf_table.get((theta, 0.1))
        if tf01 is not None:
            ax.plot(analytic_lcut(theta), tf01["max_off_diag"], "*", color=color,
                    ms=18, mec="black", mew=0.8, zorder=10)

    ax.axhline(args.pass_threshold, color="black", ls="--", lw=1, label=f"pass threshold {args.pass_threshold}σ")
    ax.set_xlabel(r"$\ell_{\rm max}$")
    ax.set_ylabel(r"max $|$off-diag shift$|$ ($\sigma_A$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(alpha=0.4, which="both")
    ax.legend(fontsize=9)
    ax.set_title("Hard-lmax cut: agreement vs cutoff; star = tf≥0.1 (transfer cut)")

    # --- right: safe lmax vs theta with analytic curve
    ax = axes[1]
    thetas = sorted(safe.keys())
    safe_lmaxes = [safe[t] for t in thetas]
    th_fine = np.linspace(min(thetas) * 0.8, max(thetas) * 1.2, 100)
    analytic_curve = np.array([analytic_lcut(t) for t in th_fine])
    ax.plot(th_fine, analytic_curve, 'k--', label=r'analytic $\ell_{\rm cut} = 3.83/\theta_1$')
    th_plot = [t for t, lm in zip(thetas, safe_lmaxes) if lm is not None]
    lm_plot = [lm for lm in safe_lmaxes if lm is not None]
    ax.plot(th_plot, lm_plot, 'o', ms=12, color='C0',
            label=f'empirical safe $\\ell_{{\\rm max}}$ ($\\leq$ {args.pass_threshold}$\\sigma$)')
    for t, lm in zip(th_plot, lm_plot):
        ax.annotate(f"{lm}", (t, lm), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel(r"$\theta_1$ (arcmin)")
    ax.set_ylabel(r"$\ell$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.4, which="both")
    ax.legend()
    ax.set_title("Empirical safe $\\ell_{\\rm max}$ vs DoTH inner scale $\\theta_1$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, transparent=True)
    print(f"\nSaved plot: {args.output}")


if __name__ == "__main__":
    main()
