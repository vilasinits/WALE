#!/usr/bin/env python3
"""
Publication-quality figures for the DoTH-filtered C_ell safe-regime characterisation.

Produces two figures:

(1) outputs/plots/doth_transfer_functions.pdf
    Two panels:
      - left:  |W_DoTH^2| transfer at each theta_1 vs ell (normalised to peak)
      - right: |W_DoTH^2 * W_pix^2| (the combined transfer used in the inference)
    Vertical dashed lines mark the analytic main-lobe cutoff 3.83/theta_1 per theta.

(2) outputs/plots/doth_safe_regime.pdf
    Two panels:
      - left:  max |off-diagonal posterior shift| vs lmax, one curve per theta_1.
               Vertical dashed lines mark the analytic cutoff per theta.
               Stars: corresponding transfer-cut (tf>=0.1) max-shift value placed
                       at the analytic cutoff ell — illustrating that the
                       principled cut achieves ~3-5x tighter agreement.
      - right: empirical safe-lmax vs theta_1, against the analytic 3.83/theta_1 line.

Usage:
    conda run -n wale python scripts/inference/plot_safe_regime_paper.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))

from compute_posterior_shifts import parse_filename, summarize  # noqa: E402


# ------------------------- style -------------------------

mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.0,
})

THETA_VALUES = [10, 15, 20, 30]  # arcmin


def theta_color(theta: float) -> tuple:
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.LogNorm(vmin=8, vmax=40)
    return cmap(norm(theta))


def analytic_lcut(theta_arcmin: float) -> float:
    return 3.83 / (theta_arcmin * np.pi / (60 * 180))


# ------------------------- figure 1: transfer functions -------------------------

def make_transfer_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for theta in THETA_VALUES:
        path = REPO_ROOT / "data" / "cls" / "theory" / (
            f"theory_doth_cls_bin4_simbin_theta{theta}.0_ratio2.0_nside512_pixwin.npz"
        )
        with np.load(path, allow_pickle=True) as d:
            ells = d["ells"]
            w_doth = d["doth_w_doth"]
            transfer = d["doth_transfer"]
        color = theta_color(theta)

        # Left: just W_DoTH^2, normalised
        wd2 = w_doth ** 2
        axes[0].plot(ells, wd2 / wd2.max(), color=color,
                     label=fr"$\theta_1 = {theta}'$")
        axes[0].axvline(analytic_lcut(theta), color=color, ls="--", lw=1.0, alpha=0.7)

        # Right: combined transfer, normalised
        axes[1].plot(ells, transfer / transfer.max(), color=color)
        axes[1].axvline(analytic_lcut(theta), color=color, ls="--", lw=1.0, alpha=0.7)

    for ax, title in zip(axes, [r"$W_{\rm DoTH}^2(\ell)$ (normalised)",
                                r"$W_{\rm DoTH}^2 \cdot W_{\rm pix}^2$ (normalised)"]):
        ax.set_xlabel(r"$\ell$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("normalised transfer")
    axes[0].legend(loc="lower center", ncol=4,
                   bbox_to_anchor=(1.05, -0.32),
                   frameon=False)

    # Annotate one cutoff
    axes[1].annotate(
        r"$\ell_{\rm cut}=3.83/\theta_1$",
        xy=(analytic_lcut(20), 5e-3),
        xytext=(40, 1.2e-2),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ------------------------- figure 2: safe-regime cliff & curve -------------------------

def gather_shifts(samples_dir: Path) -> tuple[dict, dict]:
    """Build {(theta, lmax): max_shift} and {(theta, tf): max_shift}."""
    files: dict[tuple, dict] = {}
    for p in sorted(samples_dir.glob("*_npe.npy")):
        info = parse_filename(p.name)
        if info is None:
            continue
        cell_key = (info["theta"], info["lmax"], info["tf_thresh"])
        files.setdefault(cell_key, {})[(info["training"], info["fiducial"])] = p

    lmax_shifts: dict[tuple[float, int | None], float] = {}
    tf_shifts: dict[tuple[float, float], float] = {}
    for (theta, lmax, tf_thresh), cell_files in files.items():
        if ("sim", "sim") not in cell_files:
            continue
        mu_A, sig_A = summarize(np.load(cell_files[("sim", "sim")]))
        off_diag = []
        for k in [("sim", "theory"), ("theory", "sim")]:
            if k not in cell_files:
                continue
            mu_k, _ = summarize(np.load(cell_files[k]))
            off_diag.append(np.max(np.abs((mu_k - mu_A) / sig_A)))
        if not off_diag:
            continue
        m = max(off_diag)
        if tf_thresh is None:
            lmax_shifts[(theta, lmax)] = m
        else:
            tf_shifts[(theta, tf_thresh)] = m
    return lmax_shifts, tf_shifts


def make_safe_regime_figure(out_path: Path, pass_threshold: float = 0.5) -> None:
    samples_dir = REPO_ROOT / "outputs" / "samples"
    lmax_shifts, tf_shifts = gather_shifts(samples_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Left: max |off-diag| vs lmax per theta
    ax = axes[0]
    for theta in THETA_VALUES:
        color = theta_color(theta)
        cells = [(lm if lm is not None else 1535, v)
                 for (th, lm), v in lmax_shifts.items() if th == theta]
        if not cells:
            continue
        cells.sort()
        lmaxes, shifts = zip(*cells)
        ax.plot(lmaxes, shifts, "-o", color=color, ms=6,
                label=fr"$\theta_1 = {theta}'$")
        # Analytic cutoff vertical line
        ax.axvline(analytic_lcut(theta), color=color, ls="--", lw=0.8, alpha=0.55)
        # Transfer-cut star (tf=0.1) at the analytic cutoff x-position
        tf01 = tf_shifts.get((float(theta), 0.1))
        if tf01 is not None:
            ax.plot(analytic_lcut(theta), tf01, marker="*", color=color, ms=20,
                    mec="black", mew=0.8, zorder=10, ls="")

    ax.axhline(pass_threshold, color="black", ls=":", lw=1.0)
    ax.text(13, pass_threshold * 1.15, fr"pass: $\leq {pass_threshold}\,\sigma_A$",
            fontsize=9, color="black")
    ax.set_xlabel(r"$\ell_{\rm max}$")
    ax.set_ylabel(r"max $|$off-diag shift$|$ ($\sigma_A$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 50)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(r"Hard-$\ell_{\rm max}$ vs transfer cut")

    # Custom legend including the star symbol
    leg_lines = [plt.Line2D([0], [0], color=theta_color(t), marker="o",
                            label=fr"$\theta_1 = {t}'$") for t in THETA_VALUES]
    leg_lines.append(plt.Line2D([0], [0], color="gray", marker="*", ms=14,
                                mec="black", mew=0.6, ls="",
                                label=r"transfer cut $T \geq 0.1$"))
    leg_lines.append(plt.Line2D([0], [0], color="gray", ls="--", lw=0.8,
                                label=r"$\ell_{\rm cut} = 3.83/\theta_1$"))
    ax.legend(handles=leg_lines, loc="upper left", ncol=1, fontsize=9, frameon=False)

    # Right: empirical safe-lmax vs theta with analytic curve
    ax = axes[1]
    th_fine = np.linspace(7, 40, 200)
    ax.plot(th_fine, [analytic_lcut(t) for t in th_fine], "k--",
            label=r"$\ell_{\rm cut} = 3.83/\theta_1$")
    # Compute empirical safe per theta at pass_threshold
    safe_pts = {}
    for (theta, lmax), shift in lmax_shifts.items():
        if shift > pass_threshold:
            continue
        lm = 1535 if lmax is None else lmax
        safe_pts[theta] = max(safe_pts.get(theta, 0), lm)
    if safe_pts:
        ths, lms = zip(*sorted(safe_pts.items()))
        ax.plot(ths, lms, "o", ms=10, color="C0",
                label=fr"safe $\ell_{{\rm max}}$ (max $|$shift$|\leq{pass_threshold}\sigma$)")
        for t, lm in zip(ths, lms):
            ax.annotate(f"{lm}", (t, lm), textcoords="offset points",
                        xytext=(8, 8), fontsize=10)
    # At theta=20 we showed lmax=600 is borderline at 0.56 sigma; relax-pass at 0.7
    relax_pts = {}
    for (theta, lmax), shift in lmax_shifts.items():
        if shift > 0.7:
            continue
        lm = 1535 if lmax is None else lmax
        relax_pts[theta] = max(relax_pts.get(theta, 0), lm)
    if relax_pts:
        ths, lms = zip(*sorted(relax_pts.items()))
        ax.plot(ths, lms, "s", ms=8, color="C3", mfc="white", mew=1.4,
                label=r"loose pass ($\leq 0.7\sigma$)")
    ax.set_xlabel(r"$\theta_1$ (arcmin)")
    ax.set_ylabel(r"$\ell$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(r"Empirical safe $\ell_{\rm max}$ vs $\theta_1$")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path,
                        default=REPO_ROOT / "outputs" / "plots")
    args = parser.parse_args()

    make_transfer_figure(args.outdir / "doth_transfer_functions.pdf")
    make_safe_regime_figure(args.outdir / "doth_safe_regime.pdf")


if __name__ == "__main__":
    main()
