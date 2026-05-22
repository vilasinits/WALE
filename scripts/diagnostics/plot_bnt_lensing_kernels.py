#!/usr/bin/env python3
"""
BNT-bin-4 lensing kernel diagnostic.

Builds the 4 cosmoGRID lensing kernels q_i(chi) via
`wale.CosmologyModel.Cosmology_function.get_lensing_weight_array_nz` (the same
code path the WALE theory pipeline uses) at the fiducial cosmoGRID cosmology,
then forms the four BNT-rotated kernels:

    q_BNT,i(chi) = sum_j M[i, j] * q_j(chi)

with the cosmoGRID BNT matrix (from bar_impact/notebooks/BNTcp.ipynb).

Two PDFs are produced under outputs/plots/l1_bnt/kernels/:

* bnt_kernels_overview.pdf — left: standard (dashed) and BNT (solid) kernels
  for all 4 bins; right: the same with n(z) shaded behind, to make the
  geometric origin of each kernel visible.
* bnt_bin4_kernel_zoom.pdf — close-up on BNT bin 4 vs standard bin 4 with
  mean-z, FWHM, and the negative-support range annotated.

Run time: ~30 s (single cosmology).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# cosmoGRID 4x4 BNT matrix (from bar_impact/notebooks/BNTcp.ipynb).
BNT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [0.4521097, -1.4521097, 1.0, 0.0],
        [0.0, 0.25127807, -1.251278, 1.0],
    ]
)

# cosmoGRID fiducial cosmology (matches TRUE_PARAMS used elsewhere).
FIDUCIAL = dict(Om=0.26, sigma8=0.84, w0=-1.0, H0=67.36, ns=0.9649, Ob=0.0493)


def fwhm(z: np.ndarray, k: np.ndarray) -> tuple[float, float, float]:
    """Return (z_peak, z_lo, z_hi) where k crosses half its peak."""
    peak = np.max(np.abs(k))
    if peak <= 0:
        return float("nan"), float("nan"), float("nan")
    half = 0.5 * peak
    above = np.abs(k) >= half
    if not above.any():
        return float("nan"), float("nan"), float("nan")
    i0 = np.argmax(above)
    i1 = len(above) - 1 - np.argmax(above[::-1])
    z_peak = z[int(np.argmax(np.abs(k)))]
    return float(z_peak), float(z[i0]), float(z[i1])


def build_kernels() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    from wale.CosmologyModel import Cosmology_function

    h = FIDUCIAL["H0"] / 100.0
    Ob = FIDUCIAL["Ob"]
    Om = FIDUCIAL["Om"]
    Oc = Om - Ob
    cosmo = Cosmology_function(
        h=h,
        Oc=Oc,
        Ob=Ob,
        w=FIDUCIAL["w0"],
        wa=0.0,
        sigma8=FIDUCIAL["sigma8"],
        ns=FIDUCIAL["ns"],
        dk=0.001,
        kmin=1e-3,
        kmax=1.0,
    )

    nz_files = [REPO_ROOT / "data" / "nz" / f"nz_stage3_{i}_GRID.txt" for i in range(1, 5)]
    nzs: list[tuple[np.ndarray, np.ndarray]] = []
    for f in nz_files:
        data = np.loadtxt(f)
        nzs.append((data[:, 0], data[:, 1]))

    z_max_global = max(nz[0][-1] for nz in nzs)
    chi_source = cosmo.get_chi(z_max_global)
    chis = np.linspace(100.0, chi_source - 1.0, 400)
    z_of_chi = np.array(
        [float(getattr(cosmo.get_z_from_chi(c), "root", cosmo.get_z_from_chi(c))) for c in chis]
    )

    kernels = np.zeros((4, chis.size))
    for i, (z_nz, n_z) in enumerate(nzs):
        _, w = cosmo.get_lensing_weight_array_nz(chis, z_nz, n_z)
        kernels[i] = w

    bnt = BNT_MATRIX @ kernels
    return chis, z_of_chi, kernels, nzs, bnt


def plot_overview(z_of_chi: np.ndarray, kernels: np.ndarray, bnt: np.ndarray,
                  nzs: list[tuple[np.ndarray, np.ndarray]], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax in axes:
        for i in range(4):
            ax.plot(z_of_chi, kernels[i], "--", color=colors[i], lw=1.3,
                    alpha=0.85, label=f"standard bin {i + 1}" if ax is axes[0] else None)
            ax.plot(z_of_chi, bnt[i], "-", color=colors[i], lw=2.0,
                    label=f"BNT bin {i + 1}" if ax is axes[0] else None)
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_xlabel("redshift  z")
        ax.set_ylabel(r"lensing weight  $q(z)$")
        ax.set_xlim(0, 2.5)
        ax.grid(alpha=0.3)

    # Right panel: n(z) underneath
    ax_r = axes[1]
    ax_n = ax_r.twinx()
    for i, (z_nz, n_z) in enumerate(nzs):
        ax_n.fill_between(z_nz, 0, n_z, alpha=0.10, color=colors[i])
    ax_n.set_ylabel(r"$n(z)$  (per bin, unnormalised)", color="0.4")
    ax_n.tick_params(axis="y", colors="0.4")

    axes[0].set_title("Standard vs BNT lensing kernels (cosmoGRID, fiducial)")
    axes[1].set_title("Same, with per-bin n(z) shaded")
    axes[0].legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


def plot_bin4_zoom(z_of_chi: np.ndarray, kernels: np.ndarray, bnt: np.ndarray,
                   outpath: Path) -> None:
    std4 = kernels[3]
    bnt4 = bnt[3]

    z_peak_std, z_lo_std, z_hi_std = fwhm(z_of_chi, std4)
    z_peak_bnt, z_lo_bnt, z_hi_bnt = fwhm(z_of_chi, bnt4)

    neg_mask = bnt4 < 0
    if neg_mask.any():
        z_neg = z_of_chi[neg_mask]
        z_neg_lo, z_neg_hi = float(z_neg.min()), float(z_neg.max())
    else:
        z_neg_lo = z_neg_hi = float("nan")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(z_of_chi, std4, "--", color="#1f77b4", lw=1.8, label="standard bin 4")
    ax.plot(z_of_chi, bnt4, "-", color="#d62728", lw=2.2, label="BNT bin 4")
    ax.axhline(0.0, color="0.5", lw=0.6)
    if neg_mask.any():
        ax.axvspan(z_neg_lo, z_neg_hi, color="#d62728", alpha=0.08,
                   label=f"BNT bin 4 negative support  ({z_neg_lo:.2f}–{z_neg_hi:.2f})")

    ax.axvline(z_peak_std, color="#1f77b4", lw=0.8, ls=":")
    ax.axvline(z_peak_bnt, color="#d62728", lw=0.8, ls=":")

    txt = (
        f"standard bin 4: peak z={z_peak_std:.2f}, FWHM z∈[{z_lo_std:.2f}, {z_hi_std:.2f}]\n"
        f"BNT bin 4:      peak z={z_peak_bnt:.2f}, FWHM z∈[{z_lo_bnt:.2f}, {z_hi_bnt:.2f}]"
    )
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="0.7"))

    ax.set_xlabel("redshift  z")
    ax.set_ylabel(r"lensing weight  $q_4(z)$")
    ax.set_xlim(0, 2.5)
    ax.set_title("BNT bin 4 vs standard bin 4 lensing kernel")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")

    print("\nSummary statistics:")
    print(f"  standard bin 4: peak z={z_peak_std:.3f}, FWHM=({z_lo_std:.3f}, {z_hi_std:.3f})")
    print(f"  BNT  bin 4: peak z={z_peak_bnt:.3f}, FWHM=({z_lo_bnt:.3f}, {z_hi_bnt:.3f})")
    if neg_mask.any():
        print(f"  BNT bin 4 negative support: z∈({z_neg_lo:.3f}, {z_neg_hi:.3f})  "
              f"(min q = {bnt4.min():.3e})")
    print(f"  BNT bin 4 peak / standard bin 4 peak  = {np.max(bnt4) / np.max(std4):.3f}")


def main() -> None:
    chis, z_of_chi, kernels, nzs, bnt = build_kernels()

    outdir = REPO_ROOT / "outputs" / "plots" / "l1_bnt" / "kernels"
    plot_overview(z_of_chi, kernels, bnt, nzs, outdir / "bnt_kernels_overview.pdf")
    plot_bin4_zoom(z_of_chi, kernels, bnt, outdir / "bnt_bin4_kernel_zoom.pdf")


if __name__ == "__main__":
    main()
