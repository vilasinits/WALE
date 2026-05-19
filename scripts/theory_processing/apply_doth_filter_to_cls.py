#!/usr/bin/env python3
"""
Apply DoTH filter and pixel-window correction to theory C_ell files.

The combined transfer function applied to each C_ell bin is:
    C_ell_out = C_ell_in * W_DoTH(ell)^2 * W_pix(ell, nside)^2

where:
    W_DoTH(ell) = B(ell, theta*ratio) - B(ell, theta)
    B(ell, theta) = top-hat beam transfer function (same definition as
                    l1_norm_processing_halofit.py get_beam())
    W_pix(ell, nside) = HEALPix pixel window function at the given nside

This makes theory DoTH C_ells directly comparable to simulation DoTH C_ells
computed from hp.anafast on nside=512 DoTH-filtered maps (which naturally
contain both beam and pixel-window effects).

Usage
-----
# Full grid (1299 cosmologies)
python apply_doth_filter_to_cls.py

# Fiducial only
python apply_doth_filter_to_cls.py --fiducial

# Custom theta
python apply_doth_filter_to_cls.py --theta 20.0 --theta-ratio 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    theory_dir = repo_root / "data" / "cls" / "theory"

    parser = argparse.ArgumentParser(
        description="Apply DoTH filter + pixel-window to theory C_ell NPZ files."
    )
    parser.add_argument(
        "--fiducial", action="store_true",
        help="Use fiducial input/output paths (theory_cls_bin<N>_fiducial.npz).",
    )
    parser.add_argument(
        "--tomo-bin", type=int, default=4,
        help="Tomographic bin used to resolve default fiducial paths (default: 4).",
    )
    parser.add_argument(
        "--input-file", type=Path, default=None,
        help="Input NPZ file. Required unless --fiducial is set.",
    )
    parser.add_argument(
        "--output-file", type=Path, default=None,
        help="Output NPZ path. Auto-derived from input stem if not set.",
    )
    parser.add_argument(
        "--theta", type=float, default=20.0,
        help="Inner top-hat smoothing scale in arcmin (default: 20.0).",
    )
    parser.add_argument(
        "--theta-ratio", type=float, default=2.0,
        help="Ratio theta2/theta1 (default: 2.0).",
    )
    parser.add_argument(
        "--nside", type=int, default=512,
        help="HEALPix nside for pixel-window function (default: 512).",
    )
    parser.add_argument(
        "--lmax", type=int, default=None,
        help="lmax for beam computation (default: 3*nside - 1).",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.fiducial and args.input_file is None:
        args.input_file = theory_dir / f"theory_cls_bin{args.tomo_bin}_fiducial.npz"
    elif args.input_file is None:
        args.input_file = theory_dir / f"theory_cls_bin{args.tomo_bin}_simbin.npz"

    return args


# ---------------------------------------------------------------------------
# Beam helpers (same top-hat definition as l1_norm_processing_halofit.py)
# ---------------------------------------------------------------------------

def compute_beam(theta_arcmin: float, lmax: int) -> np.ndarray:
    """Top-hat beam transfer function B(ℓ) for smoothing scale theta_arcmin."""
    def top_hat(b, radius):
        return np.where(np.abs(b) <= radius, 1 / (np.cos(radius) - 1) / (-2 * np.pi), 0)

    t = theta_arcmin * np.pi / (60 * 180)
    b = np.linspace(0.0, t * 1.2, 10000)
    bw = top_hat(b, t)
    return np.asarray(hp.sphtfunc.beam2bl(bw, b, lmax))


def compute_doth_transfer(
    ells: np.ndarray,
    theta: float,
    theta_ratio: float,
    lmax: int,
) -> np.ndarray:
    """W_DoTH(ell) = B(ell, theta*ratio) - B(ell, theta), evaluated at ell centers."""
    beam1 = compute_beam(theta, lmax)
    beam2 = compute_beam(theta * theta_ratio, lmax)
    w_doth = beam2 - beam1            # shape (lmax+1,)
    ell_idx = np.clip(np.round(ells).astype(int), 0, lmax)
    return w_doth[ell_idx]            # shape (n_ell,)


def compute_pixwin(ells: np.ndarray, nside: int) -> np.ndarray:
    """HEALPix pixel window function W_pix(ell, nside) evaluated at ell centers."""
    pixwin = hp.pixwin(nside)
    max_ell = len(pixwin) - 1
    ell_idx = np.clip(np.round(ells).astype(int), 0, max_ell)
    return pixwin[ell_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    theory_dir = repo_root / "data" / "cls" / "theory"

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    lmax = int(args.lmax if args.lmax is not None else 3 * args.nside - 1)
    theta_tag = f"theta{args.theta:.1f}_ratio{args.theta_ratio:.1f}"

    # Auto-derive output path
    if args.output_file is not None:
        output_file = Path(args.output_file)
    else:
        # e.g. theory_cls_bin4_simbin.npz → theory_doth_cls_bin4_simbin_theta20.0_ratio2.0_nside512_pixwin.npz
        stem = input_file.stem  # e.g. theory_cls_bin4_simbin or theory_cls_bin4_fiducial
        # Replace leading "theory_cls" with "theory_doth_cls"
        new_stem = stem.replace("theory_cls", "theory_doth_cls", 1)
        output_file = theory_dir / f"{new_stem}_{theta_tag}_nside{args.nside}_pixwin.npz"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_file}. Use --overwrite to replace it."
        )

    # Load input
    with np.load(input_file, allow_pickle=True) as d:
        data = {k: d[k] for k in d.keys()}
        ells = np.asarray(d["ells"], dtype=float)
        cls_in = np.asarray(d["cls"], dtype=float)

    if cls_in.ndim == 1:
        cls_in = cls_in[None, :]
    if cls_in.ndim != 2:
        raise ValueError(f"Expected 1D or 2D cls array, got {cls_in.shape}.")
    if cls_in.shape[1] != len(ells):
        raise ValueError(f"cls.shape[1]={cls_in.shape[1]} != len(ells)={len(ells)}.")

    # Compute transfer functions
    w_doth = compute_doth_transfer(ells, args.theta, args.theta_ratio, lmax)
    w_pix = compute_pixwin(ells, args.nside)
    transfer = (w_doth * w_pix) ** 2

    print(f"DoTH transfer function range: [{w_doth.min():.4f}, {w_doth.max():.4f}]")
    print(f"Pixel window range:           [{w_pix.min():.4f}, {w_pix.max():.4f}]")
    print(f"Combined transfer (W²_DoTH × W²_pix) range: [{transfer.min():.4e}, {transfer.max():.4e}]")

    cls_out = cls_in * transfer[None, :]

    # Save
    data["cls"] = cls_out
    data["theta"] = np.float64(args.theta)
    data["theta_ratio"] = np.float64(args.theta_ratio)
    data["pixwin_nside"] = np.int64(args.nside)
    data["doth_lmax"] = np.int64(lmax)
    data["doth_formula"] = np.array(
        "cls_out = cls_in * (B(theta*ratio,ell) - B(theta,ell))^2 * pixwin(nside,ell)^2"
    )
    data["doth_w_doth"] = w_doth
    data["doth_w_pix"] = w_pix
    data["doth_transfer"] = transfer
    data["source_file"] = np.array(str(input_file))

    np.savez_compressed(output_file, **data)
    print(f"Saved: {output_file}")
    print(f"  Input cls shape:  {cls_in.shape}  range [{cls_in.min():.3e}, {cls_in.max():.3e}]")
    print(f"  Output cls shape: {cls_out.shape}  range [{cls_out.min():.3e}, {cls_out.max():.3e}]")


if __name__ == "__main__":
    main()
