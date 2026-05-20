#!/usr/bin/env python3
"""
Combine the three .npy companions from `l1_norm_processing_halofit.py` into a
single NPZ matching the C_ell NPZ schema (so downstream SBI scripts can consume
it uniformly).

Inputs (defaults match the convention used in Step 1 of the L1 plan):
    {base}_l1.npy       shape (N, nbins)
    {base}_kappa.npy    shape (N, nbins) — must have all rows identical
    {base}_variance.npy shape (N,)

Output:
    {output}.npz with fields:
        params (N, 6)
        param_names (6,)
        selected_indices (N,)        [only for halofit grid]
        kappa_bins (nbins,)          [shared 1-D]
        l1_norms (N, nbins)
        variances (N,)
        theta, theta_ratio, tomo_bin (scalars)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

PARAM_NAMES = np.array(
    [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True,
                   help="Base path used in --combined-output, e.g. data/l1/simulations/raw_sim_doth_l1_bin4_theta30.0_ratio2.0_nobaryons.npy")
    p.add_argument("--params-file", type=Path, default=None,
                   help="Selected cosmology params .npy. Required unless --fiducial.")
    p.add_argument("--selected-indices-file", type=Path, default=None,
                   help="Selected indices .npy (1299,). Required unless --fiducial.")
    p.add_argument("--fiducial", action="store_true",
                   help="Treat as fiducial-cosmology pack (200 perms of the fiducial cosmo).")
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--theta-ratio", type=float, default=2.0)
    p.add_argument("--tomo-bin", type=int, default=4)
    p.add_argument("--output", type=Path, required=True,
                   help="Output NPZ path.")
    p.add_argument("--fiducial-params", type=float, nargs=6,
                   default=[0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493],
                   help="Fiducial cosmology parameters [Om, sigma8, w0, H0, ns, Ob].")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve the three companion paths from the base.
    base, ext = (args.base.with_suffix(""), args.base.suffix or ".npy")
    l1_path = base.parent / f"{base.name}_l1{ext}"
    kappa_path = base.parent / f"{base.name}_kappa{ext}"
    var_path = base.parent / f"{base.name}_variance{ext}"
    for pth in [l1_path, kappa_path, var_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing companion: {pth}")

    l1_norms = np.load(l1_path)              # (N, nbins)
    all_kappas = np.load(kappa_path)         # (N, nbins)
    variances = np.load(var_path)            # (N,)

    if l1_norms.ndim != 2:
        raise ValueError(f"Expected 2-D l1, got {l1_norms.shape}")
    N, nbins = l1_norms.shape
    if all_kappas.shape != l1_norms.shape:
        raise ValueError(f"Kappa shape {all_kappas.shape} != l1 shape {l1_norms.shape}")
    if variances.shape != (N,):
        raise ValueError(f"Variance shape {variances.shape} != (N,)=({N},)")

    # Sanity: all kappa rows must be identical (shared grid).
    if not np.allclose(all_kappas, all_kappas[0:1, :], rtol=0, atol=0):
        max_dev = np.abs(all_kappas - all_kappas[0:1, :]).max()
        raise ValueError(
            f"Kappa rows not identical across the grid (max dev {max_dev}). "
            "Re-run l1_norm_processing_halofit.py with explicit --kappa-min/--kappa-max."
        )
    kappa_bins = all_kappas[0]  # (nbins,)

    if args.fiducial:
        params = np.tile(np.asarray(args.fiducial_params, dtype=float), (N, 1))
        selected_indices = np.arange(N, dtype=int)
    else:
        if args.params_file is None or args.selected_indices_file is None:
            raise ValueError(
                "Halofit grid pack requires --params-file and --selected-indices-file."
            )
        params = np.load(args.params_file)
        selected_indices = np.load(args.selected_indices_file)
        if params.shape[0] != N:
            raise ValueError(f"params has {params.shape[0]} rows, l1 has {N}.")
        if selected_indices.shape[0] != N:
            raise ValueError(
                f"selected_indices has {selected_indices.shape[0]} rows, l1 has {N}."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace.")

    np.savez_compressed(
        args.output,
        params=params.astype(float),
        param_names=PARAM_NAMES,
        selected_indices=selected_indices.astype(int),
        kappa_bins=kappa_bins.astype(float),
        l1_norms=l1_norms.astype(float),
        variances=variances.astype(float),
        theta=float(args.theta),
        theta_ratio=float(args.theta_ratio),
        tomo_bin=int(args.tomo_bin),
    )
    print(f"Packed: {args.output}")
    print(f"  rows={N}, nbins={nbins}, kappa range [{kappa_bins.min():.4e}, {kappa_bins.max():.4e}]")


if __name__ == "__main__":
    main()
