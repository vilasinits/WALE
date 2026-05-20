#!/usr/bin/env python3
"""
Generate cov-injected pseudo-realizations of an ℓ₁-norm (or any binned) data
vector. Drops in for the L1 SBI step, mirroring the C_ℓ
`generate_theory_cls_realizations.py`.

For each unique cosmology (grouped via `selected_indices // 7`):
    1. Compute a per-cosmology *mean* data vector:
         - mean-mode 'as-is':         use the row as-is (theory: all 7 perms of a
                                       given cosmo are identical).
         - mean-mode 'average-perms': average the 7 (or however many) perms of
                                       that cosmology.
    2. Draw n-draws pseudo-realizations from N(mean, Cov_fid), where Cov_fid is
       the (nbins, nbins) covariance estimated from the 200 fiducial L1
       realizations. Sampling is via numpy.random.default_rng().multivariate_normal
       with method='svd' (robust to mild rank deficiency).

Output NPZ schema (uniform with the sim packed NPZ):
    params (n_unique * n_draws, 6)
    param_names (6,)
    selected_indices (n_unique * n_draws,)   # ranges to ease identification
    kappa_bins (nbins,)
    l1_norms (n_unique * n_draws, nbins)
    theta, theta_ratio, tomo_bin (scalars)
    cov_approach: "fidcov"
    mean_mode: as-is | average-perms
    n_draws_per_cosmo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mean-source", type=Path, required=True,
                   help="Input NPZ (sim or theory) with fields params, selected_indices, kappa_bins, l1_norms.")
    p.add_argument("--cov-source", type=Path, required=True,
                   help="NPZ with the fiducial L1 realizations (~200 rows) for covariance estimation.")
    p.add_argument("--mean-mode", choices=["as-is", "average-perms"], required=True)
    p.add_argument("--data-key", type=str, default="l1_norms",
                   help="Field name to read as the data vector (default l1_norms).")
    p.add_argument("--n-draws", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=0.0,
                   help="Diagonal regularisation added to fiducial cov (default 0).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with np.load(args.mean_source, allow_pickle=True) as d:
        params = np.asarray(d["params"], dtype=float)
        selected_indices = np.asarray(d["selected_indices"], dtype=int)
        kappa_bins = np.asarray(d["kappa_bins"], dtype=float)
        l1_in = np.asarray(d[args.data_key], dtype=float)
        param_names = np.asarray(d["param_names"]) if "param_names" in d.files else np.array([])
        theta = float(d["theta"])
        theta_ratio = float(d["theta_ratio"])
        tomo_bin = int(d["tomo_bin"])

    with np.load(args.cov_source, allow_pickle=True) as d:
        l1_fid = np.asarray(d[args.data_key], dtype=float)
        kappa_bins_fid = np.asarray(d["kappa_bins"], dtype=float)

    if not np.allclose(kappa_bins, kappa_bins_fid):
        raise ValueError("kappa_bins of --mean-source and --cov-source must match.")

    N, nbins = l1_in.shape

    # Compute per-unique-cosmo mean
    cosmo_ids = selected_indices // 7
    unique_ids = np.unique(cosmo_ids)
    n_unique = unique_ids.size

    if args.mean_mode == "as-is":
        # Each row stands alone — but rows that share a cosmo_id should be near-identical.
        # We still group to one mean per unique cosmo (just by picking the first row).
        means = np.zeros((n_unique, nbins))
        params_per_cosmo = np.zeros((n_unique, params.shape[1]))
        for j, cid in enumerate(unique_ids):
            sel = np.where(cosmo_ids == cid)[0]
            block = l1_in[sel]
            # Diagnostic check: rows within a cosmo are nearly identical (theory)
            scale = np.maximum(np.abs(block).max(), 1e-30)
            max_dev = np.abs(block - block[0:1]).max() / scale
            if max_dev > 1e-3:
                print(f"  [warn] cosmo_id={cid}: rows differ by max relative {max_dev:.2e}")
            means[j] = block[0]
            params_per_cosmo[j] = params[sel[0]]
    elif args.mean_mode == "average-perms":
        means = np.zeros((n_unique, nbins))
        params_per_cosmo = np.zeros((n_unique, params.shape[1]))
        for j, cid in enumerate(unique_ids):
            sel = np.where(cosmo_ids == cid)[0]
            means[j] = l1_in[sel].mean(axis=0)
            params_per_cosmo[j] = params[sel[0]]
    else:
        raise ValueError(args.mean_mode)

    # Fiducial covariance
    if l1_fid.ndim != 2:
        raise ValueError(f"Fiducial data must be 2-D, got {l1_fid.shape}")
    cov = np.cov(l1_fid.T)
    if args.eps > 0:
        cov = cov + args.eps * np.eye(nbins)
    rank = np.linalg.matrix_rank(cov)
    print(f"Fiducial cov rank: {rank}/{nbins}, n_fid={l1_fid.shape[0]}")

    # Draw n_draws per unique cosmology
    rng = np.random.default_rng(args.seed)
    total = n_unique * args.n_draws
    out_data = np.zeros((total, nbins), dtype=float)
    out_params = np.zeros((total, params.shape[1]), dtype=float)
    out_indices = np.zeros(total, dtype=int)

    for j, cid in enumerate(unique_ids):
        draws = rng.multivariate_normal(means[j], cov, size=args.n_draws, method="svd")
        sl = slice(j * args.n_draws, (j + 1) * args.n_draws)
        out_data[sl] = draws
        out_params[sl] = params_per_cosmo[j][None, :]
        # synthetic selected_indices: cid*7 + k for k in [0..n_draws-1]
        out_indices[sl] = cid * 7 + np.arange(args.n_draws)
        if (j + 1) % 50 == 0 or (j + 1) == n_unique:
            print(f"  {j+1}/{n_unique} cosmologies processed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace.")

    np.savez_compressed(
        args.output,
        params=out_params,
        param_names=param_names,
        selected_indices=out_indices,
        kappa_bins=kappa_bins,
        l1_norms=out_data,
        theta=theta,
        theta_ratio=theta_ratio,
        tomo_bin=tomo_bin,
        cov_approach=np.array("fidcov"),
        mean_mode=np.array(args.mean_mode),
        n_draws_per_cosmo=int(args.n_draws),
    )
    print(f"Saved {args.output}  shape={out_data.shape}, n_unique={n_unique}, n_draws={args.n_draws}")


if __name__ == "__main__":
    main()
