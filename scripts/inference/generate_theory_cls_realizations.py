#!/usr/bin/env python3
"""
Generate pseudo-realizations of theory C_ell by injecting realistic scatter.

Two covariance approaches:
  A (percosmo) — per-cosmology covariance from ~7 simulation realizations.
                 C_cosmo is rank-deficient; SVD sampling handles it safely.
  B (fidcov)   — constant fiducial covariance from 200 fiducial realizations.
                 Full rank (200 samples, 60 features).

For each unique cosmology (grouped by selected_indices // 7 in the sim file),
`--n-draws` samples are drawn around the pixel-window-corrected theory mean.

Output
------
data/cls/theory/theory_cls_bin4_realizations_percosmo_cov.npz
data/cls/theory/theory_cls_bin4_realizations_fidcov.npz

Fields in each output:
  params       (N_total, 6)
  cls          (N_total, 60)
  ells         (60,)
  ell_edges    (61,) or empty
  cov_approach str
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    repo_cls = REPO_ROOT / "data" / "cls"
    parser = argparse.ArgumentParser(
        description="Generate theory C_ell pseudo-realizations via covariance injection."
    )
    parser.add_argument(
        "--theory-cls-file", type=Path,
        default=repo_cls / "theory" / "theory_cls_bin4_simbin_nside512_pixwin.npz",
        help="NPZ with pixwin-corrected theory cls (params, cls, ells).",
    )
    parser.add_argument(
        "--sim-cls-file", type=Path,
        default=repo_cls / "simulations" / "sim_cls_bin4_nobaryons.npz",
        help="NPZ with sim cls (params, cls, ells, selected_indices) for Approach A.",
    )
    parser.add_argument(
        "--fiducial-cls-file", type=Path,
        default=repo_cls / "simulations" / "sim_cls_bin4_fiducial.npz",
        help="NPZ with fiducial sim cls for Approach B covariance (cls, ells).",
    )
    parser.add_argument(
        "--n-draws", type=int, default=7,
        help="Number of pseudo-realizations per unique cosmology (default: 7).",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-14,
        help="Diagonal regularization added to per-cosmo covariance (default: 1e-14).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=repo_cls / "theory",
        help="Directory for output NPZ files.",
    )
    parser.add_argument(
        "--output-percosmo-file", type=Path, default=None,
        help="Explicit output path for Approach A (percosmo covariance). "
             "Defaults to <output-dir>/theory_cls_bin4_realizations_percosmo_cov.npz.",
    )
    parser.add_argument(
        "--output-fidcov-file", type=Path, default=None,
        help="Explicit output path for Approach B (fiducial covariance). "
             "Defaults to <output-dir>/theory_cls_bin4_realizations_fidcov.npz.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def load_sim_cls(sim_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(sim_file, allow_pickle=True) as d:
        params = np.asarray(d["params"], dtype=float)
        cls = np.asarray(d["cls"], dtype=float)
        ells = np.asarray(d["ells"], dtype=float)
        ell_edges = np.asarray(d["ell_edges"], dtype=float) if "ell_edges" in d else None
        if "selected_indices" in d:
            selected_indices = np.asarray(d["selected_indices"], dtype=int)
        else:
            # Fall back to row index as a proxy for grouping
            selected_indices = np.arange(len(params), dtype=int)
    return params, cls, ells, ell_edges, selected_indices


def load_theory_cls(theory_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(theory_file, allow_pickle=True) as d:
        cls = np.asarray(d["cls"], dtype=float)
        params = np.asarray(d["params"], dtype=float)
    return params, cls


def load_fiducial_cls(fid_file: Path) -> np.ndarray:
    with np.load(fid_file, allow_pickle=True) as d:
        cls = np.asarray(d["cls"], dtype=float)
    valid = ~np.all(np.isnan(cls), axis=1)
    return cls[valid]


def generate_realizations(
    theory_cls: np.ndarray,
    sim_cls: np.ndarray,
    selected_indices: np.ndarray,
    params: np.ndarray,
    n_draws: int,
    eps: float,
    seed: int,
    approach: str,
    fid_cov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw pseudo-realizations for each unique cosmology.

    approach: 'percosmo' or 'fidcov'
    fid_cov: pre-computed fiducial covariance (required for 'fidcov')
    """
    rng = np.random.default_rng(seed)
    cosmo_ids = selected_indices // 7

    unique_cosmo_ids = np.unique(cosmo_ids)
    n_unique = len(unique_cosmo_ids)
    n_features = theory_cls.shape[1]

    out_cls = np.zeros((n_unique * n_draws, n_features), dtype=float)
    out_params = np.zeros((n_unique * n_draws, params.shape[1]), dtype=float)

    for i, cid in enumerate(unique_cosmo_ids):
        mask = cosmo_ids == cid
        theory_mean = theory_cls[mask][0]          # identical for all rows with same cid
        cosmo_params = params[mask][0]

        if approach == "percosmo":
            sim_rows = sim_cls[mask]               # shape (k, 60), k ≈ 7
            if sim_rows.shape[0] < 2:
                # Only 1 row — cannot estimate covariance; use zero scatter
                cov = np.diag(np.full(n_features, eps))
            else:
                cov = np.cov(sim_rows.T) + eps * np.eye(n_features)
        else:
            cov = fid_cov

        draws = rng.multivariate_normal(theory_mean, cov, size=n_draws, method="svd")

        start = i * n_draws
        out_cls[start:start + n_draws] = draws
        out_params[start:start + n_draws] = cosmo_params[None, :]

        if (i + 1) % 50 == 0 or (i + 1) == n_unique:
            print(f"  {i+1}/{n_unique} cosmologies processed")

    return out_params, out_cls


def save_output(
    out_file: Path,
    params: np.ndarray,
    cls: np.ndarray,
    ells: np.ndarray,
    ell_edges: np.ndarray | None,
    approach: str,
    overwrite: bool,
) -> None:
    if out_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_file}. Use --overwrite to replace it.")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        params=params,
        cls=cls,
        ells=ells,
        ell_edges=ell_edges if ell_edges is not None else np.array([], dtype=float),
        cov_approach=np.array(approach),
    )
    np.savez_compressed(out_file, **payload)
    print(f"Saved: {out_file}  (shape: {cls.shape})")


def main() -> None:
    args = parse_args()

    print("Loading simulation cls ...")
    sim_params, sim_cls, ells, ell_edges, selected_indices = load_sim_cls(args.sim_cls_file)
    print(f"  sim cls shape: {sim_cls.shape}, selected_indices: {selected_indices.shape}")

    print("Loading theory cls (pixwin-corrected) ...")
    theory_params, theory_cls = load_theory_cls(args.theory_cls_file)
    print(f"  theory cls shape: {theory_cls.shape}")

    if not np.allclose(sim_params, theory_params, rtol=1e-5):
        raise ValueError("sim and theory params arrays do not match — check input files.")

    if theory_cls.shape != sim_cls.shape:
        raise ValueError(
            f"Shape mismatch: theory {theory_cls.shape} vs sim {sim_cls.shape}."
        )

    print("Loading fiducial cls for Approach B covariance ...")
    fid_cls = load_fiducial_cls(args.fiducial_cls_file)
    print(f"  fiducial cls shape: {fid_cls.shape}")
    n_features = sim_cls.shape[1]
    C_fid = np.cov(fid_cls.T)  # (60, 60), full rank from 200 samples
    print(f"  Fiducial covariance rank: {np.linalg.matrix_rank(C_fid)}/{n_features}")

    cosmo_ids = selected_indices // 7
    n_unique = len(np.unique(cosmo_ids))
    print(f"Unique cosmologies: {n_unique}, n_draws per cosmo: {args.n_draws}")
    print(f"Total output rows: {n_unique * args.n_draws}")

    # --- Approach A: per-cosmology covariance --------------------------------
    print("\n=== Approach A: per-cosmology covariance ===")
    params_a, cls_a = generate_realizations(
        theory_cls=theory_cls,
        sim_cls=sim_cls,
        selected_indices=selected_indices,
        params=sim_params,
        n_draws=args.n_draws,
        eps=args.eps,
        seed=args.seed,
        approach="percosmo",
    )
    out_a = (
        args.output_percosmo_file
        if args.output_percosmo_file is not None
        else args.output_dir / "theory_cls_bin4_realizations_percosmo_cov.npz"
    )
    save_output(out_a, params_a, cls_a, ells, ell_edges, "percosmo", args.overwrite)

    # --- Approach B: fiducial covariance -------------------------------------
    print("\n=== Approach B: fiducial covariance ===")
    params_b, cls_b = generate_realizations(
        theory_cls=theory_cls,
        sim_cls=sim_cls,
        selected_indices=selected_indices,
        params=sim_params,
        n_draws=args.n_draws,
        eps=args.eps,
        seed=args.seed,
        approach="fidcov",
        fid_cov=C_fid,
    )
    out_b = (
        args.output_fidcov_file
        if args.output_fidcov_file is not None
        else args.output_dir / "theory_cls_bin4_realizations_fidcov.npz"
    )
    save_output(out_b, params_b, cls_b, ells, ell_edges, "fidcov", args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()
