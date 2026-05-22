#!/usr/bin/env python3
"""
Compute the LDT theory wavelet ℓ₁-norm for the cosmoGRID Halofit grid.

Consumes the packed sim NPZ produced by `pack_l1_npz.py` so the theory uses the
*same* kappa binning as the simulation (essential for direct comparison and
shared NPE inputs).

Output NPZ schema (uniform with the sim packed NPZ — extra theory-only fields
are added):

    params (N, 6)
    param_names (6,)
    selected_indices (N,)
    kappa_bins (nbins,)        # shared 1-D
    l1_norms (N, nbins)        # the LDT theory ell_1 datavector
    pdf_theory (N, nbins)
    variances (N,)             # variance copied from the sim file (input)
    variance_ldt (N,)
    recal_value (N,)
    theta, theta_ratio, tomo_bin (scalars)

Usage:
    conda run -n wale python scripts/theory_processing/compute_theory_l1_halofit.py \\
        --sim-npz data/l1/simulations/sim_doth_l1_bin4_theta30.0_ratio2.0_nobaryons.npz \\
        --nz-file data/nz/nz_stage3_4_GRID.txt \\
        --output  data/l1/theory/theory_doth_l1_bin4_theta30.0_ratio2.0_simbin.npz \\
        --n-jobs  50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-npz", type=Path, required=True,
                   help="Packed sim NPZ containing params, kappa_bins, l1_norms, variances.")
    p.add_argument("--nz-file", type=Path, default=None,
                   help="Path to n(z) for the tomographic bin. Default: data/nz/nz_stage3_{tomo_bin}_GRID.txt")
    p.add_argument("--theta", type=float, default=None,
                   help="Theta1 in arcmin. Default: read from sim-npz.")
    p.add_argument("--theta-ratio", type=float, default=None,
                   help="Theta ratio. Default: read from sim-npz.")
    p.add_argument("--tomo-bin", type=int, default=None,
                   help="Tomographic bin. Default: read from sim-npz.")
    p.add_argument("--filter-type", type=str, default="tophat", choices=["tophat", "starlet"])
    p.add_argument("--no-recal", action="store_true",
                   help="Disable the σ²_LDT/σ²_sim recalibration; use raw LDT variance.")
    p.add_argument("--n-jobs", type=int, default=50)
    p.add_argument("--backend", type=str, default="loky")
    p.add_argument("--ngrid-critical", type=int, default=5)
    p.add_argument("--nlambdas", type=int, default=61)
    p.add_argument("--nplanes", type=int, default=69)
    p.add_argument("--verbose", type=int, default=5)
    p.add_argument("--n-cosmo", type=int, default=None,
                   help="Limit number of cosmologies (default: all rows).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Encourage joblib processes not to oversubscribe BLAS / OpenMP / XLA.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1 "
        "inter_op_parallelism_threads=1",
    )

    from wale.cosmogrid_fulldv import FullDVConfig, run_cosmogrid_fulldv

    if not args.sim_npz.exists():
        raise FileNotFoundError(args.sim_npz)

    with np.load(args.sim_npz, allow_pickle=True) as d:
        params = np.asarray(d["params"], dtype=float)
        selected_indices = np.asarray(d["selected_indices"], dtype=int)
        kappa_bins = np.asarray(d["kappa_bins"], dtype=float)        # (nbins,)
        l1_norms_sim = np.asarray(d["l1_norms"], dtype=float)        # (N, nbins)
        variances = np.asarray(d["variances"], dtype=float)          # (N,)
        sim_theta = float(d["theta"])
        sim_theta_ratio = float(d["theta_ratio"])
        sim_tomo_bin = int(d["tomo_bin"])
        param_names = np.asarray(d["param_names"]) if "param_names" in d.files else None

    theta1 = args.theta if args.theta is not None else sim_theta
    theta_ratio = args.theta_ratio if args.theta_ratio is not None else sim_theta_ratio
    tomo_bin = args.tomo_bin if args.tomo_bin is not None else sim_tomo_bin
    nz_file = args.nz_file or (REPO_ROOT / "data" / "nz" / f"nz_stage3_{tomo_bin}_GRID.txt")
    if not Path(nz_file).exists():
        raise FileNotFoundError(f"n(z) file not found: {nz_file}")

    N, nbins = l1_norms_sim.shape

    # Broadcast the shared kappa grid to (N, nbins) for the theory call.
    kappa_sim = np.broadcast_to(kappa_bins[None, :], (N, nbins)).copy()

    if args.n_cosmo is not None and args.n_cosmo < N:
        params = params[: args.n_cosmo]
        kappa_sim = kappa_sim[: args.n_cosmo]
        l1_norms_sim = l1_norms_sim[: args.n_cosmo]
        variances = variances[: args.n_cosmo]
        selected_indices = selected_indices[: args.n_cosmo]
        N = args.n_cosmo

    config = FullDVConfig(
        theta1=float(theta1),
        nz_file=str(nz_file),
        nplanes=int(args.nplanes),
        nlambdas=int(args.nlambdas),
        ngrid_critical=int(args.ngrid_critical),
        filter_type=args.filter_type,
        disable_recal=bool(args.no_recal),
    )

    print(f"Running LDT theory ell_1 for {N} cosmologies")
    print(f"  theta1={theta1}, ratio={theta_ratio}, tomo_bin={tomo_bin}, nz={nz_file}")
    print(f"  kappa range [{kappa_bins.min():.4e}, {kappa_bins.max():.4e}], nbins={nbins}")
    print(f"  n_jobs={args.n_jobs}, backend={args.backend}")

    outputs = run_cosmogrid_fulldv(
        params=params,
        kappa_sim=kappa_sim,
        l1_norms_sim=l1_norms_sim,
        variance_sim=variances,
        config=config,
        n_cosmo=N,
        n_jobs=args.n_jobs,
        backend=args.backend,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace.")

    # outputs['kappa_bins'] is (N, nbins) — collapse to 1-D since rows are identical.
    kb = np.asarray(outputs["kappa_bins"])
    if kb.ndim == 2:
        if not np.allclose(kb, kb[0:1, :]):
            raise ValueError("Theory kappa_bins rows are not identical — unexpected.")
        kb_1d = kb[0]
    else:
        kb_1d = kb

    save_kwargs = dict(
        params=outputs["params"],
        param_names=param_names if param_names is not None else np.array([]),
        selected_indices=selected_indices,
        kappa_bins=kb_1d.astype(float),
        l1_norms=np.asarray(outputs["l1_theory"], dtype=float),
        pdf_theory=np.asarray(outputs["pdf_theory"], dtype=float),
        variances=variances.astype(float),
        variance_ldt=np.asarray(outputs["variance_ldt"], dtype=float),
        recal_value=np.asarray(outputs["recal_value"], dtype=float),
        theta=float(theta1),
        theta_ratio=float(theta_ratio),
        tomo_bin=int(tomo_bin),
        filter_type=np.array(args.filter_type),
    )
    np.savez_compressed(args.output, **save_kwargs)
    print(f"Saved theory ell_1 grid: {args.output}")
    print(f"  shape l1_norms = {save_kwargs['l1_norms'].shape}")
    print(f"  recal_value mean={save_kwargs['recal_value'].mean():.3e}, "
          f"min={save_kwargs['recal_value'].min():.3e}, "
          f"max={save_kwargs['recal_value'].max():.3e}")


if __name__ == "__main__":
    main()
