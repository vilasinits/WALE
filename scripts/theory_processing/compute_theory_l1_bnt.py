#!/usr/bin/env python3
"""
Compute the LDT theory wavelet ℓ₁-norm for **BNT-rotated** tomographic bins on
the cosmoGRID Halofit grid.

This is the BNT analogue of `compute_theory_l1_halofit.py`. The only difference
is in how the lensing kernel is built: `FullDVConfig` is given `bnt_nz_files`
(4 cosmoGRID n(z) text files) and `bnt_coeffs` (one row of the BNT matrix), and
`InitialiseVariables` linearly combines the 4 standard kernels with those
coefficients.

CPU usage is hard-capped at HARD_CPU_CAP=40 workers.

Usage:
    conda run -n wale python scripts/theory_processing/compute_theory_l1_bnt.py \\
        --sim-npz data/l1_bnt/simulations/sim_doth_l1_bnt4_bin4_theta30.0_ratio2.0_nobaryons.npz \\
        --output  data/l1_bnt/theory/theory_doth_l1_bnt4_bin4_theta30.0_ratio2.0_simbin.npz \\
        --n-jobs  40
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

HARD_CPU_CAP = 40

# cosmoGRID 4x4 BNT matrix rows. Indexed 0..3 → BNT bin 1..4.
BNT_ROWS = (
    (1.0, 0.0, 0.0, 0.0),
    (-1.0, 1.0, 0.0, 0.0),
    (0.4521097, -1.4521097, 1.0, 0.0),
    (0.0, 0.25127807, -1.251278, 1.0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-npz", type=Path, required=True,
                   help="Packed BNT sim NPZ produced by pack_l1_npz.py.")
    p.add_argument("--bnt-bin", type=int, default=4, choices=(1, 2, 3, 4),
                   help="1-indexed BNT bin (selects the row of BNT_ROWS).")
    p.add_argument("--theta", type=float, default=None,
                   help="Theta1 in arcmin. Default: read from sim-npz.")
    p.add_argument("--theta-ratio", type=float, default=None,
                   help="Theta ratio. Default: read from sim-npz.")
    p.add_argument("--filter-type", type=str, default="tophat",
                   choices=["tophat", "starlet"])
    p.add_argument("--no-recal", action="store_true",
                   help="Disable the σ²_LDT/σ²_sim recalibration; use raw LDT variance.")
    p.add_argument("--n-jobs", type=int, default=HARD_CPU_CAP,
                   help=f"joblib n_jobs; hard-capped at {HARD_CPU_CAP}.")
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

    # Cap per-worker thread spawning. With joblib loky launching N workers,
    # the BLAS/OpenMP/XLA stacks inside each worker can each spawn their own
    # thread pool — so a "n_jobs=40" run can balloon to >100 active threads
    # without these caps in place.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # JAX/XLA-specific thread limits (must be set BEFORE jax is imported in
    # workers; we set via env so all loky child processes inherit them).
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
        kappa_bins = np.asarray(d["kappa_bins"], dtype=float)
        l1_norms_sim = np.asarray(d["l1_norms"], dtype=float)
        variances = np.asarray(d["variances"], dtype=float)
        sim_theta = float(d["theta"])
        sim_theta_ratio = float(d["theta_ratio"])
        sim_tomo_bin = int(d["tomo_bin"])
        param_names = np.asarray(d["param_names"]) if "param_names" in d.files else None

    theta1 = args.theta if args.theta is not None else sim_theta
    theta_ratio = args.theta_ratio if args.theta_ratio is not None else sim_theta_ratio
    bnt_bin = args.bnt_bin
    # The sim NPZ stores tomo_bin = BNT bin number (we set this in the pack step
    # at the user's request — the NPZ semantically holds BNT bin 4 data).
    if sim_tomo_bin != bnt_bin:
        print(f"WARNING: --bnt-bin {bnt_bin} ≠ sim NPZ tomo_bin {sim_tomo_bin}; "
              f"using --bnt-bin (will write tomo_bin={bnt_bin}).")

    bnt_nz_files = tuple(
        str(REPO_ROOT / "data" / "nz" / f"nz_stage3_{i}_GRID.txt") for i in range(1, 5)
    )
    for f in bnt_nz_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"BNT n(z) file not found: {f}")
    bnt_coeffs = BNT_ROWS[bnt_bin - 1]
    print(f"BNT bin {bnt_bin}: coeffs = {bnt_coeffs}")

    N, nbins = l1_norms_sim.shape
    kappa_sim = np.broadcast_to(kappa_bins[None, :], (N, nbins)).copy()

    if args.n_cosmo is not None and args.n_cosmo < N:
        params = params[: args.n_cosmo]
        kappa_sim = kappa_sim[: args.n_cosmo]
        l1_norms_sim = l1_norms_sim[: args.n_cosmo]
        variances = variances[: args.n_cosmo]
        selected_indices = selected_indices[: args.n_cosmo]
        N = args.n_cosmo

    n_jobs = max(1, min(int(args.n_jobs), HARD_CPU_CAP))
    print(f"Workers (n_jobs): {n_jobs}  (hard cap: {HARD_CPU_CAP})")

    # nz_file is used only for provenance / metadata in this mode — pass the
    # BNT-bin-{N} file so the printed and stored value is informative.
    provenance_nz = str(REPO_ROOT / "data" / "nz" / f"nz_stage3_{bnt_bin}_GRID.txt")

    config = FullDVConfig(
        theta1=float(theta1),
        nz_file=provenance_nz,
        nplanes=int(args.nplanes),
        nlambdas=int(args.nlambdas),
        ngrid_critical=int(args.ngrid_critical),
        filter_type=args.filter_type,
        disable_recal=bool(args.no_recal),
        bnt_nz_files=bnt_nz_files,
        bnt_coeffs=bnt_coeffs,
    )

    print(f"Running LDT theory ell_1 for {N} cosmologies — BNT bin {bnt_bin}")
    print(f"  theta1={theta1}, ratio={theta_ratio}")
    print(f"  kappa range [{kappa_bins.min():.4e}, {kappa_bins.max():.4e}], nbins={nbins}")
    print(f"  backend={args.backend}, no_recal={args.no_recal}")

    outputs = run_cosmogrid_fulldv(
        params=params,
        kappa_sim=kappa_sim,
        l1_norms_sim=l1_norms_sim,
        variance_sim=variances,
        config=config,
        n_cosmo=N,
        n_jobs=n_jobs,
        backend=args.backend,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace.")

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
        tomo_bin=int(bnt_bin),
        filter_type=np.array(args.filter_type),
        bnt_bin=int(bnt_bin),
        bnt_coeffs=np.asarray(bnt_coeffs, dtype=float),
    )
    np.savez_compressed(args.output, **save_kwargs)
    print(f"Saved theory ell_1 grid: {args.output}")
    print(f"  shape l1_norms = {save_kwargs['l1_norms'].shape}")
    print(f"  recal_value mean={save_kwargs['recal_value'].mean():.3e}, "
          f"min={save_kwargs['recal_value'].min():.3e}, "
          f"max={save_kwargs['recal_value'].max():.3e}")


if __name__ == "__main__":
    main()
