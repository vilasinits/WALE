#!/usr/bin/env python3
"""
Compute angular power spectra C_ell from CosmoGRID HEALPix simulation maps
for the Halofit-selected cosmologies.

The output is intentionally aligned with the theory file schema:
  - params
  - param_names
  - ells
  - cls
  - tomo_bin
  - n_cosmo

Additional metadata fields are stored to make provenance explicit.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable
from functools import partial

import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm

PARAM_NAMES = np.array(
    [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
)


def seed_worker() -> None:
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute binned C_ell from Halofit-selected CosmoGRID simulation maps."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_selection_dir = repo_root / "data" / "halofit_cosmo_selection"
    default_output_file = (
        repo_root / "data" / "cls" / "simulations" / "sim_cls_bin4_nobaryons.npz"
    )

    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=default_selection_dir,
        help="Directory containing selected_indices_halofit.npy and selected_params_halofit.npy.",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Optional explicit path to selected_indices_halofit.npy.",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="Optional explicit path to selected_params_halofit.npy.",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base CosmoGRID directory. Defaults to new_grid (nobaryons) "
            "or grid (baryonified)."
        ),
    )
    parser.add_argument(
        "--actual-file",
        type=Path,
        default=None,
        help="Optional explicit path to actual.txt mapping file.",
    )
    parser.add_argument(
        "--baryonified",
        action="store_true",
        help="Use baryonified maps instead of nobaryons maps.",
    )

    parser.add_argument(
        "--tomo-bin",
        type=int,
        default=4,
        help="Tomographic bin number (1-indexed; map key is kg/stage3_lensing<tomo_bin>).",
    )
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Maximum multipole (default: 3*nside).",
    )
    parser.add_argument(
        "--ell-min",
        type=float,
        default=10.0,
        help="Minimum ell for logarithmic binning (default: 10).",
    )
    parser.add_argument(
        "--n-ell-bins",
        type=int,
        default=60,
        help="Number of logarithmic ell bins (default: 60).",
    )

    parser.add_argument(
        "--anafast-iter",
        type=int,
        default=3,
        help="Healpy anafast iter argument (default: 3).",
    )
    parser.add_argument(
        "--use-pixel-weights",
        action="store_true",
        help="Pass use_pixel_weights=True to healpy.anafast.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto-detect physical cores).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Pool chunksize (default: auto).",
    )
    parser.add_argument(
        "--test-n",
        type=int,
        default=None,
        help="Process only the first N selected realizations.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output_file,
        help="Output NPZ file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def load_selected_data(
    selection_file: Path, params_file: Path
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.load(selection_file).astype(int)
    params = np.load(params_file)

    if indices.ndim != 1:
        raise ValueError(f"Expected 1D selection indices, got shape {indices.shape}.")
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"Expected params shape (N, 6), got {params.shape}.")
    if len(indices) != len(params):
        raise ValueError(
            "selected indices and params length mismatch: "
            f"{len(indices)} vs {len(params)}."
        )
    return indices, params


def read_actual_mapping(actual_file: Path) -> list[int]:
    if not actual_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {actual_file}")
    with open(actual_file, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def build_file_records(
    selected_indices: np.ndarray,
    base_dir: Path,
    map_filename: str,
    actual_file: Path,
) -> list[dict[str, object]]:
    actual_cosmo_nums = read_actual_mapping(actual_file)
    records: list[dict[str, object]] = []

    for row_idx, idx in enumerate(selected_indices):
        cosmo_idx = int(idx) // 7
        perm_num = int(idx) % 7
        if cosmo_idx >= len(actual_cosmo_nums):
            records.append(
                {
                    "row_idx": row_idx,
                    "selected_idx": int(idx),
                    "file_path": None,
                    "exists": False,
                    "error": f"cosmo_idx {cosmo_idx} out of range for actual.txt",
                }
            )
            continue

        cosmo_num = actual_cosmo_nums[cosmo_idx]
        file_path = (
            base_dir / f"cosmo_{cosmo_num:06d}" / f"perm_{perm_num:04d}" / map_filename
        )
        exists = file_path.exists()
        records.append(
            {
                "row_idx": row_idx,
                "selected_idx": int(idx),
                "file_path": str(file_path),
                "exists": bool(exists),
                "error": None if exists else f"missing file: {file_path}",
            }
        )
    return records


def build_log_ell_edges(ell_min: float, lmax: int, n_ell_bins: int) -> np.ndarray:
    ell_lo = max(2, int(np.ceil(ell_min)))
    if ell_lo >= lmax:
        raise ValueError(f"ell_min={ell_lo} must be smaller than lmax={lmax}.")
    edges = np.geomspace(float(ell_lo), float(lmax) + 1e-6, n_ell_bins + 1)
    edges[0] = float(ell_lo)
    edges[-1] = float(lmax) + 1e-6
    return edges


def binned_ells_from_edges(ell_edges: np.ndarray) -> np.ndarray:
    n_bins = len(ell_edges) - 1
    ell_centers = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        lo = int(np.ceil(ell_edges[i]))
        hi = int(np.floor(ell_edges[i + 1] - 1e-9))
        if hi < lo:
            ell_centers[i] = np.clip(
                round(np.sqrt(ell_edges[i] * ell_edges[i + 1])), lo, hi + 1
            )
            continue
        l = np.arange(lo, hi + 1)
        w = 2.0 * l + 1.0
        ell_centers[i] = np.average(l, weights=w)
    return ell_centers


def bin_cl_with_edges(cl: np.ndarray, ell_edges: np.ndarray) -> np.ndarray:
    n_bins = len(ell_edges) - 1
    binned = np.full(n_bins, np.nan, dtype=float)
    max_l = len(cl) - 1

    for i in range(n_bins):
        lo = int(np.ceil(ell_edges[i]))
        hi = int(np.floor(ell_edges[i + 1] - 1e-9))
        lo = max(0, lo)
        hi = min(max_l, hi)

        if hi < lo:
            ell0 = int(
                np.clip(round(np.sqrt(ell_edges[i] * ell_edges[i + 1])), 0, max_l)
            )
            binned[i] = float(cl[ell0])
            continue

        l = np.arange(lo, hi + 1)
        w = 2.0 * l + 1.0
        binned[i] = float(np.average(cl[l], weights=w))

    return binned


def _worker_compute_cls(
    record: dict[str, object],
    tomo_bin: int,
    lmax: int,
    ell_edges: np.ndarray,
    anafast_iter: int,
    use_pixel_weights: bool,
) -> tuple[int, np.ndarray | None, str | None]:
    row_idx = int(record["row_idx"])
    if not bool(record["exists"]):
        return row_idx, None, str(record.get("error"))

    file_path = str(record["file_path"])
    map_key = f"kg/stage3_lensing{tomo_bin}"

    try:
        with h5py.File(file_path, "r") as f:
            if map_key not in f:
                return row_idx, None, f"missing dataset {map_key} in {file_path}"
            kappa = np.asarray(f[map_key], dtype=np.float64)

        cl_full = hp.anafast(
            kappa,
            lmax=lmax,
            iter=anafast_iter,
            use_pixel_weights=use_pixel_weights,
        )
        cl_binned = bin_cl_with_edges(cl_full, ell_edges)
        return row_idx, cl_binned, None
    except Exception as exc:
        return row_idx, None, f"{type(exc).__name__}: {exc}"


def _auto_workers() -> int:
    try:
        import psutil

        return max(1, int(psutil.cpu_count(logical=False) or 1))
    except Exception:
        return max(1, mp.cpu_count() // 2)


def _iter_records(
    records: Iterable[dict[str, object]],
    num_workers: int,
    chunksize: int,
    worker,
    total: int,
) -> Iterable[tuple[int, np.ndarray | None, str | None]]:
    if num_workers == 1:
        for rec in records:
            yield worker(rec)
        return

    with mp.Pool(processes=num_workers, initializer=seed_worker) as pool:
        yield from pool.imap_unordered(worker, records, chunksize=chunksize)


def main() -> None:
    args = parse_args()

    selection_dir = Path(args.selection_dir)
    selection_file = args.selection_file or (
        selection_dir / "selected_indices_halofit.npy"
    )
    params_file = args.params_file or (selection_dir / "selected_params_halofit.npy")

    if args.base_dir is not None:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(
            "/home/tersenov/CosmoGridV1/stage3_forecast/grid"
            if args.baryonified
            else "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid"
        )

    map_filename = (
        "projected_probes_maps_baryonified512.h5"
        if args.baryonified
        else "projected_probes_maps_nobaryons512.h5"
    )
    actual_file = args.actual_file or (base_dir / "actual.txt")

    selected_indices, selected_params = load_selected_data(selection_file, params_file)
    if args.test_n is not None:
        n_keep = int(max(1, min(args.test_n, len(selected_indices))))
        selected_indices = selected_indices[:n_keep]
        selected_params = selected_params[:n_keep]

    n_rows = len(selected_indices)
    lmax = int(args.lmax if args.lmax is not None else 3 * int(args.nside))
    ell_edges = build_log_ell_edges(args.ell_min, lmax, args.n_ell_bins)
    ell_centers = binned_ells_from_edges(ell_edges)

    records = build_file_records(selected_indices, base_dir, map_filename, actual_file)
    existing_records = [r for r in records if bool(r["exists"])]
    missing_count = len(records) - len(existing_records)

    if args.num_workers is None:
        num_workers = _auto_workers()
    else:
        num_workers = max(1, int(args.num_workers))
    if args.chunksize is None:
        chunksize = max(1, len(existing_records) // max(1, num_workers * 4))
    else:
        chunksize = max(1, int(args.chunksize))

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_file}. Use --overwrite to replace it."
        )

    if args.verbose:
        print(f"Selection file: {selection_file}")
        print(f"Params file:    {params_file}")
        print(f"Base dir:       {base_dir}")
        print(f"Map file:       {map_filename}")
        print(f"actual.txt:     {actual_file}")
        print(f"Rows selected:  {n_rows}")
        print(f"Rows missing:   {missing_count}")
        print(f"tomo_bin:       {args.tomo_bin}")
        print(f"nside/lmax:     {args.nside}/{lmax}")
        print(f"ell bins:       {args.n_ell_bins}")
        print(f"workers/chunk:  {num_workers}/{chunksize}")

    cls = np.full((n_rows, len(ell_centers)), np.nan, dtype=np.float64)
    valid_mask = np.zeros(n_rows, dtype=bool)
    error_messages = [""] * n_rows

    for rec in records:
        if not bool(rec["exists"]):
            row_idx = int(rec["row_idx"])
            error_messages[row_idx] = str(rec["error"])

    worker = partial(
        _worker_compute_cls,
        tomo_bin=int(args.tomo_bin),
        lmax=lmax,
        ell_edges=ell_edges,
        anafast_iter=int(args.anafast_iter),
        use_pixel_weights=bool(args.use_pixel_weights),
    )

    iterator = _iter_records(
        existing_records,
        num_workers=num_workers,
        chunksize=chunksize,
        worker=worker,
        total=len(existing_records),
    )
    for row_idx, cl_row, err in tqdm(
        iterator, total=len(existing_records), desc="Computing C_ell"
    ):
        if cl_row is None:
            error_messages[row_idx] = err or "unknown error"
            continue
        cls[row_idx] = cl_row
        valid_mask[row_idx] = True

    np.savez_compressed(
        output_file,
        params=selected_params,
        param_names=PARAM_NAMES,
        ells=ell_centers,
        cls=cls,
        tomo_bin=int(args.tomo_bin),
        n_cosmo=int(n_rows),
        selected_indices=selected_indices,
        valid_mask=valid_mask,
        ell_edges=ell_edges,
        nside=int(args.nside),
        lmax=int(lmax),
        map_type=np.array("baryonified" if args.baryonified else "nobaryons"),
        map_filename=np.array(map_filename),
        base_dir=np.array(str(base_dir)),
        n_valid=int(np.sum(valid_mask)),
        n_missing=int(missing_count),
        error_messages=np.array(error_messages, dtype=object),
    )

    print(f"Saved simulation C_ell output: {output_file}")
    print(f"Valid rows: {int(np.sum(valid_mask))}/{n_rows}")
    if missing_count > 0:
        print(f"Missing files: {missing_count}")


if __name__ == "__main__":
    main()
