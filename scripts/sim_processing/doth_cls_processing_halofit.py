#!/usr/bin/env python3
"""
Compute DoTH-filtered angular power spectra C_ell from CosmoGRID HEALPix maps
for the Halofit-selected cosmologies.

The DoTH (Difference of Top-Hats) filter is applied in spherical harmonic space
before computing the power spectrum:
    kappa_DoTH = kappa_smooth(theta*theta_ratio) - kappa_smooth(theta)
    C_ell_DoTH = C_ell[ hp.anafast(kappa_DoTH) ]

Output schema is identical to sim_cls files:
  params, param_names, ells, cls, tomo_bin, ell_edges, selected_indices, ...
plus filter metadata: theta, theta_ratio, lmax_factor.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import Iterable

import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm


PARAM_NAMES = np.array(
    [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
)
FIDUCIAL_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
FIDUCIAL_DIR = Path("/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial")
FIDUCIAL_N_PERMS = 200


def seed_worker() -> None:
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


# ---------------------------------------------------------------------------
# DoTH beam / smoothing (self-contained; mirrors l1_norm_processing_halofit.py)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _get_beam_cached(theta: float, lmax: int) -> tuple:
    """Top-hat beam transfer function B(ℓ) for a given smoothing scale.

    theta : smoothing scale in arcminutes
    lmax  : maximum multipole
    Returns array of length lmax+1 (cached as tuple for hashability).
    """
    def top_hat(b, radius):
        return np.where(np.abs(b) <= radius, 1 / (np.cos(radius) - 1) / (-2 * np.pi), 0)

    t = theta * np.pi / (60 * 180)        # arcmin → radians
    b = np.linspace(0.0, t * 1.2, 10000)
    bw = top_hat(b, t)
    beam = hp.sphtfunc.beam2bl(bw, b, lmax)
    return tuple(beam)


def get_beam(theta: float, lmax: int) -> np.ndarray:
    return np.asarray(_get_beam_cached(theta, lmax))


def smooth_map_dual(
    kappa_map: np.ndarray,
    theta1: float,
    theta2: float,
    nside: int = 512,
    lmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth kappa at two scales using a single alm transform."""
    if lmax is None:
        lmax = 3 * nside - 1
    almkappa = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, use_pixel_weights=True)
    beam1 = get_beam(theta1, lmax)
    beam2 = get_beam(theta2, lmax)
    kappa_s1 = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam1), nside, lmax=lmax)
    kappa_s2 = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam2), nside, lmax=lmax)
    return kappa_s1, kappa_s2


# ---------------------------------------------------------------------------
# Ell binning (identical to cls_processing_halofit.py)
# ---------------------------------------------------------------------------

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
            ell0 = int(np.clip(round(np.sqrt(ell_edges[i] * ell_edges[i + 1])), 0, max_l))
            binned[i] = float(cl[ell0])
            continue
        l = np.arange(lo, hi + 1)
        w = 2.0 * l + 1.0
        binned[i] = float(np.average(cl[l], weights=w))
    return binned


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DoTH-filtered binned C_ell from Halofit-selected or fiducial "
            "CosmoGRID simulation maps."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_selection_dir = repo_root / "data" / "halofit_cosmo_selection"

    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology (200 perms) instead of Halofit grid.")
    parser.add_argument("--fiducial-dir", type=Path, default=FIDUCIAL_DIR)
    parser.add_argument("--n-perms", type=int, default=FIDUCIAL_N_PERMS)

    parser.add_argument("--selection-dir", type=Path, default=default_selection_dir)
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--params-file", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--actual-file", type=Path, default=None)
    parser.add_argument("--baryonified", action="store_true")

    parser.add_argument("--tomo-bin", type=int, default=4)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--lmax", type=int, default=None,
                        help="Maximum multipole (default: 3*nside).")
    parser.add_argument("--ell-min", type=float, default=10.0)
    parser.add_argument("--n-ell-bins", type=int, default=60)

    # DoTH-specific
    parser.add_argument("--theta", type=float, default=20.0,
                        help="Inner top-hat smoothing scale in arcmin (default: 20.0).")
    parser.add_argument("--theta-ratio", type=float, default=2.0,
                        help="Ratio theta2/theta1 (default: 2.0 → theta2=40 arcmin).")

    parser.add_argument("--anafast-iter", type=int, default=3)
    parser.add_argument("--use-pixel-weights", action="store_true")

    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--test-n", type=int, default=None)

    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# File record helpers (identical logic to cls_processing_halofit.py)
# ---------------------------------------------------------------------------

def load_selected_data(selection_file: Path, params_file: Path):
    indices = np.load(selection_file).astype(int)
    params = np.load(params_file)
    return indices, params


def read_actual_mapping(actual_file: Path) -> list[int]:
    if not actual_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {actual_file}")
    with open(actual_file, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def build_file_records(selected_indices, base_dir, map_filename, actual_file):
    actual_cosmo_nums = read_actual_mapping(actual_file)
    records = []
    for row_idx, idx in enumerate(selected_indices):
        cosmo_idx = int(idx) // 7
        perm_num = int(idx) % 7
        if cosmo_idx >= len(actual_cosmo_nums):
            records.append({"row_idx": row_idx, "selected_idx": int(idx),
                             "file_path": None, "exists": False,
                             "error": f"cosmo_idx {cosmo_idx} out of range"})
            continue
        cosmo_num = actual_cosmo_nums[cosmo_idx]
        file_path = base_dir / f"cosmo_{cosmo_num:06d}" / f"perm_{perm_num:04d}" / map_filename
        exists = file_path.exists()
        records.append({"row_idx": row_idx, "selected_idx": int(idx),
                         "file_path": str(file_path), "exists": bool(exists),
                         "error": None if exists else f"missing file: {file_path}"})
    return records


def build_fiducial_records(fiducial_dir, map_filename, n_perms):
    records = []
    for perm_num in range(n_perms):
        file_path = fiducial_dir / f"perm_{perm_num:04d}" / map_filename
        exists = file_path.exists()
        records.append({"row_idx": perm_num, "perm_num": perm_num,
                         "file_path": str(file_path), "exists": bool(exists),
                         "error": None if exists else f"missing file: {file_path}"})
    return records


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_compute_doth_cls(
    record: dict,
    tomo_bin: int,
    lmax: int,
    ell_edges: np.ndarray,
    theta: float,
    theta_ratio: float,
    anafast_iter: int,
    use_pixel_weights: bool,
) -> tuple[int, np.ndarray | None, str | None]:
    row_idx = int(record["row_idx"])
    if not bool(record["exists"]):
        return row_idx, None, str(record.get("error"))

    map_key = f"kg/stage3_lensing{tomo_bin}"
    try:
        with h5py.File(str(record["file_path"]), "r") as f:
            if map_key not in f:
                return row_idx, None, f"missing dataset {map_key}"
            kappa = np.asarray(f[map_key], dtype=np.float64)

        nside = hp.npix2nside(len(kappa))
        kappa_s1, kappa_s2 = smooth_map_dual(
            kappa, theta, theta * theta_ratio, nside=nside, lmax=lmax
        )
        kappa_doth = kappa_s2 - kappa_s1

        cl_full = hp.anafast(
            kappa_doth,
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


def _iter_records(records, num_workers, chunksize, worker, total):
    if num_workers == 1:
        for rec in records:
            yield worker(rec)
        return
    with mp.Pool(processes=num_workers, initializer=seed_worker) as pool:
        yield from pool.imap_unordered(worker, records, chunksize=chunksize)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    map_filename = (
        "projected_probes_maps_baryonified512.h5"
        if args.baryonified
        else "projected_probes_maps_nobaryons512.h5"
    )
    lmax = int(args.lmax if args.lmax is not None else 3 * int(args.nside))
    ell_edges = build_log_ell_edges(args.ell_min, lmax, args.n_ell_bins)
    ell_centers = binned_ells_from_edges(ell_edges)

    theta_tag = f"theta{args.theta:.1f}_ratio{args.theta_ratio:.1f}"
    map_tag = "baryonified" if args.baryonified else "nobaryons"

    if args.fiducial:
        fiducial_dir = Path(args.fiducial_dir)
        n_perms = int(args.n_perms)
        records = build_fiducial_records(fiducial_dir, map_filename, n_perms)
        if args.test_n is not None:
            records = records[: int(args.test_n)]
        n_rows = len(records)
        selected_params = np.tile(FIDUCIAL_PARAMS, (n_rows, 1))
        selected_indices = None
        base_dir_str = str(fiducial_dir)
        mode_label = "fiducial"
        default_out = (
            repo_root / "data" / "cls" / "simulations"
            / f"sim_doth_cls_bin{args.tomo_bin}_{theta_tag}_fiducial.npz"
        )
    else:
        selection_dir = Path(args.selection_dir)
        selection_file = args.selection_file or (selection_dir / "selected_indices_halofit.npy")
        params_file = args.params_file or (selection_dir / "selected_params_halofit.npy")
        if args.base_dir is not None:
            base_dir = Path(args.base_dir)
        else:
            base_dir = Path(
                "/home/tersenov/CosmoGridV1/stage3_forecast/grid"
                if args.baryonified
                else "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid"
            )
        actual_file = args.actual_file or (base_dir / "actual.txt")
        selected_indices, selected_params = load_selected_data(selection_file, params_file)
        if args.test_n is not None:
            n_keep = int(max(1, min(args.test_n, len(selected_indices))))
            selected_indices = selected_indices[:n_keep]
            selected_params = selected_params[:n_keep]
        n_rows = len(selected_indices)
        records = build_file_records(selected_indices, base_dir, map_filename, actual_file)
        base_dir_str = str(base_dir)
        mode_label = "halofit"
        default_out = (
            repo_root / "data" / "cls" / "simulations"
            / f"sim_doth_cls_bin{args.tomo_bin}_{theta_tag}_{map_tag}.npz"
        )

    existing_records = [r for r in records if bool(r["exists"])]
    missing_count = len(records) - len(existing_records)

    num_workers = max(1, int(args.num_workers)) if args.num_workers is not None else _auto_workers()
    chunksize = (
        max(1, int(args.chunksize))
        if args.chunksize is not None
        else max(1, len(existing_records) // max(1, num_workers * 4))
    )

    output_file = Path(args.output_file) if args.output_file is not None else default_out
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_file}. Use --overwrite to replace it.")

    if args.verbose:
        print(f"Mode:           {mode_label}")
        print(f"DoTH filter:    theta={args.theta} arcmin, ratio={args.theta_ratio}")
        print(f"Map file:       {map_filename}")
        print(f"Rows total:     {n_rows}  (missing: {missing_count})")
        print(f"tomo_bin:       {args.tomo_bin}")
        print(f"nside/lmax:     {args.nside}/{lmax}")
        print(f"ell bins:       {args.n_ell_bins}")
        print(f"workers/chunk:  {num_workers}/{chunksize}")

    cls = np.full((n_rows, len(ell_centers)), np.nan, dtype=np.float64)
    valid_mask = np.zeros(n_rows, dtype=bool)
    error_messages = [""] * n_rows

    for rec in records:
        if not bool(rec["exists"]):
            error_messages[int(rec["row_idx"])] = str(rec["error"])

    worker = partial(
        _worker_compute_doth_cls,
        tomo_bin=int(args.tomo_bin),
        lmax=lmax,
        ell_edges=ell_edges,
        theta=float(args.theta),
        theta_ratio=float(args.theta_ratio),
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
        iterator, total=len(existing_records), desc="Computing DoTH C_ell"
    ):
        if cl_row is None:
            error_messages[row_idx] = err or "unknown error"
            continue
        cls[row_idx] = cl_row
        valid_mask[row_idx] = True

    payload = dict(
        params=selected_params,
        param_names=PARAM_NAMES,
        ells=ell_centers,
        cls=cls,
        tomo_bin=int(args.tomo_bin),
        n_cosmo=int(n_rows),
        valid_mask=valid_mask,
        ell_edges=ell_edges,
        nside=int(args.nside),
        lmax=int(lmax),
        map_type=np.array(map_tag),
        map_filename=np.array(map_filename),
        base_dir=np.array(base_dir_str),
        n_valid=int(np.sum(valid_mask)),
        n_missing=int(missing_count),
        error_messages=np.array(error_messages, dtype=object),
        mode=np.array(mode_label),
        theta=np.float64(args.theta),
        theta_ratio=np.float64(args.theta_ratio),
        doth_formula=np.array("kappa_doth = kappa_smooth(theta*ratio) - kappa_smooth(theta)"),
    )
    if selected_indices is not None:
        payload["selected_indices"] = selected_indices

    np.savez_compressed(output_file, **payload)
    print(f"Saved DoTH C_ell output: {output_file}")
    print(f"Valid rows: {int(np.sum(valid_mask))}/{n_rows}")
    if missing_count > 0:
        print(f"Missing files: {missing_count}")


if __name__ == "__main__":
    main()
