#!/usr/bin/env python3
"""
BNT L1-norm processing for cosmoGRID maps (single bin, single theta).

Loads all 4 standard tomographic kappa maps from each h5 file, applies the
cosmoGRID BNT matrix, then runs the DoTH (Difference-of-Top-Hats) filter on
the requested BNT bin and computes the wavelet L1-norm = pdf * |kappa|.

Provenance of the BNT matrix: bar_impact/scripts/bnt_l1_norm_processing_new_mask.py
and bar_impact/notebooks/BNTcp.ipynb (derived for cosmoGRID Stage-III n(z)).

CPU usage is hard-capped at HARD_CPU_CAP=40 workers regardless of --num-workers,
to honour the user-set cap for this investigation.
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
from functools import lru_cache, partial
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm


HARD_CPU_CAP = 40

# cosmoGRID BNT matrix (provenance: bar_impact + BNTcp.ipynb).
BNT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [0.4521097, -1.4521097, 1.0, 0.0],
        [0.0, 0.25127807, -1.251278, 1.0],
    ]
)


@contextlib.contextmanager
def suppress_stdout():
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved


def seed_worker():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    return kg + np.random.normal(loc=0.0, scale=sigma_pix, size=npix)


@lru_cache(maxsize=8)
def get_beam(theta, lmax):
    def top_hat(b, radius):
        return np.where(
            np.abs(b) <= radius, 1 / (np.cos(radius) - 1) / (-2 * np.pi), 0
        )

    t = theta * np.pi / (60 * 180)
    b = np.linspace(0.0, t * 1.2, 10000)
    bw = top_hat(b, t)
    return hp.sphtfunc.beam2bl(bw, b, lmax)


def smooth_map_dual(kappa_map, theta1, theta2, nside=512, lmax_factor=3.0, fast_mode=False):
    lmax = int(nside * lmax_factor - 1)
    if fast_mode:
        almk = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, iter=0)
    else:
        almk = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, use_pixel_weights=True)
    beam1 = get_beam(theta1, lmax)
    beam2 = get_beam(theta2, lmax)
    k1 = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almk, beam1), nside, lmax=lmax)
    k2 = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almk, beam2), nside, lmax=lmax)
    return k1, k2


def process_file(file_path, bnt_row=3, noise_level=0.26, add_noise=True,
                 theta=30.0, theta_ratio=2.0, nbins=200, nside=512, kappa_range=None,
                 lmax_factor=3.0, fast_mode=False, dataset_suffix="halofit",
                 verbose=False, force_overwrite=False):
    """
    Read all 4 kappa bins, apply BNT, and compute the DoTH L1-norm on BNT bin `bnt_row`.

    Output filenames mirror the non-BNT version but with `_bnt{row+1}_` infix so they
    don't collide with standard-bin runs.
    """
    bnt_tag = f"bnt{bnt_row + 1}"
    theta_str = f"theta{theta:.1f}"
    suffix_common = f"_{bnt_tag}_bin{bnt_row + 1}_{theta_str}_ratio{theta_ratio:.1f}"
    if add_noise:
        suffix_common += f"_noisy_s{noise_level:.2f}"
    suffix_common += f"_{dataset_suffix}.npy"

    variance_save = file_path.replace(".h5", "_variance" + suffix_common)
    l1_save = file_path.replace(".h5", "_l1_norm" + suffix_common)
    kappa_save = file_path.replace(".h5", "_kappa" + suffix_common)

    if not force_overwrite and all(os.path.exists(p) for p in (variance_save, l1_save, kappa_save)):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, output files already exist.")
        return variance_save, l1_save, kappa_save

    try:
        with h5py.File(file_path, "r") as f:
            kgs = np.stack(
                [np.array(f[f"kg/stage3_lensing{i}"]) for i in (1, 2, 3, 4)],
                axis=0,
            )

        if add_noise:
            kgs = np.stack(
                [add_shape_noise(kgs[i], sigma_e=noise_level, nside=nside) for i in range(4)],
                axis=0,
            )

        kgs_bnt = BNT_MATRIX @ kgs
        kg = kgs_bnt[bnt_row]
        del kgs, kgs_bnt

        kappa_smooth1, kappa_smooth2 = smooth_map_dual(
            kg, theta, theta * theta_ratio,
            nside=nside, lmax_factor=lmax_factor, fast_mode=fast_mode,
        )
        k_doth = kappa_smooth2 - kappa_smooth1
        variance = float(np.var(k_doth))

        if kappa_range is not None:
            pdf, edges = np.histogram(
                k_doth, bins=nbins, range=kappa_range, density=True
            )
        else:
            pdf, edges = np.histogram(k_doth, bins=nbins, density=True)

        kappa_centers = 0.5 * (edges[:-1] + edges[1:])
        l1 = pdf * np.abs(kappa_centers)

        np.save(variance_save, variance)
        np.save(l1_save, l1)
        np.save(kappa_save, kappa_centers)

        if verbose:
            print(f"Processed: {os.path.basename(file_path)}  var={variance:.4e}")
        return variance_save, l1_save, kappa_save
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None, None, None


def load_selected_indices(selection_file):
    indices = np.load(selection_file)
    print(f"Loaded {len(indices)} selected simulation indices from {selection_file}")
    return indices


def build_file_paths_from_indices(indices, base_dir, baryonified=False):
    base_dir = Path(base_dir)
    actual_txt = base_dir / "actual.txt"
    if not actual_txt.exists():
        raise FileNotFoundError(f"Mapping file not found: {actual_txt}")

    with open(actual_txt, "r") as f:
        actual_cosmo_nums = [int(line.strip()) for line in f if line.strip()]

    filename = (
        "projected_probes_maps_baryonified512.h5" if baryonified
        else "projected_probes_maps_nobaryons512.h5"
    )
    paths, missing = [], []
    for idx in indices:
        cosmo_idx, perm_num = idx // 7, idx % 7
        actual = actual_cosmo_nums[cosmo_idx]
        p = base_dir / f"cosmo_{actual:06d}" / f"perm_{perm_num:04d}" / filename
        (paths if p.exists() else missing).append(str(p))
    if missing:
        print(f"\nWarning: {len(missing)} files not found (e.g., {missing[0]}).")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Apply BNT to all 4 kappa maps then compute L1-norm on a chosen BNT bin."
    )
    parser.add_argument("--fiducial", action="store_true")
    parser.add_argument("--selection-file", type=str,
                        default="/home/tersenov/software/bar_impact/data/selected_indices_halofit.npy")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--baryonified", action="store_true")
    parser.add_argument("--bnt-bin", type=int, default=4, choices=(1, 2, 3, 4),
                        help="1-indexed BNT bin (default 4).")
    parser.add_argument("--noise-level", type=float, default=0.26)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--theta", type=float, default=30.0)
    parser.add_argument("--theta-ratio", type=float, default=2.0)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--kappa-min", type=float, default=None)
    parser.add_argument("--kappa-max", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help=f"Hard-capped to HARD_CPU_CAP={HARD_CPU_CAP}.")
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test-n", type=int, default=None)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--lmax-factor", type=float, default=3.0)
    parser.add_argument("--save-combined", action="store_true")
    parser.add_argument("--combined-output", default=None)
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    bnt_row = args.bnt_bin - 1

    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/new_grid/"
    filename = (
        "projected_probes_maps_baryonified512.h5" if args.baryonified
        else "projected_probes_maps_nobaryons512.h5"
    )

    if args.fiducial:
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        file_paths = [
            os.path.join(base_dir, p, filename) for p in perm_dirs
            if os.path.exists(os.path.join(base_dir, p, filename))
        ]
        print(f"Fiducial mode: found {len(file_paths)} files")
    else:
        selected = load_selected_indices(args.selection_file)
        file_paths = build_file_paths_from_indices(selected, base_dir, args.baryonified)

    if not file_paths:
        print("Error: no valid file paths found.")
        return

    if args.test_n is not None:
        file_paths = file_paths[: args.test_n]
        print(f"Testing mode: processing only first {len(file_paths)} files")

    if args.num_workers is None:
        try:
            import psutil
            args.num_workers = psutil.cpu_count(logical=False)
        except ImportError:
            args.num_workers = max(1, mp.cpu_count() // 2)
    args.num_workers = max(1, min(args.num_workers, HARD_CPU_CAP))
    print(f"Workers: {args.num_workers}  (hard cap: {HARD_CPU_CAP})")

    if args.chunksize is None:
        args.chunksize = max(1, len(file_paths) // (args.num_workers * 4))

    kappa_range = (
        (args.kappa_min, args.kappa_max)
        if args.kappa_min is not None and args.kappa_max is not None
        else None
    )

    dataset_suffix = "fiducial" if args.fiducial else "halofit"

    print(f"\n{'=' * 70}")
    print(f"BNT L1-norm processing: BNT bin {args.bnt_bin}")
    print(f"  theta={args.theta} arcmin, ratio={args.theta_ratio}, nbins={args.nbins}")
    if kappa_range:
        print(f"  kappa range: [{kappa_range[0]:.4e}, {kappa_range[1]:.4e}]")
    else:
        print("  kappa range: adaptive (per-file)")
    print(f"  noise: {'OFF' if args.no_noise else f'ON (sigma_e={args.noise_level})'}")
    print(f"  dataset: {dataset_suffix},  baryonified={args.baryonified}")
    print(f"  files: {len(file_paths)}")
    print(f"{'=' * 70}\n")

    with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
        process_func = partial(
            process_file,
            bnt_row=bnt_row,
            noise_level=args.noise_level,
            add_noise=not args.no_noise,
            theta=args.theta,
            theta_ratio=args.theta_ratio,
            nbins=args.nbins,
            nside=args.nside,
            kappa_range=kappa_range,
            lmax_factor=args.lmax_factor,
            fast_mode=args.fast_mode,
            dataset_suffix=dataset_suffix,
            verbose=args.verbose,
            force_overwrite=args.force_overwrite,
        )
        results = list(tqdm(
            pool.imap(process_func, file_paths, chunksize=args.chunksize),
            total=len(file_paths),
            desc=f"BNT bin {args.bnt_bin}",
        ))

    successful = [r for r in results if r is not None and r[0] is not None]
    print(f"\nProcessed {len(successful)}/{len(file_paths)} files")

    if not args.save_combined:
        return

    map_suffix = "baryonified" if args.baryonified else "nobaryons"
    if args.combined_output:
        base = args.combined_output
        # Strip a trailing ".npy" if user supplied one — we add per-component suffixes
        if base.endswith(".npy"):
            base = base[:-4]
        var_out = f"{base}_variance.npy"
        l1_out = f"{base}_l1.npy"
        kap_out = f"{base}_kappa.npy"
    else:
        tag = f"_bnt{args.bnt_bin}_bin{args.bnt_bin}_theta{args.theta:.1f}_ratio{args.theta_ratio:.1f}"
        if not args.no_noise:
            tag += f"_noisy_s{args.noise_level:.2f}"
        var_out = os.path.join(base_dir, f"all_variances_{dataset_suffix}_{map_suffix}{tag}.npy")
        l1_out = os.path.join(base_dir, f"all_l1_norms_{dataset_suffix}_{map_suffix}{tag}.npy")
        kap_out = os.path.join(base_dir, f"all_kappas_{dataset_suffix}_{map_suffix}{tag}.npy")

    print(f"\nLoading and combining {len(successful)} per-file outputs...")
    vars_list, l1_list, kap_list = [], [], []
    for var_p, l1_p, kap_p in tqdm(successful, desc="Combining"):
        try:
            vars_list.append(np.load(var_p, allow_pickle=True))
            l1_list.append(np.load(l1_p, allow_pickle=True))
            kap_list.append(np.load(kap_p, allow_pickle=True))
        except Exception as e:
            if args.verbose:
                print(f"Skipping due to load error: {e}")

    all_variances = np.array(vars_list)
    all_l1 = np.stack(l1_list, axis=0)
    all_kappa = np.stack(kap_list, axis=0)
    print(f"Combined shapes — variances {all_variances.shape}, l1 {all_l1.shape}, kappa {all_kappa.shape}")

    os.makedirs(os.path.dirname(var_out), exist_ok=True)
    np.save(var_out, all_variances)
    np.save(l1_out, all_l1)
    np.save(kap_out, all_kappa)
    print(f"Saved:\n  {var_out}\n  {l1_out}\n  {kap_out}")


if __name__ == "__main__":
    main()
