#!/usr/bin/env python3
"""
L1 Norm Processing Script (Halofit Selection) - Processes only cosmologies passing Halofit criteria.
"""

import os
import h5py
import healpy as hp
import numpy as np
import argparse
import multiprocessing as mp
import contextlib
import sys
import io
from pathlib import Path
from tqdm import tqdm
from functools import partial, lru_cache


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout


def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """
    Adds shape noise to a full-sky Healpix convergence (kappa) map.
    
    Parameters:
    - kg: np.ndarray, the input kappa map
    - sigma_e: float, intrinsic ellipticity dispersion per galaxy
    - galaxy_density: float, galaxy number density per arcmin²
    - nside: int, Healpix resolution parameter
    
    Returns:
    - noisy_kg: np.ndarray, kappa map with added shape noise
    """
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise


@lru_cache(maxsize=8)
def get_beam(theta, lmax):
    """
    Compute the beam transfer function for a top-hat filter.
    Cached to avoid recomputing for the same theta/lmax combination.
    
    Parameters:
    - theta: smoothing scale in arcmin
    - lmax: maximum multipole
    
    Returns:
    - beam: beam transfer function (as tuple for hashability)
    """
    def top_hat(b, radius):
        # Returns a top-hat filter
        return np.where(np.abs(b) <= radius, 1/(np.cos(radius) - 1)/(-2*np.pi), 0)
    
    t = theta * np.pi / (60 * 180)  # convert arcmin to radians
    b = np.linspace(0.0, t * 1.2, 10000)
    bw = top_hat(b, t)
    beam = hp.sphtfunc.beam2bl(bw, b, lmax)
    return beam


def smooth_map(kappa_map, theta, nside=512):
    """
    Smooth a HEALPix kappa map using a top-hat filter.
    
    Parameters:
    - kappa_map: input convergence map
    - theta: smoothing scale in arcmin
    - nside: HEALPix resolution parameter
    
    Returns:
    - kappa_smooth: smoothed convergence map
    """
    lmax = nside * 3 - 1
    beam = get_beam(theta, lmax)
    almkappa = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, use_pixel_weights=True)
    kappa_smooth = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam), nside, lmax=lmax)
    return kappa_smooth


def smooth_map_dual(kappa_map, theta1, theta2, nside=512, lmax_factor=3.0, fast_mode=False):
    """
    Smooth a HEALPix kappa map at two different scales efficiently.
    Computes alm transform only once, then applies two different beams.
    
    Parameters:
    - kappa_map: input convergence map
    - theta1: first smoothing scale in arcmin
    - theta2: second smoothing scale in arcmin
    - nside: HEALPix resolution parameter
    - lmax_factor: lmax = nside * lmax_factor - 1 (default 3.0, use 2.0 for speed)
    - fast_mode: if True, use iter=0 (faster but less accurate)
    
    Returns:
    - kappa_smooth1: map smoothed with theta1
    - kappa_smooth2: map smoothed with theta2
    """
    lmax = int(nside * lmax_factor - 1)
    
    # Compute alm only once (major bottleneck)
    if fast_mode:
        # Fast mode: no iterative refinement
        almkappa = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, iter=0)
    else:
        # Standard mode: use pixel weights for better accuracy
        almkappa = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, use_pixel_weights=True)
    
    # Apply two different beams
    beam1 = get_beam(theta1, lmax)
    beam2 = get_beam(theta2, lmax)
    
    kappa_smooth1 = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam1), nside, lmax=lmax)
    kappa_smooth2 = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam2), nside, lmax=lmax)
    
    return kappa_smooth1, kappa_smooth2


def compute_l1_norm_from_pdf(kappa_values, pdf_values):
    """
    Compute L1 norm as integral of |kappa| * pdf.
    
    Parameters:
    - kappa_values: kappa bin centers
    - pdf_values: probability density values
    
    Returns:
    - l1_norm: scalar L1 norm value
    """
    return np.trapz(np.abs(kappa_values) * pdf_values, kappa_values)


def process_file(file_path, bin_number=2, noise_level=0.26, add_noise=True, 
                theta=15.0, nbins=500, nside=512, kappa_range=None, 
                lmax_factor=3.0, fast_mode=False, dataset_suffix="halofit", verbose=False):
    """
    Process a single file using cosmogrid approach:
    - Load kappa map
    - Optionally add shape noise
    - Smooth with theta and 2*theta
    - Compute variance of the difference
    - Create PDF histogram with fixed kappa range
    - Compute L1 norm vector as |kappa| * pdf
    - Save variance (scalar), L1 norm vector (array), and kappa vector (array)
    
    Parameters:
    - kappa_range: tuple (kappa_min, kappa_max) for fixed histogram range. 
                   If None, uses data range.
    - lmax_factor: lmax = nside * lmax_factor - 1
    - fast_mode: use faster HEALPix settings (iter=0)
    - dataset_suffix: suffix to add to output files (e.g., "halofit", "fiducial")
    """
    
    # Define output filenames based on bin number, theta, and noise level
    if add_noise:
        variance_suffix = f"_variance_bin{bin_number}_theta{theta:.1f}_noisy_s{noise_level:.2f}_{dataset_suffix}.npy"
        l1_suffix = f"_l1_norm_bin{bin_number}_theta{theta:.1f}_noisy_s{noise_level:.2f}_{dataset_suffix}.npy"
        kappa_suffix = f"_kappa_bin{bin_number}_theta{theta:.1f}_noisy_s{noise_level:.2f}_{dataset_suffix}.npy"
    else:
        variance_suffix = f"_variance_bin{bin_number}_theta{theta:.1f}_{dataset_suffix}.npy"
        l1_suffix = f"_l1_norm_bin{bin_number}_theta{theta:.1f}_{dataset_suffix}.npy"
        kappa_suffix = f"_kappa_bin{bin_number}_theta{theta:.1f}_{dataset_suffix}.npy"
    
    variance_save_path = file_path.replace(".h5", variance_suffix)
    l1_save_path = file_path.replace(".h5", l1_suffix)
    kappa_save_path = file_path.replace(".h5", kappa_suffix)
    
    # Map key based on bin number
    map_key = f"kg/stage3_lensing{bin_number}"
    
    # Skip if files already exist
    if os.path.exists(variance_save_path) and os.path.exists(l1_save_path) and os.path.exists(kappa_save_path):
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, output files already exist.")
        return variance_save_path, l1_save_path, kappa_save_path
    
    try:
        # Load kappa map
        with h5py.File(file_path, "r") as f:
            kg = np.array(f[map_key])
        
        # Add shape noise if requested
        if add_noise:
            kg = add_shape_noise(kg, sigma_e=noise_level, nside=nside)
        
        # Smooth with theta and 2*theta efficiently (compute alm only once)
        kappa_smooth1, kappa_smooth2 = smooth_map_dual(
            kg, theta, theta * 2, nside=nside, 
            lmax_factor=lmax_factor, fast_mode=fast_mode
        )
        
        # Compute the mass map difference
        k_massmap_simulation = kappa_smooth2 - kappa_smooth1
        
        # Compute variance
        variance = np.var(k_massmap_simulation)
        
        # Create PDF from histogram with fixed or adaptive range
        if kappa_range is not None:
            # Use fixed range for consistency across all files
            kappa_min, kappa_max = kappa_range
            pdf_simulation, bin_edges = np.histogram(
                k_massmap_simulation, bins=nbins, range=(kappa_min, kappa_max), density=True
            )
        else:
            # Use data-driven range (default behavior)
            pdf_simulation, bin_edges = np.histogram(k_massmap_simulation, bins=nbins, density=True)
        
        kappa_sim = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Compute L1 norm vector as |kappa| * pdf (following cosmogrid notebook)
        l1_norm_vector = pdf_simulation * np.abs(kappa_sim)
        
        # Save results
        np.save(variance_save_path, variance)
        np.save(l1_save_path, l1_norm_vector)
        np.save(kappa_save_path, kappa_sim)
        
        if verbose:
            print(f"Processed: {os.path.basename(file_path)}")
            print(f"  Variance: {variance:.6e}, L1 norm vector length: {len(l1_norm_vector)}")
            print(f"  Kappa range: [{kappa_sim.min():.6e}, {kappa_sim.max():.6e}]")
        
        return variance_save_path, l1_save_path, kappa_save_path
        
    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None, None, None


def load_selected_indices(selection_file):
    """Load the selected indices from the Halofit selection."""
    indices = np.load(selection_file)
    print(f"Loaded {len(indices)} selected simulation indices from {selection_file}")
    return indices


def build_file_paths_from_indices(indices, base_dir, baryonified=False):
    """Build file paths from selected indices using actual.txt mapping."""
    base_dir = Path(base_dir)
    
    # Load actual cosmology number mapping
    actual_txt = base_dir / 'actual.txt'
    if not actual_txt.exists():
        raise FileNotFoundError(f"Mapping file not found: {actual_txt}")
    
    with open(actual_txt, 'r') as f:
        actual_cosmo_nums = [int(line.strip()) for line in f if line.strip()]
    
    # Note: Only baryonified512.h5 files exist in the data directories
    filename = "projected_probes_maps_baryonified512.h5"
    
    file_paths = []
    missing_files = []
    
    for idx in indices:
        cosmo_idx = idx // 7  # Which cosmology (0-2423)
        perm_num = idx % 7     # Which permutation (0-6)
        
        # Get actual cosmology number from mapping
        actual_cosmo_num = actual_cosmo_nums[cosmo_idx]
        
        cosmo_dir = f"cosmo_{actual_cosmo_num:06d}"
        perm_dir = f"perm_{perm_num:04d}"
        
        file_path = base_dir / cosmo_dir / perm_dir / filename
        
        if file_path.exists():
            file_paths.append(str(file_path))
        else:
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found:")
        if len(missing_files) <= 10:
            for f in missing_files:
                print(f"  - {f}")
        else:
            print(f"  - {missing_files[0]}")
            print(f"  ... ({len(missing_files)-2} more)")
            print(f"  - {missing_files[-1]}")
    
    return file_paths


def main():
    """Main function to handle command-line arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Process HEALPix maps to compute L1 norms (Halofit selection or fiducial cosmology)."
    )
    
    # Dataset selection
    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of Halofit selection.")
    
    # Selection file
    parser.add_argument("--selection-file", type=str,
                        default="/home/tersenov/software/bar_impact/data/selected_indices_halofit.npy",
                        help="Path to the Halofit selection indices file (ignored if --fiducial is set).")
    
    # Main processing options
    parser.add_argument("--base-dir", type=str,
                        default=None,
                        help="Base directory for data (default: auto-selected based on --fiducial).")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")
    
    # Bin selection
    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument("--bin-number", type=int, default=1, 
                        help="Single bin number to process (default: 1)")
    bin_group.add_argument("--bins", type=str,
                        help="Comma-separated list of bin numbers to process (e.g., '1,2,3,4')")
    
    # Noise options
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Shape noise level (sigma_e)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Don't add shape noise to maps.")
    
    # Algorithm parameters
    parser.add_argument("--theta", type=float, default=15.0,
                        help="Smoothing scale in arcmin (default: 15.0)")
    parser.add_argument("--nbins", type=int, default=500,
                        help="Number of bins for PDF histogram (default: 500)")
    parser.add_argument("--nside", type=int, default=512,
                        help="HEALPix nside parameter (default: 512)")
    parser.add_argument("--kappa-min", type=float, default=None,
                        help="Minimum kappa value for fixed histogram range (default: auto)")
    parser.add_argument("--kappa-max", type=float, default=None,
                        help="Maximum kappa value for fixed histogram range (default: auto)")
    
    # Execution options
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: auto-detect physical cores)")
    parser.add_argument("--chunksize", type=int, default=None,
                        help="Number of files per worker task (default: auto-calculate)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information.")
    parser.add_argument("--test-n", type=int, default=None,
                        help="Process only the first N files for testing (default: process all)")
    
    # Performance options
    parser.add_argument("--fast-mode", action="store_true",
                        help="Use faster HEALPix settings (iter=0, lower accuracy)")
    parser.add_argument("--lmax-factor", type=float, default=3.0,
                        help="lmax = nside * factor - 1 (default: 3.0, use 2.0 for speed)")
    
    # Output options
    parser.add_argument("--save-combined", action="store_true",
                        help="Save combined L1 norms to a single file.")
    parser.add_argument("--combined-output", 
                        help="Path for combined output file.")
    
    args = parser.parse_args()
    
    # Set the base directory based on fiducial flag or override
    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/home/tersenov/CosmoGridV1/stage3_forecast/grid/"
    
    # Set the filename based on the baryonified flag
    if args.baryonified:
        filename = "projected_probes_maps_baryonified512.h5"
    else:
        filename = "projected_probes_maps_nobaryons512.h5"
    
    # Build file paths based on dataset type
    if args.fiducial:
        # Fiducial cosmology: process all permutations
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]  # "perm_0000" to "perm_0199"
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
        print(f"Fiducial mode: found {len(file_paths)} files")
    else:
        # Halofit selection: load selected indices
        selected_indices = load_selected_indices(args.selection_file)
        
        # Build file paths from selected indices
        file_paths = build_file_paths_from_indices(
            selected_indices, 
            base_dir, 
            args.baryonified
        )
    
    if not file_paths:
        print("Error: No valid file paths found!")
        return
    
    # Limit files for testing
    if args.test_n is not None:
        file_paths = file_paths[:args.test_n]
        print(f"Testing mode: processing only first {len(file_paths)} files")
    
    # Auto-detect optimal number of workers
    if args.num_workers is None:
        # Use physical cores (not logical/hyperthreaded)
        try:
            import psutil
            args.num_workers = psutil.cpu_count(logical=False)
        except ImportError:
            # Fallback: use half of logical cores
            args.num_workers = max(1, mp.cpu_count() // 2)
        print(f"Auto-detected {args.num_workers} worker processes")
    
    # Auto-calculate optimal chunksize
    if args.chunksize is None:
        # Rule of thumb: total_tasks / (num_workers * 4)
        args.chunksize = max(1, len(file_paths) // (args.num_workers * 4))
        print(f"Auto-calculated chunksize: {args.chunksize}")
    
    # Parse bin numbers
    if args.bins:
        bin_numbers = [int(b.strip()) for b in args.bins.split(',')]
        print(f"Processing multiple bins: {bin_numbers}")
    else:
        bin_numbers = [args.bin_number]
        print(f"Processing single bin: {args.bin_number}")
    
    # Set up kappa range
    if args.kappa_min is not None and args.kappa_max is not None:
        kappa_range = (args.kappa_min, args.kappa_max)
        print(f"Using fixed kappa range: [{args.kappa_min}, {args.kappa_max}]")
    else:
        kappa_range = None
        print("Using adaptive kappa range (data-driven)")
    
    # Print configuration information
    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "Halofit selection"
    print(f"\n{'='*70}")
    print(f"Processing {len(file_paths)} {map_type} files ({dataset_type})")
    print(f"{'='*70}")
    if len(bin_numbers) == 1:
        print(f"Map key: kg/stage3_lensing{bin_numbers[0]}")
    else:
        print(f"Map keys: {', '.join([f'kg/stage3_lensing{b}' for b in bin_numbers])}")
    print(f"Smoothing scale (theta): {args.theta} arcmin")
    print(f"PDF histogram bins: {args.nbins}")
    print(f"HEALPix nside: {args.nside}")
    print(f"HEALPix lmax: {int(args.nside * args.lmax_factor - 1)} (factor: {args.lmax_factor})")
    print(f"Fast mode: {'ON (iter=0)' if args.fast_mode else 'OFF (use_pixel_weights=True)'}")
    print(f"Workers: {args.num_workers}, Chunksize: {args.chunksize}")
    if kappa_range:
        print(f"Fixed kappa range: [{kappa_range[0]}, {kappa_range[1]}]")
    
    # Process each bin
    all_bin_results = {}
    dataset_suffix = "fiducial" if args.fiducial else "halofit"
    
    for bin_number in bin_numbers:
        print(f"\n{'='*60}")
        print(f"Processing bin {bin_number}")
        print(f"{'='*60}")
        
        # Determine suffix for output files
        if args.no_noise:
            variance_suffix = f"_variance_bin{bin_number}_theta{args.theta:.1f}_{dataset_suffix}.npy"
            l1_suffix = f"_l1_norm_bin{bin_number}_theta{args.theta:.1f}_{dataset_suffix}.npy"
            kappa_suffix = f"_kappa_bin{bin_number}_theta{args.theta:.1f}_{dataset_suffix}.npy"
        else:
            variance_suffix = f"_variance_bin{bin_number}_theta{args.theta:.1f}_noisy_s{args.noise_level:.2f}_{dataset_suffix}.npy"
            l1_suffix = f"_l1_norm_bin{bin_number}_theta{args.theta:.1f}_noisy_s{args.noise_level:.2f}_{dataset_suffix}.npy"
            kappa_suffix = f"_kappa_bin{bin_number}_theta{args.theta:.1f}_noisy_s{args.noise_level:.2f}_{dataset_suffix}.npy"
        print(f"Variance output suffix: {variance_suffix}")
        print(f"L1 norm output suffix: {l1_suffix}")
        print(f"Kappa output suffix: {kappa_suffix}")
        
        # Process files in parallel with progress bar
        with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
            process_func = partial(
                process_file,
                bin_number=bin_number,
                noise_level=args.noise_level,
                add_noise=not args.no_noise,
                theta=args.theta,
                nbins=args.nbins,
                nside=args.nside,
                kappa_range=kappa_range,
                lmax_factor=args.lmax_factor,
                fast_mode=args.fast_mode,
                dataset_suffix=dataset_suffix,
                verbose=args.verbose,
            )
            results = list(tqdm(
                pool.imap(process_func, file_paths, chunksize=args.chunksize),
                total=len(file_paths),
                desc=f"Processing bin {bin_number}"
            ))
        
        # Collect successful triples (variance, l1, kappa) preserving original order
        successful_triples = [r for r in results if r is not None and r[0] is not None and r[1] is not None and r[2] is not None]
        successful_variance = [r[0] for r in successful_triples]
        successful_l1 = [r[1] for r in successful_triples]
        successful_kappa = [r[2] for r in successful_triples]
        processed = len(successful_triples)
        print(f"Bin {bin_number} processing complete: {processed}/{len(file_paths)} files processed")

        all_bin_results[bin_number] = {
            'variance': successful_variance,
            'l1': successful_l1,
            'kappa': successful_kappa,
            'triples': successful_triples,
        }
    
    print(f"\n{'='*60}")
    print(f"All bins processing complete")
    print(f"{'='*60}")
    
    # Optionally save combined results
    if args.save_combined:
        for bin_number in bin_numbers:
            bin_results = all_bin_results[bin_number]
            successful_variance = bin_results['variance']
            successful_l1 = bin_results['l1']
            successful_kappa = bin_results['kappa']
            
            if not successful_variance or not successful_l1 or not successful_kappa:
                print(f"No successful files for bin {bin_number}, skipping combined output")
                continue
            
            # Generate default output paths if not specified
            combined_output_base = args.combined_output
            map_suffix = "baryonified" if args.baryonified else "nobaryons"
            dataset_name = "fiducial" if args.fiducial else "halofit"
            
            if args.no_noise:
                noise_str = ""
            else:
                noise_str = f"_noisy_s{args.noise_level:.2f}"
            
            if not combined_output_base:
                combined_variance_output = os.path.join(
                    base_dir, 
                    f"all_variances_{dataset_name}_{map_suffix}_bin{bin_number}_theta{args.theta:.1f}{noise_str}.npy"
                )
                combined_l1_output = os.path.join(
                    base_dir,
                    f"all_l1_norms_{dataset_name}_{map_suffix}_bin{bin_number}_theta{args.theta:.1f}{noise_str}.npy"
                )
                combined_kappa_output = os.path.join(
                    base_dir,
                    f"all_kappas_{dataset_name}_{map_suffix}_bin{bin_number}_theta{args.theta:.1f}{noise_str}.npy"
                )
            else:
                # If custom output is specified, create separate files for variance, L1, and kappa
                base, ext = os.path.splitext(combined_output_base)
                if len(bin_numbers) > 1:
                    combined_variance_output = f"{base}_variance_bin{bin_number}_theta{args.theta:.1f}{ext}"
                    combined_l1_output = f"{base}_l1_bin{bin_number}_theta{args.theta:.1f}{ext}"
                    combined_kappa_output = f"{base}_kappa_bin{bin_number}_theta{args.theta:.1f}{ext}"
                else:
                    combined_variance_output = f"{base}_variance{ext}"
                    combined_l1_output = f"{base}_l1{ext}"
                    combined_kappa_output = f"{base}_kappa{ext}"
            
            print(f"\nBin {bin_number}: Loading and combining {len(successful_variance)} result files...")
            
            # Load all variance, L1 and kappa results
            all_variances = []
            all_l1_norms = []
            all_kappas = []
            skipped_files = 0
            
            # Use the preserved triples to ensure alignment
            for var_path, l1_path, kappa_path in tqdm(all_bin_results[bin_number]['triples'], 
                                                     desc=f"Loading bin {bin_number} results",
                                                     total=len(all_bin_results[bin_number]['triples'])):
                try:
                    variance_data = np.load(var_path, allow_pickle=True)
                    l1_data = np.load(l1_path, allow_pickle=True)
                    kappa_data = np.load(kappa_path, allow_pickle=True)

                    # Variance is a scalar, L1 norm is a vector, kappa is vector
                    all_variances.append(variance_data)
                    all_l1_norms.append(l1_data)
                    all_kappas.append(kappa_data)

                except Exception as e:
                    skipped_files += 1
                    if args.verbose:
                        print(f"Error loading {os.path.basename(var_path)} or {os.path.basename(l1_path)} or {os.path.basename(kappa_path)}: {e}")
            
            # Convert lists to numpy arrays
            if all_variances and all_l1_norms and all_kappas:
                all_variances = np.array(all_variances)
                all_l1_norms = np.array(all_l1_norms)  # Will be shape (n_files, nbins)
                all_kappas = np.array(all_kappas)      # Will be shape (n_files, nbins)
                
                print(f"Bin {bin_number} variance shape: {all_variances.shape}")
                print(f"Bin {bin_number} L1 norm shape: {all_l1_norms.shape}")
                print(f"Bin {bin_number} kappa shape: {all_kappas.shape}")
                
                # Save combined arrays
                np.save(combined_variance_output, all_variances)
                np.save(combined_l1_output, all_l1_norms)
                np.save(combined_kappa_output, all_kappas)
                
                print(f"Saved combined variances to: {combined_variance_output}")
                print(f"Saved combined L1 norms to: {combined_l1_output}")
                print(f"Saved combined kappa arrays to: {combined_kappa_output}")
                
                if skipped_files > 0:
                    print(f"Note: {skipped_files} files were skipped during combination.")
            else:
                print(f"No valid files found for bin {bin_number} combined output!")
    
    dataset_label = "fiducial cosmology" if args.fiducial else "Halofit selection"
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Total simulations processed: {len(file_paths)} ({dataset_label})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
