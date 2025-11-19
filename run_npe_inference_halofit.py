#!/usr/bin/env python3
# filepath: /home/tersenov/software/bar_impact/scripts/run_npe_inference_halofit.py

import os
import sys
import argparse
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
from jaxili.inference import NPE
from getdist import plots, MCSamples

# Add tarp package to path if needed
tarp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tarp', 'src')
if tarp_path not in sys.path:
    sys.path.insert(0, tarp_path)
from tarp import get_tarp_coverage

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NPE inference on Halofit selection or fiducial cosmology")
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, 
                        default='/home/tersenov/CosmoGridV1/stage3_forecast',
                        help="Base directory for data")
    
    # Dataset selection
    parser.add_argument("--training-dataset", type=str, choices=["halofit", "fiducial"],
                        default="halofit",
                        help="Which dataset to use for training (halofit selection or fiducial)")
    
    parser.add_argument("--simulation-type", type=str, choices=["baryonified", "nobaryons"],
                        default="baryonified", 
                        help="Type of simulation to use for training (baryonified or nobaryons)")
    
    # Analysis configuration
    bin_group = parser.add_mutually_exclusive_group(required=False)
    bin_group.add_argument("--bin", type=int, default=1, 
                        help="Which redshift bin to analyze")
    bin_group.add_argument("--bins", type=str, 
                        help="Comma-separated list of redshift bins to analyze for tomographic inference")
    
    # Datavector bin range selection (for kappa values)
    bin_range_group = parser.add_mutually_exclusive_group(required=False)
    bin_range_group.add_argument("--bin-range", type=str,
                        help="Global bin range for all redshift bins in format 'start:end' (both inclusive, 0-indexed)")
    bin_range_group.add_argument("--bin-ranges", type=str,
                        help="Separate bin ranges for each redshift bin in format 'start1:end1,start2:end2,...' (0-indexed)")
    
    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    
    # Smoothing scale used when computing datavectors (must match preprocessing runs)
    parser.add_argument("--theta", type=float, default=15.0,
                        help="Smoothing scale (theta) used for the datavectors; must match preprocessing runs")
    parser.add_argument("--thetas", type=str, default=None,
                        help="Comma-separated list of thetas (e.g. '15.0,30.0'). If provided, overrides --theta and datavectors will be concatenations of each theta's features.")
    
    # Fiducial configuration  
    parser.add_argument("--fiducial-dataset", type=str, choices=["halofit", "fiducial"],
                        default="fiducial",
                        help="Which dataset to use for fiducial observation (halofit or fiducial)")
    
    parser.add_argument("--fiducial-type", type=str, choices=["baryonified", "nobaryons"],
                        default=None,  # Will default to match simulation-type if not specified
                        help="Type of fiducial (baryonified or nobaryons). If not specified, matches --simulation-type")
    
    # Training parameters
    parser.add_argument("--train", action="store_true", 
                        help="Train model (if not specified, will try to load existing model)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--epochs", type=int, default=1000, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=40, 
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                        help="Learning rate")
    
    # Sampling parameters
    parser.add_argument("--num-samples", type=int, default=3000, 
                        help="Number of posterior samples to generate")
    parser.add_argument("--random-seed", type=int, default=1, 
                        help="Random seed for sampling")
    
    # Coverage testing parameters
    parser.add_argument("--run-coverage-test", action="store_true",
                        help="Run TARP coverage test to assess posterior quality")
    parser.add_argument("--coverage-num-sims", type=int, default=100,
                        help="Number of simulations to use for coverage testing (default: 100)")
    parser.add_argument("--coverage-num-samples", type=int, default=1000,
                        help="Number of posterior samples per simulation for coverage testing (default: 1000)")
    parser.add_argument("--coverage-bootstrap", action="store_true",
                        help="Use bootstrap to estimate coverage uncertainties")
    parser.add_argument("--coverage-num-bootstrap", type=int, default=100,
                        help="Number of bootstrap iterations for coverage uncertainties (default: 100)")
    parser.add_argument("--coverage-seed", type=int, default=42,
                        help="Random seed for coverage testing")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="/home/tersenov/software/bar_impact/outputs/plots",
                        help="Directory to save output plots")
    parser.add_argument("--samples-dir", type=str, default="/home/tersenov/software/bar_impact/outputs/samples",
                        help="Directory to save posterior samples")
    
    # Preprocessing parameters
    parser.add_argument("--variance-threshold", type=float, default=1e-10,
                        help="Variance threshold for removing zero-variance features (default: 1e-10)")
    parser.add_argument("--save-feature-mask", action="store_true",
                        help="Save the feature mask used for preprocessing")
    parser.add_argument("--rebin-factor", type=int, default=1,
                        help="Rebin datavectors by averaging this many adjacent bins (default: 1, no rebinning)")
    
    # GPU configuration
    parser.add_argument("--gpu", type=str, default="0", 
                        help="GPU index to use")
    
    args = parser.parse_args()
    
    # Set fiducial type to match simulation type if not specified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type
    
    return args

def remove_zero_variance_features(data, threshold=1e-10, verbose=True):
    """
    Remove features (columns) with zero or near-zero variance across all samples.
    
    This is important because bins that are always zero or constant can cause
    NaN losses during training.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        threshold: variance threshold below which features are removed
        verbose: if True, print information about removed features
        
    Returns:
        filtered_data: data with zero-variance features removed
        valid_indices: boolean mask of valid (non-zero-variance) features
    """
    # Compute variance across samples for each feature
    feature_variance = np.var(data, axis=0)
    
    # Find features with variance above threshold
    valid_indices = feature_variance > threshold
    
    # Count removed features
    n_removed = np.sum(~valid_indices)
    n_total = len(valid_indices)
    
    if verbose:
        print(f"\nPreprocessing: Removing zero-variance features")
        print(f"  Total features: {n_total}")
        print(f"  Features removed (variance < {threshold}): {n_removed} ({100*n_removed/n_total:.2f}%)")
        print(f"  Features retained: {np.sum(valid_indices)} ({100*np.sum(valid_indices)/n_total:.2f}%)")
        
        # Show some statistics
        if n_removed > 0:
            removed_variances = feature_variance[~valid_indices]
            print(f"  Variance range of removed features: [{removed_variances.min():.2e}, {removed_variances.max():.2e}]")
        if np.sum(valid_indices) > 0:
            kept_variances = feature_variance[valid_indices]
            print(f"  Variance range of kept features: [{kept_variances.min():.2e}, {kept_variances.max():.2e}]")
    
    # Filter data
    filtered_data = data[:, valid_indices]
    
    return filtered_data, valid_indices

def rebin_datavector(data, rebin_factor, verbose=True):
    """
    Rebin datavector by averaging adjacent bins to reduce dimensionality.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        rebin_factor: number of adjacent bins to average together
        verbose: if True, print rebinning information
        
    Returns:
        rebinned_data: data with reduced feature dimension
    """
    if rebin_factor <= 1:
        return data
    
    n_samples, n_features = data.shape
    n_rebinned = n_features // rebin_factor
    
    # Truncate to make it evenly divisible
    n_kept = n_rebinned * rebin_factor
    if n_kept < n_features:
        data = data[:, :n_kept]
        if verbose:
            print(f"\nRebinning: Truncating {n_features - n_kept} features to make evenly divisible by {rebin_factor}")
    
    # Reshape and average
    rebinned_data = data.reshape(n_samples, n_rebinned, rebin_factor).mean(axis=2)
    
    if verbose:
        print(f"\nRebinning datavector:")
        print(f"  Rebin factor: {rebin_factor}")
        print(f"  Original features: {n_features}")
        print(f"  Rebinned features: {n_rebinned}")
        print(f"  Reduction: {100*(1 - n_rebinned/n_features):.1f}%")
    
    return rebinned_data

def parse_bin_ranges(args, num_redshift_bins):
    """Parse bin range arguments and return list of (start, end) tuples for each redshift bin."""
    if args.bin_range:
        # Global bin range for all redshift bins
        try:
            start, end = map(int, args.bin_range.split(':'))
            bin_ranges = [(start, end)] * num_redshift_bins
            print(f"Using global bin range [{start}:{end}] for all redshift bins")
        except ValueError:
            raise ValueError("Global bin range must be in format 'start:end' (e.g., '10:50')")
    elif args.bin_ranges:
        # Separate bin ranges for each redshift bin
        try:
            range_strs = args.bin_ranges.split(',')
            if len(range_strs) != num_redshift_bins:
                raise ValueError(f"Number of bin ranges ({len(range_strs)}) must match number of redshift bins ({num_redshift_bins})")
            
            bin_ranges = []
            for range_str in range_strs:
                start, end = map(int, range_str.strip().split(':'))
                bin_ranges.append((start, end))
            
            print(f"Using separate bin ranges: {bin_ranges}")
        except ValueError as e:
            if "must match number" in str(e):
                raise e
            else:
                raise ValueError("Bin ranges must be in format 'start1:end1,start2:end2,...' (e.g., '10:50,20:60')")
    else:
        # No bin range specified, use all bins
        bin_ranges = None
        print("No bin range specified, using all datavector bins")
    
    return bin_ranges

def run_tarp_coverage_test(posterior, combined_data_vector, params, args):
    """
    Run TARP coverage test on the posterior estimator.
    
    This function samples from the posterior for multiple simulations from the
    training set, then uses TARP to assess whether the posterior coverage is well-calibrated.
    
    Args:
        posterior: Trained posterior object from NPE
        combined_data_vector: Full training data vector (n_sims, n_features)
        params: True parameter values for all simulations (n_sims, n_params)
        args: Command-line arguments
        
    Returns:
        ecp: Expected coverage probability
        alpha: Credibility levels
    """
    print("\n" + "="*60)
    print("Running TARP Coverage Test")
    print("="*60)
    
    # Select subset of simulations for coverage testing
    n_total_sims = combined_data_vector.shape[0]
    n_test_sims = min(args.coverage_num_sims, n_total_sims)
    
    # Randomly select test simulations
    np.random.seed(args.coverage_seed)
    test_indices = np.random.choice(n_total_sims, size=n_test_sims, replace=False)
    
    print(f"Using {n_test_sims} simulations from training set for coverage testing")
    print(f"Generating {args.coverage_num_samples} posterior samples per simulation")
    
    # Extract test data and parameters
    test_data = combined_data_vector[test_indices]
    test_params = params[test_indices]
    
    # Convert to numpy for TARP
    test_data_np = np.array(test_data)
    test_params_np = np.array(test_params)
    
    # Generate posterior samples for each test simulation
    all_samples = []
    master_key = random.PRNGKey(args.coverage_seed)
    
    print("Generating posterior samples for each test simulation...")
    for i, x_obs in enumerate(test_data_np):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_test_sims} simulations")
        
        sample_key, master_key = jax.random.split(master_key)
        samples = posterior.sample(
            x=x_obs, num_samples=args.coverage_num_samples, key=sample_key
        )
        all_samples.append(np.array(samples))
    
    # Stack samples into shape (n_samples, n_sims, n_dims)
    all_samples = np.stack(all_samples, axis=1)
    
    print(f"Posterior samples shape: {all_samples.shape}")
    print(f"True parameters shape: {test_params_np.shape}")
    
    # Compute TARP coverage
    print("\nComputing TARP coverage...")
    ecp, alpha = get_tarp_coverage(
        samples=all_samples,
        theta=test_params_np,
        references="random",
        metric="euclidean",
        num_alpha_bins=None,
        norm=True,
        bootstrap=args.coverage_bootstrap,
        num_bootstrap=args.coverage_num_bootstrap if args.coverage_bootstrap else 100,
        seed=args.coverage_seed
    )
    
    print("TARP coverage computation complete!")
    print("="*60 + "\n")
    
    return ecp, alpha

def plot_tarp_coverage(ecp, alpha, args, output_dir, filename_base):
    """
    Plot TARP coverage diagnostics.
    
    Args:
        ecp: Expected coverage probability from TARP
        alpha: Credibility levels from TARP
        args: Command-line arguments
        output_dir: Directory to save plots
        filename_base: Base filename for saved plot
    """
    plt.figure(figsize=(6, 6))
    
    if args.coverage_bootstrap:
        # ecp has shape (n_bootstrap, n_bins+1)
        # Compute mean and std across bootstrap samples
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std = np.std(ecp, axis=0)
        
        # Plot mean coverage with error band
        plt.plot(alpha, ecp_mean, 'b-', linewidth=2, label='TARP Coverage')
        plt.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std, 
                        alpha=0.3, color='blue', label='Bootstrap uncertainty')
    else:
        # ecp is 1D array
        plt.plot(alpha, ecp, 'b-', linewidth=2, label='TARP Coverage')
    
    # Plot ideal calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Ideal calibration')
    
    # Formatting
    plt.xlabel('Credibility Level', fontsize=12)
    plt.ylabel('Expected Coverage Probability', fontsize=12)
    plt.title('TARP Coverage Diagnostic', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save plot
    coverage_plot_path = os.path.join(output_dir, f"{filename_base}_tarp_coverage.pdf")
    plt.savefig(coverage_plot_path, transparent=True, dpi=300)
    print(f"Saved TARP coverage plot to {coverage_plot_path}")
    
    plt.close()
    
    # Save coverage data
    coverage_data_path = os.path.join(output_dir, f"{filename_base}_tarp_coverage_data.npz")
    if args.coverage_bootstrap:
        np.savez(coverage_data_path, ecp=ecp, alpha=alpha, 
                ecp_mean=ecp_mean, ecp_std=ecp_std,
                bootstrap=True)
    else:
        np.savez(coverage_data_path, ecp=ecp, alpha=alpha, bootstrap=False)
    print(f"Saved TARP coverage data to {coverage_data_path}")

def construct_paths(args):
    """Construct file paths based on provided arguments."""
    
    # Parse bin options
    if args.bins:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
        is_multi_bin = True
    else:
        bin_indices = [args.bin]
        bin_desc = f"bin{args.bin}"
        is_multi_bin = False
    
    # Determine base directories based on dataset type
    if args.training_dataset == "halofit":
        training_base_dir = os.path.join(args.data_dir, "grid")
        # For Halofit, we need the selected parameters file
        params_path = "/home/tersenov/software/bar_impact/data/selected_params_halofit.npy"
    else:  # fiducial
        training_base_dir = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial")
        # For fiducial, use the standard cosmo_params.npy
        params_filename = f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
        params_path = os.path.join(args.data_dir, "grid", params_filename)
    
    if args.fiducial_dataset == "halofit":
        fiducial_base_dir = os.path.join(args.data_dir, "grid")
    else:  # fiducial
        fiducial_base_dir = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial")
    
    # Construct file paths for each bin
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    l1_paths = []
    fiducial_paths = []

    # Determine list of thetas to use. If --thetas provided, parse it; otherwise use single --theta
    if getattr(args, 'thetas', None):
        theta_list = [float(t.strip()) for t in args.thetas.split(',') if t.strip()!='']
        if len(theta_list) == 0:
            theta_list = [args.theta]
    else:
        theta_list = [args.theta]

    for bin_idx in bin_indices:
        bin_spec = f"bin{bin_idx}"

        # For each theta, create a filename and store the full path. We keep a list per bin so
        # callers can load multiple theta-specific datavectors and concatenate them.
        l1_theta_paths = []
        fid_theta_paths = []
        for theta in theta_list:
            l1_filename = f"all_l1_norms_{args.training_dataset}_{args.simulation_type}_{bin_spec}_theta{theta:.1f}{noise_suffix}.npy"
            l1_theta_paths.append(os.path.join(training_base_dir, l1_filename))

            fiducial_filename = f"all_l1_norms_{args.fiducial_dataset}_{args.fiducial_type}_{bin_spec}_theta{theta:.1f}{noise_suffix}.npy"
            fid_theta_paths.append(os.path.join(fiducial_base_dir, fiducial_filename))

        l1_paths.append(l1_theta_paths)
        fiducial_paths.append(fid_theta_paths)

    return params_path, l1_paths, fiducial_paths, bin_desc, theta_list

def main():
    args = parse_arguments()
    
    # Construct file paths (l1_paths and fiducial_paths are lists of lists when multiple thetas are used)
    params_path, l1_paths, fiducial_paths, bin_spec, theta_list = construct_paths(args)
    # Create a compact theta descriptor for filenames (e.g. 'theta15.0_30.0')
    theta_desc_str = "_".join([f"{t:.1f}" for t in theta_list])
    theta_desc_pref = f"theta{theta_desc_str}"
    # keep theta_list available on args for downstream code if needed
    args.theta_list = theta_list
    print(f"Using parameters file: {params_path}")
    print(f"Using training datavector files (per bin, per theta): {l1_paths}")
    print(f"Using fiducial files (per bin, per theta): {fiducial_paths}")
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Device used by jax:", jax.devices())

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)
    print(f"Loaded parameters, shape: {params.shape}")

    # Load and process data from each bin. l1_paths is a list of lists: one inner list per bin,
    # containing one file path per theta. We'll load each theta-specific datavector, optionally
    # apply the per-bin kappa bin_range to each theta, then concatenate the theta features.
    l1_full_bins = []
    for bin_idx, paths_per_bin in enumerate(l1_paths):
        loaded_arrays = []
        for p in paths_per_bin:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Training data file not found: {p}")
            arr = np.load(p, allow_pickle=True)
            loaded_arrays.append(arr)
            print(f"Loaded training data from {p}, shape: {arr.shape}")

        # Ensure consistent number of simulations across thetas
        n_sims = loaded_arrays[0].shape[0]
        for arr in loaded_arrays:
            if arr.shape[0] != n_sims:
                raise ValueError("Inconsistent number of simulations across theta files for the same bin")

        # Determine number of redshift bins for bin range parsing (computed from number of bins)
        # Note: bin_ranges refer to kappa-bin indices per theta; we therefore apply the slice to each theta
        # before concatenation so the resulting features correspond to the requested kappa range for each theta.
        l1_full_bins.append(loaded_arrays)

    # Determine number of redshift bins for bin range parsing
    num_redshift_bins = len(l1_full_bins)

    # Parse bin ranges
    bin_ranges = parse_bin_ranges(args, num_redshift_bins)

    # For each bin, optionally slice each theta's kappa range then concatenate theta-wise features
    bin_data_list = []
    for i, arrays_per_bin in enumerate(l1_full_bins):
        processed_theta_arrays = []
        for arr in arrays_per_bin:
            if bin_ranges:
                start_bin, end_bin = bin_ranges[i]
                arr_proc = arr[:, start_bin:end_bin+1]
            else:
                arr_proc = arr
            processed_theta_arrays.append(arr_proc)

        # Concatenate theta-specific feature blocks for this redshift bin
        bin_data = np.concatenate(processed_theta_arrays, axis=1)
        if bin_ranges:
            print(f"Applied bin range [{start_bin}:{end_bin}] to redshift bin {i+1} for each theta")

        bin_data_list.append(bin_data)

    # Concatenate all bins together along feature dimension
    l1_combined = np.concatenate(bin_data_list, axis=1)
    print(f"Combined training datavector shape (before preprocessing): {l1_combined.shape}")

    # Verify params and data alignment
    if params.shape[0] != l1_combined.shape[0]:
        raise ValueError(f"Mismatch between params ({params.shape[0]}) and data ({l1_combined.shape[0]}) shapes!")

    # Apply rebinning if requested (before variance filtering)
    if args.rebin_factor > 1:
        l1_combined = rebin_datavector(l1_combined, args.rebin_factor, verbose=True)
        print(f"Combined training datavector shape (after rebinning): {l1_combined.shape}")

    # Remove zero-variance features AFTER rebinning to prevent NaN losses
    l1_combined, valid_feature_mask = remove_zero_variance_features(
        l1_combined, 
        threshold=args.variance_threshold, 
        verbose=True
    )
    print(f"Combined training datavector shape (after variance filtering): {l1_combined.shape}")
    
    # Save the valid feature mask for later use with fiducial data
    # This ensures we apply the same filtering to the fiducial observation
    n_features_per_bin = [bd.shape[1] for bd in bin_data_list]
    print(f"Features per bin before filtering: {n_features_per_bin}")
    
    # Optionally save the feature mask
    if args.save_feature_mask:
        os.makedirs(args.output_dir, exist_ok=True)
        mask_filename = f"feature_mask_{args.training_dataset}_{args.simulation_type}_{bin_spec}_{theta_desc_pref}"
        if args.rebin_factor > 1:
            mask_filename += f"_rebin{args.rebin_factor}"
        if args.noisy:
            mask_filename += f"_noisy_s{args.noise_level:.2f}"
        if bin_ranges:
            if args.bin_range:
                start, end = bin_ranges[0]
                mask_filename += f"_binrange{start}-{end}"
            else:
                range_desc = "_binranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
                mask_filename += range_desc
        mask_filename += ".npy"
        mask_path = os.path.join(args.output_dir, mask_filename)
        np.save(mask_path, valid_feature_mask)
        print(f"Saved feature mask to: {mask_path}")

    # Convert to JAX arrays
    params = jnp.array(params)
    l1_combined = jnp.array(l1_combined)

    # Create checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration
    datavector_desc = f"{args.training_dataset}_{args.simulation_type}_{bin_spec}"
    # Include theta(s) used during preprocessing so checkpoint names reflect the run
    datavector_desc += f"_{theta_desc_pref}"
    if args.rebin_factor > 1:
        datavector_desc += f"_rebin{args.rebin_factor}"
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
    if bin_ranges:
        if args.bin_range:
            # Global bin range
            start, end = bin_ranges[0]  # All ranges are the same
            datavector_desc += f"_binrange{start}-{end}"
        else:
            # Individual bin ranges
            range_desc = "_binranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
            datavector_desc += range_desc
    
    checkpoint_name = f"cosmoGRID_l1_norms_weights_{datavector_desc}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Checkpoint path: {checkpoint_path}")

    # Initialize NPE
    inference = NPE()
    inference = inference.append_simulations(params, l1_combined)
    print("Added simulations to NPE")

    # Train or load the model
    if args.train:
        print(f"Starting NPE training for {args.epochs} epochs...")
        metrics, density_estimator = inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            training_batch_size=args.batch_size
        )
        print("Training completed")
    else:
        print("Attempting to load existing model...")
        try:
            inference.load(checkpoint_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Please use --train to train a new model")
            return

    # Build posterior
    posterior = inference.build_posterior()
    print("Built posterior")

    # Run coverage test if requested
    if args.run_coverage_test:
        ecp, alpha = run_tarp_coverage_test(posterior, l1_combined, params, args)
        
        # Create filename base for coverage plots
        coverage_filename_base = f"l1norms_{args.training_dataset}_{args.simulation_type}_{bin_spec}_{theta_desc_pref}"
        if args.rebin_factor > 1:
            coverage_filename_base += f"_rebin{args.rebin_factor}"
        if args.noisy:
            coverage_filename_base += f"_noisy_s{args.noise_level:.2f}"
        if bin_ranges:
            if args.bin_range:
                start, end = bin_ranges[0]
                coverage_filename_base += f"_binrange{start}-{end}"
            else:
                range_desc = "_binranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
                coverage_filename_base += range_desc

        plot_tarp_coverage(ecp, alpha, args, args.output_dir, coverage_filename_base)

    # Load fiducial data for each bin. fiducial_paths is a list of lists (per-bin, per-theta)
    fid_data_list = []
    for i, fid_paths_per_bin in enumerate(fiducial_paths):
        fid_means_per_theta = []
        for p in fid_paths_per_bin:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Fiducial data file not found: {p}")
            fid_full = np.load(p, allow_pickle=True)
            print(f"Loaded fiducial data from {p}, shape: {fid_full.shape}")
            # Average over all fiducial permutations
            fid_mean_theta = np.mean(fid_full, axis=0)
            # Apply bin range if specified (slice per-theta before concatenation)
            if bin_ranges:
                start_bin, end_bin = bin_ranges[i]
                fid_mean_theta = fid_mean_theta[start_bin:end_bin+1]
            fid_means_per_theta.append(fid_mean_theta)

        # Concatenate theta-specific fiducial means for this redshift bin
        fid_concat = np.concatenate(fid_means_per_theta)
        fid_data_list.append(fid_concat)
    
    # Concatenate all bins' fiducial data
    fid_mean_combined = np.concatenate(fid_data_list)
    print(f"Combined fiducial data shape (before preprocessing): {fid_mean_combined.shape}")
    
    # Apply rebinning if requested (must match training data)
    if args.rebin_factor > 1:
        # Fiducial is 1D, so reshape to 2D, rebin, then flatten
        fid_mean_combined = fid_mean_combined.reshape(1, -1)
        fid_mean_combined = rebin_datavector(fid_mean_combined, args.rebin_factor, verbose=False)
        fid_mean_combined = fid_mean_combined.flatten()
        print(f"Combined fiducial data shape (after rebinning): {fid_mean_combined.shape}")
    
    # Apply the same feature filtering as training data
    if fid_mean_combined.shape[0] != valid_feature_mask.shape[0]:
        raise ValueError(
            f"Fiducial data dimension ({fid_mean_combined.shape[0]}) doesn't match "
            f"training data dimension before filtering ({valid_feature_mask.shape[0]}). "
            f"Make sure fiducial and training data have the same bin configuration."
        )
    
    fid_mean_combined = fid_mean_combined[valid_feature_mask]
    print(f"Combined fiducial data shape (after preprocessing): {fid_mean_combined.shape}")
    
    # Verify dimensions match
    if fid_mean_combined.shape[0] != l1_combined.shape[1]:
        raise ValueError(
            f"Fiducial data dimension ({fid_mean_combined.shape[0]}) doesn't match "
            f"training data feature dimension ({l1_combined.shape[1]}) after filtering!"
        )

    # Sample from the posterior
    print("Sampling from posterior...")
    num_samples = args.num_samples
    master_key = random.PRNGKey(args.random_seed)
    sample_key, master_key = jax.random.split(master_key)
    samples = posterior.sample(
        x=fid_mean_combined, num_samples=num_samples, key=sample_key
    )
    print(f"Generated {num_samples} samples")

    # True parameters for plotting (fiducial cosmology)
    true_params = jnp.array([[2.600e-01, 8.400e-01, -1.000e+00, 6.736e+01, 9.649e-01, 4.930e-02]])

    # Create visualization
    labels = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
    
    # Create descriptive sample label
    training_desc = f"{args.training_dataset} {args.simulation_type}"
    fiducial_desc = f"{args.fiducial_dataset} {args.fiducial_type}"
    if args.noisy:
        training_desc += f" n{args.noise_level:.2f}"
        fiducial_desc += f" n{args.noise_level:.2f}"
    
    sample_label = f"{training_desc} DV vs {fiducial_desc} fid, {bin_spec}"
    
    samples_bin = MCSamples(
        samples=samples,
        names=labels,
        label=sample_label,
    )

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4

    g.triangle_plot([samples_bin], filled=True,
                   line_args=[{'color': 'blue'}],
                   contour_colors=['blue'],
                   markers={
                       label: val for label, val in zip(labels, true_params[0])
                   })

    # Save plot with descriptive filename
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_filename = f"posterior_{args.training_dataset}_{args.simulation_type}_vs_{args.fiducial_dataset}_{args.fiducial_type}_{bin_spec}_{theta_desc_pref}"
    if args.rebin_factor > 1:
        plot_filename += f"_rebin{args.rebin_factor}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    if bin_ranges:
        if args.bin_range:
            # Global bin range
            start, end = bin_ranges[0]  # All ranges are the same
            plot_filename += f"_binrange{start}-{end}"
        else:
            # Individual bin ranges
            range_desc = "_binranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
            plot_filename += range_desc
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_filename = f"posterior_samples_{args.training_dataset}_{args.simulation_type}_vs_{args.fiducial_dataset}_{args.fiducial_type}_{bin_spec}_{theta_desc_pref}"
    if args.rebin_factor > 1:
        samples_filename += f"_rebin{args.rebin_factor}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    if bin_ranges:
        if args.bin_range:
            # Global bin range
            start, end = bin_ranges[0]  # All ranges are the same
            samples_filename += f"_binrange{start}-{end}"
        else:
            # Individual bin ranges
            range_desc = "_binranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
            samples_filename += range_desc
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")

if __name__ == "__main__":
    main()
