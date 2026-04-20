#!/usr/bin/env python3
"""
NPE Inference Script for Halofit Selection with Starlet Filter
Performs NPE inference on L1 norms computed using starlet wavelet transform
for cosmologies passing Halofit criteria.
"""

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
    parser = argparse.ArgumentParser(
        description="Run NPE inference on Halofit selection with starlet filter"
    )
    
    # Dataset selection
    parser.add_argument("--training-dataset", type=str, 
                        choices=["halofit", "fiducial"],
                        default="halofit",
                        help="Which dataset to use for training (halofit or fiducial)")
    
    parser.add_argument("--data-dir", type=str, 
                        default='/home/tersenov/CosmoGridV1/stage3_forecast',
                        help="Base directory for data")
    
    # Analysis configuration
    bin_group = parser.add_mutually_exclusive_group(required=False)
    bin_group.add_argument("--bin", type=int, default=1, 
                        help="Which redshift bin to analyze")
    bin_group.add_argument("--bins", type=str, 
                        help="Comma-separated list of redshift bins to analyze for tomographic inference")
    
    # Scale selection (starlet has 5 scales)
    scale_group = parser.add_mutually_exclusive_group(required=False)
    scale_group.add_argument("--scale", type=int, default=0, 
                        help="Which scale index to analyze (0-4, 0-indexed)")
    scale_group.add_argument("--scales", type=str, 
                        help="Comma-separated list of scale indices to analyze (0-4, 0-indexed)")
    
    # Datavector bin range selection
    bin_range_group = parser.add_mutually_exclusive_group(required=False)
    bin_range_group.add_argument("--bin-range", type=str,
                        help="Global bin range for all redshift bins in format 'start:end' (both inclusive, 0-indexed)")
    bin_range_group.add_argument("--bin-ranges", type=str,
                        help="Separate bin ranges for each redshift bin in format 'start1:end1,start2:end2,...' (0-indexed)")
    
    # Noise options
    parser.add_argument("--noisy", action="store_true", 
                        help="Use noisy datavectors")
    parser.add_argument("--noise-level", type=float, default=0.26, 
                        help="Noise level for both datavectors and fiducial (when --noisy is set)")
    
    # Mask options
    parser.add_argument("--masked", action="store_true",
                        help="Use masked datavectors (Euclid-like sky mask)")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the sky mask in square degrees (default: 14000)")
    
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
    parser.add_argument("--output-dir", type=str, 
                        default="/home/tersenov/software/bar_impact/outputs/plots",
                        help="Directory to save output plots")
    parser.add_argument("--samples-dir", type=str, 
                        default="/home/tersenov/software/bar_impact/outputs/samples",
                        help="Directory to save posterior samples")
    
    # GPU configuration
    parser.add_argument("--gpu", type=str, default="0", 
                        help="GPU index to use")
    
    args = parser.parse_args()
    
    # Set mask suffixes
    if args.masked:
        area_tag = int(round(args.mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"
        mask_label = f"masked_{area_tag}sqdeg"
    else:
        area_tag = None
        mask_suffix = ""
        mask_label = ""
    
    args.mask_area_tag = area_tag
    args.mask_suffix = mask_suffix
    args.mask_label = mask_label
    
    return args


def parse_bin_ranges(args, num_redshift_bins):
    """Parse bin range arguments and return list of (start, end) tuples for each redshift bin."""
    if args.bin_range:
        # Global bin range for all redshift bins
        try:
            start, end = map(int, args.bin_range.split(':'))
            bin_ranges = [(start, end)] * num_redshift_bins
            print(f"Using global bin range [{start}:{end}] for all redshift bins")
        except ValueError:
            raise ValueError("Global bin range must be in format 'start:end' (e.g., '10:30')")
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
                raise ValueError("Bin ranges must be in format 'start1:end1,start2:end2,...' (e.g., '10:30,15:35')")
    else:
        # No bin range specified, use all bins
        bin_ranges = None
        print("No bin range specified, using all datavector bins")
    
    return bin_ranges


def run_tarp_coverage_test(posterior, combined_data_vector, params, args):
    """
    Run TARP coverage test on the posterior estimator.
    
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
    
    # Determine base directory and params file based on training dataset
    if args.training_dataset == "halofit":
        base_dir = os.path.join(args.data_dir, "grid")
        params_filename = "selected_params_halofit.npy"
        params_path = os.path.join("/home/tersenov/software/bar_impact/data", params_filename)
        dataset_suffix = "halofit"
    else:  # fiducial
        base_dir = os.path.join(args.data_dir, "fiducial", "cosmo_fiducial")
        params_filename = "selected_params_fiducial.npy"
        params_path = os.path.join("/home/tersenov/software/bar_impact/data", params_filename)
        dataset_suffix = "fiducial"
    
    # Parse bin options
    if args.bins:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
        is_multi_bin = True
    else:
        bin_indices = [args.bin]
        bin_desc = f"bin{args.bin}"
        is_multi_bin = False
    
    # Construct datavector paths
    # Include "starlet" in filenames to match processing script output
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    snr_suffix = "_snr-13to13"  # Default SNR range from l1_norm_processing_halofit_starlet.py
    l1_paths = []
    
    for bin_idx in bin_indices:
        bin_spec = f"bin{bin_idx}"
        
        # Path for Halofit or fiducial data with starlet processing
        if args.training_dataset == "halofit":
            l1_filename = f"all_l1_norms_starlet_halofit_baryonified_{bin_spec}{args.mask_suffix}{noise_suffix}{snr_suffix}.npy"
        else:
            l1_filename = f"all_l1_norms_starlet_fiducial_nobaryons_{bin_spec}{args.mask_suffix}{noise_suffix}{snr_suffix}.npy"
        
        l1_path = os.path.join(base_dir, l1_filename)
        l1_paths.append(l1_path)
    
    # Set bin description for output files
    bin_spec_for_output = bin_desc
    
    return params_path, l1_paths, bin_spec_for_output, dataset_suffix


def main():
    args = parse_arguments()
    
    # Construct file paths
    params_path, l1_paths, bin_spec, dataset_suffix = construct_paths(args)
    print(f"Using parameters file: {params_path}")
    print(f"Using datavector files: {l1_paths}")
    print(f"Training dataset: {args.training_dataset}")
    
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Device used by jax:", jax.devices())

    # Load cosmological parameters
    params = np.load(params_path, allow_pickle=True)
    print(f"Loaded parameters, shape: {params.shape}")

    # Load and process data from each bin
    l1_full_bins = []
    for l1_path in l1_paths:
        l1_full = np.load(l1_path, allow_pickle=True)
        l1_full_bins.append(l1_full)
        print(f"Loaded data from {l1_path}, shape: {l1_full.shape}")

    # Determine number of redshift bins for bin range parsing
    num_redshift_bins = len(l1_full_bins)
    
    # Parse bin ranges
    bin_ranges = parse_bin_ranges(args, num_redshift_bins)

    # Extract scale data - either single scale or multiple scales
    if args.scales:
        # Parse comma-separated scales
        scale_indices = [int(s.strip()) for s in args.scales.split(',')]
        scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
        print(f"Using multiple scales: {[s+1 for s in scale_indices]}")
        
        # Process each bin's data with selected scales
        bin_data_list = []
        for i, l1_full in enumerate(l1_full_bins):
            # Extract and concatenate scales for this bin
            l1_scales = []
            for scale_idx in scale_indices:
                scale_data = l1_full[:, scale_idx]
                
                # Apply bin range if specified
                if bin_ranges:
                    start_bin, end_bin = bin_ranges[i]
                    scale_data = scale_data[:, start_bin:end_bin+1]  # +1 because end is inclusive
                    print(f"Applied bin range [{start_bin}:{end_bin}] to redshift bin {i+1}, scale {scale_idx+1}")
                
                l1_scales.append(scale_data)
            
            # Concatenate along feature dimension (axis=1)
            bin_data = np.concatenate([scale_data.reshape(scale_data.shape[0], -1) 
                                      for scale_data in l1_scales], axis=1)
            bin_data_list.append(bin_data)
        
        # Now concatenate all bins together
        l1_scale = np.concatenate(bin_data_list, axis=1)
        
    else:
        # Single scale case
        scale_desc = f"scale{args.scale+1}"
        print(f"Using single {scale_desc}")
        
        # Process each bin's data with the selected scale
        bin_data_list = []
        for i, l1_full in enumerate(l1_full_bins):
            scale_data = l1_full[:, args.scale]
            
            # Apply bin range if specified
            if bin_ranges:
                start_bin, end_bin = bin_ranges[i]
                scale_data = scale_data[:, start_bin:end_bin+1]  # +1 because end is inclusive
                print(f"Applied bin range [{start_bin}:{end_bin}] to redshift bin {i+1}, scale {args.scale+1}")
            
            bin_data_list.append(scale_data)
        
        # Now concatenate all bins together
        l1_scale = np.concatenate(bin_data_list, axis=1)
    
    print(f"Combined datavector shape: {l1_scale.shape}")

    # Convert to JAX arrays
    params = jnp.array(params)
    l1_scale = jnp.array(l1_scale)

    # Create checkpoint path
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a descriptive checkpoint name based on data configuration and training params
    datavector_desc = f"halofit_starlet_{dataset_suffix}_{bin_spec}_{scale_desc}"
    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        datavector_desc += f"_{args.mask_label}"
    if bin_ranges:
        if args.bin_range:
            # Global bin range
            start, end = bin_ranges[0]  # All ranges are the same
            datavector_desc += f"_dvrange{start}-{end}"
        else:
            # Individual bin ranges
            range_desc = "_dvranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
            datavector_desc += range_desc
    # Add training parameters for tracking
    datavector_desc += f"_ep{args.epochs}_bs{args.batch_size}"
    
    checkpoint_name = f"cosmoGRID_starlet_weights_{datavector_desc}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f"Checkpoint path: {checkpoint_path}")

    # Initialize NPE
    inference = NPE()
    inference = inference.append_simulations(params, l1_scale)
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
        ecp, alpha = run_tarp_coverage_test(posterior, l1_scale, params, args)
        
        # Create filename base for coverage plots with more descriptive info
        coverage_filename_base = f"tarp_halofit_starlet_{dataset_suffix}_{bin_spec}_{scale_desc}"
        if args.noisy:
            coverage_filename_base += f"_noisy_s{args.noise_level:.2f}"
        if args.masked:
            coverage_filename_base += f"_{args.mask_label}"
        if bin_ranges:
            if args.bin_range:
                start, end = bin_ranges[0]
                coverage_filename_base += f"_dvrange{start}-{end}"
            else:
                range_desc = "_dvranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
                coverage_filename_base += range_desc
        # Add coverage test parameters
        coverage_filename_base += f"_nsims{args.coverage_num_sims}_nsamp{args.coverage_num_samples}"
        
        plot_tarp_coverage(ecp, alpha, args, args.output_dir, coverage_filename_base)

    # For fiducial inference, use the mean of the fiducial data as observation
    # For Halofit, we would typically use a different fiducial observation
    # Here we'll compute the mean across all samples as the "fiducial" observation
    fid_mean_scale = np.mean(l1_scale, axis=0)
    print(f"Fiducial observation shape: {fid_mean_scale.shape}")

    # Sample from the posterior
    print("Sampling from posterior...")
    num_samples = args.num_samples
    master_key = random.PRNGKey(args.random_seed)
    sample_key, master_key = jax.random.split(master_key)
    samples = posterior.sample(
        x=fid_mean_scale, num_samples=num_samples, key=sample_key
    )
    print(f"Generated {num_samples} samples")

    # True parameters for plotting
    true_params = jnp.array([[2.600e-01, 8.400e-01, -1.000e+00, 6.736e+01, 9.649e-01, 4.930e-02]])

    # Create visualization
    labels = [r"$\Omega_{m}$", r"$S_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
    
    # Create descriptive sample label
    sample_label = f"{dataset_suffix} starlet, {bin_spec}, {scale_desc}"
    if args.noisy:
        sample_label += f", noisy s={args.noise_level:.2f}"
    
    samples_bin_scale = MCSamples(
        samples=samples,
        names=labels,
        label=sample_label,
    )

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4

    g.triangle_plot([samples_bin_scale], filled=True,
                   line_args=[{'color': 'blue'}],
                   contour_colors=['blue'],
                   markers={
                       label: val for label, val in zip(labels, true_params[0])
                   })

    # Save plot with descriptive filename
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Include more run parameters in filename for tracking
    plot_filename = f"posterior_halofit_starlet_{dataset_suffix}_{bin_spec}_{scale_desc}"
    if args.noisy:
        plot_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        plot_filename += f"_{args.mask_label}"
    if bin_ranges:
        if args.bin_range:
            # Global bin range
            start, end = bin_ranges[0]  # All ranges are the same
            plot_filename += f"_dvrange{start}-{end}"
        else:
            # Individual bin ranges
            range_desc = "_dvranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
            plot_filename += range_desc
    # Add number of samples for reference
    plot_filename += f"_nsamp{args.num_samples}"
    plot_filename += ".pdf"
    
    plt.savefig(os.path.join(args.output_dir, plot_filename), transparent=True)
    print(f"Saved plot to {os.path.join(args.output_dir, plot_filename)}")

    # Save posterior samples with descriptive filename
    os.makedirs(args.samples_dir, exist_ok=True)
    
    # Include more run parameters in filename for tracking
    samples_filename = f"posterior_samples_halofit_starlet_{dataset_suffix}_{bin_spec}_{scale_desc}"
    if args.noisy:
        samples_filename += f"_noisy_s{args.noise_level:.2f}"
    if args.masked:
        samples_filename += f"_{args.mask_label}"
    if bin_ranges:
        if args.bin_range:
            # Global bin range
            start, end = bin_ranges[0]  # All ranges are the same
            samples_filename += f"_dvrange{start}-{end}"
        else:
            # Individual bin ranges
            range_desc = "_dvranges" + "-".join([f"{start}-{end}" for start, end in bin_ranges])
            samples_filename += range_desc
    # Add number of samples for reference
    samples_filename += f"_nsamp{args.num_samples}"
    samples_filename += "_npe.npy"
    
    np.save(os.path.join(args.samples_dir, samples_filename), samples_bin_scale.samples)
    print(f"Saved posterior samples to {os.path.join(args.samples_dir, samples_filename)}")

    print("Done!")


if __name__ == "__main__":
    main()
