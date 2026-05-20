#!/usr/bin/env python3
"""
Generic NPE inference on a binned datavector (C_ell, ell_1-norm, …).

A data-vector-agnostic refactor of `run_npe_inference_cls.py`:

  --data-key   : NPZ field name for the datavector matrix (default "cls").
                  For ell_1-norm work, pass --data-key l1_norms.
  --axis-key   : NPZ field name for the per-bin axis values (default "ells").
                  For ell_1-norm work, pass --axis-key kappa_bins.
  --axis-min   : keep only bins with axis_value >= axis_min.
  --axis-max   : keep only bins with axis_value <= axis_max.
                 --lmax is preserved as a back-compat alias of --axis-max
                 when --data-key == "cls".

GPU pinning defaults to CUDA_VISIBLE_DEVICES=1 (override via --gpu).

Output naming: `samples_<data_key>_<sim_stem>_fid_<fid_stem><cuts>_npe.npy`.

This script is **additive**; the older `run_npe_inference_cls.py` is unchanged
to keep the existing C_ell sweep drivers reproducible.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from jaxili.inference import NPE


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUE_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
PARAM_LABELS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic NPE inference on a binned datavector.")

    # Data sources
    parser.add_argument("--sim-cls-file", type=Path, required=True,
                        help="Training NPZ. Must contain --data-key and --axis-key arrays + 'params'.")
    parser.add_argument("--fiducial-cls-file", type=Path, required=True,
                        help="Fiducial NPZ; the data-key field will be averaged across rows for the observation.")

    parser.add_argument("--data-key", type=str, default="cls",
                        help="NPZ field name for the datavector (default 'cls').")
    parser.add_argument("--axis-key", type=str, default="ells",
                        help="NPZ field name for the bin-axis values (default 'ells').")

    # Bin-cut on the axis
    parser.add_argument("--axis-min", type=float, default=None,
                        help="Keep bins with axis_value >= axis_min.")
    parser.add_argument("--axis-max", type=float, default=None,
                        help="Keep bins with axis_value <= axis_max.")
    parser.add_argument("--lmax", type=float, default=None,
                        help="Back-compat alias of --axis-max for --data-key cls.")

    # Optional transfer-function mask (C_ell only)
    parser.add_argument("--transfer-file", type=Path, default=None)
    parser.add_argument("--transfer-threshold", type=float, default=None)

    # Training
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(REPO_ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    # Sampling
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--variance-threshold", type=float, default=1e-10)
    parser.add_argument("--remove-dc", action="store_true",
                        help=("Subtract the per-row mean from each training datavector "
                              "and from the fiducial observation before standardisation. "
                              "Useful for the DC-offset sanity test (L1)."))

    # Output (default: routed under outputs/plots/{cls,l1/thetaT}/posteriors)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for posterior plots. Default: auto by data-key/theta.")
    parser.add_argument("--samples-dir", type=str,
                        default=str(REPO_ROOT / "outputs" / "samples"))

    # GPU pinning (defaults to GPU 1 per the L1 plan)
    parser.add_argument("--gpu", type=str, default="1")

    args = parser.parse_args()
    if args.lmax is not None and args.axis_max is None:
        args.axis_max = args.lmax
    return args


def standardise(train: np.ndarray, obs: np.ndarray):
    mu = train.mean(axis=0)
    sigma = train.std(axis=0).clip(1e-30)
    return (train - mu) / sigma, (obs - mu) / sigma, mu, sigma


def remove_zero_variance_features(data, threshold=1e-10, verbose=True):
    feature_var = np.var(data, axis=0)
    mask = feature_var > threshold
    if verbose:
        print(f"Variance filter: dropped {int((~mask).sum())}/{len(mask)} features "
              f"(threshold={threshold:.1e})")
    return data[:, mask], mask


def _format_cut_tag(args) -> str:
    """Build a short, filename-safe tag describing the axis cuts."""
    if args.data_key == "cls":
        tags = []
        if args.axis_max is not None:
            tags.append(f"_lmax{int(args.axis_max)}")
        if args.transfer_threshold is not None:
            tags.append(f"_tf{args.transfer_threshold:g}".replace(".", "p"))
        if getattr(args, "remove_dc", False):
            tags.append("_dcrm")
        return "".join(tags)

    def _fmt(v: float) -> str:
        return f"{v:+.4g}".replace(".", "p").replace("+", "p").replace("-", "m")
    tags = []
    if args.axis_min is not None:
        tags.append(f"_amin{_fmt(args.axis_min)}")
    if args.axis_max is not None:
        tags.append(f"_amax{_fmt(args.axis_max)}")
    if args.transfer_threshold is not None:
        tags.append(f"_tf{args.transfer_threshold:g}".replace(".", "p"))
    if getattr(args, "remove_dc", False):
        tags.append("_dcrm")
    return "".join(tags)


def _default_output_dir(args, sim_file: Path) -> str:
    """Resolve the default output dir based on data_key and theta from the sim file."""
    if args.output_dir is not None:
        return args.output_dir
    if args.data_key == "cls":
        return str(REPO_ROOT / "outputs" / "plots" / "cls" / "posteriors")
    # L1 (or any non-cls): try to read theta from the sim NPZ for routing
    try:
        with np.load(sim_file, allow_pickle=True) as d:
            theta = float(d["theta"]) if "theta" in d.files else None
    except Exception:
        theta = None
    if theta is None:
        return str(REPO_ROOT / "outputs" / "plots" / "l1" / "posteriors")
    return str(REPO_ROOT / "outputs" / "plots" / "l1" / f"theta{theta:.0f}" / "posteriors")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("JAX devices:", jax.devices())
    args.output_dir = _default_output_dir(args, Path(args.sim_cls_file))

    # --- Load training data
    sim_file = Path(args.sim_cls_file)
    if not sim_file.exists():
        raise FileNotFoundError(sim_file)
    with np.load(sim_file, allow_pickle=True) as d:
        params = np.asarray(d["params"], dtype=float)
        data_train = np.asarray(d[args.data_key], dtype=float)
        axis_vals = np.asarray(d[args.axis_key], dtype=float)

    print(f"Training file: {sim_file}")
    print(f"  params shape : {params.shape}")
    print(f"  {args.data_key} shape : {data_train.shape}")
    print(f"  {args.axis_key} shape : {axis_vals.shape}; "
          f"range [{axis_vals.min():.4g}, {axis_vals.max():.4g}]")

    # --- Load fiducial data
    fid_file = Path(args.fiducial_cls_file)
    if not fid_file.exists():
        raise FileNotFoundError(fid_file)
    with np.load(fid_file, allow_pickle=True) as d:
        data_fid = np.asarray(d[args.data_key], dtype=float)
        axis_fid = np.asarray(d[args.axis_key], dtype=float)
    print(f"Fiducial file: {fid_file}")
    print(f"  {args.data_key} shape : {data_fid.shape}  ({data_fid.shape[0]} realizations)")

    if not np.allclose(axis_vals, axis_fid, rtol=1e-5):
        raise ValueError(f"{args.axis_key} grids of sim and fiducial do not match.")

    # --- Bin-axis cut
    mask = np.ones(axis_vals.size, dtype=bool)
    if args.axis_min is not None:
        mask &= axis_vals >= float(args.axis_min)
    if args.axis_max is not None:
        mask &= axis_vals <= float(args.axis_max)
    if args.transfer_threshold is not None:
        if args.transfer_file is None:
            raise ValueError("--transfer-threshold needs --transfer-file.")
        with np.load(args.transfer_file, allow_pickle=True) as d:
            transfer = np.asarray(d["doth_transfer"], dtype=float)
            ells_tf = np.asarray(d["ells"], dtype=float)
        if not np.allclose(axis_vals, ells_tf, rtol=1e-5):
            raise ValueError("Transfer-file axis does not match training axis.")
        thresh = float(args.transfer_threshold) * np.max(np.abs(transfer))
        mask &= np.abs(transfer) >= thresh
    if not mask.all():
        data_train = data_train[:, mask]
        data_fid = data_fid[:, mask]
        axis_used = axis_vals[mask]
        print(f"Kept {mask.sum()}/{mask.size} bins "
              f"(range [{axis_used.min():.4g}, {axis_used.max():.4g}])")

    # --- Fiducial obs mean
    valid_fid = ~np.all(np.isnan(data_fid), axis=1)
    fid_obs = np.nanmean(data_fid[valid_fid], axis=0)
    print(f"Fiducial observation: mean over {int(valid_fid.sum())} realizations")

    # --- Optional DC-offset removal (per-row mean over bins) --------------------
    if args.remove_dc:
        train_dc = data_train.mean(axis=1, keepdims=True)
        fid_dc = float(fid_obs.mean())
        data_train = data_train - train_dc
        fid_obs = fid_obs - fid_dc
        print(f"DC-removed: training |mean DC| {np.mean(np.abs(train_dc)):.3e}, "
              f"fiducial DC {fid_dc:.3e}")

    # --- Standardise + variance filter
    data_train, fid_obs, mu, sig = standardise(data_train, fid_obs)
    data_train, vmask = remove_zero_variance_features(data_train, args.variance_threshold)
    fid_obs = fid_obs[vmask]
    print(f"Training shape after filtering : {data_train.shape}")

    if params.shape[0] != data_train.shape[0]:
        raise ValueError(f"params rows {params.shape[0]} != train rows {data_train.shape[0]}")

    # --- Tags & paths
    cut_tag = _format_cut_tag(args)
    sim_stem = sim_file.stem
    fid_stem = fid_file.stem
    checkpoint_name = f"npe_{args.data_key}_{sim_stem}{cut_tag}"
    checkpoint_path = os.path.join(os.path.abspath(args.checkpoint_dir), checkpoint_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stem = f"{args.data_key}_{sim_stem}_fid_{fid_stem}{cut_tag}"
    print(f"Checkpoint: {checkpoint_path}")

    # --- Train or load
    params_jax = jnp.array(params)
    data_jax = jnp.array(data_train)
    if args.train:
        inference = NPE().append_simulations(params_jax, data_jax)
        print(f"Training for {args.epochs} epochs ...")
        inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            training_batch_size=args.batch_size,
        )
        print("Training complete.")
    else:
        n_params = params_jax.shape[1]
        n_feat = data_jax.shape[1]
        exmp_input = (jnp.zeros((1, n_params)), jnp.zeros((1, n_feat)))
        ckpt_base = Path(checkpoint_path) / "NDE_w_Standardization"
        versions = sorted(ckpt_base.glob("version_*"),
                          key=lambda p: int(p.name.split("_")[1])) if ckpt_base.exists() else []
        if not versions:
            print(f"No checkpoints found at {ckpt_base}. Run with --train.")
            return
        ckpt_versioned = str(versions[-1])
        print(f"Loading {ckpt_versioned}")
        inference = NPE.load_from_checkpoints(ckpt_versioned, exmp_input)

    posterior = inference.build_posterior()

    # --- Sample
    fid_jax = jnp.array(fid_obs)
    master_key = random.PRNGKey(args.random_seed)
    sample_key, _ = jax.random.split(master_key)
    samples = posterior.sample(x=fid_jax, num_samples=args.num_samples, key=sample_key)
    print(f"Generated {args.num_samples} samples.")

    # --- Save
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_path = os.path.join(args.samples_dir, f"samples_{stem}_npe.npy")
    np.save(samples_path, np.array(samples))
    print(f"Saved samples: {samples_path}")

    # --- Plot
    os.makedirs(args.output_dir, exist_ok=True)
    mc = MCSamples(samples=np.array(samples), names=PARAM_LABELS,
                   label=f"{args.data_key} NPE")
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4
    g.triangle_plot([mc], filled=True,
                    markers={lbl: val for lbl, val in zip(PARAM_LABELS, TRUE_PARAMS)})
    plot_path = os.path.join(args.output_dir, f"posterior_{stem}.pdf")
    plt.savefig(plot_path, transparent=True)
    plt.close()
    print(f"Saved plot: {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
