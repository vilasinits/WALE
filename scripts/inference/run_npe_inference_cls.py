#!/usr/bin/env python3
"""
NPE inference using C_ell power spectra as datavectors.

Training data  : sim_cls_bin<N>_nobaryons.npz  (fields: params, cls, ells)
Fiducial       : mean over sim_cls_bin<N>_fiducial.npz  (field: cls)
ell selection  : all bins with ell_center <= --lmax (use all when --lmax is omitted)
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

tarp_path = str(
    Path(__file__).resolve().parents[2] / "tarp" / "src"
)
if tarp_path not in sys.path:
    sys.path.insert(0, tarp_path)
from tarp import get_tarp_coverage


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUE_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
PARAM_LABELS = [r"$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_cls = REPO_ROOT / "data" / "cls"
    parser = argparse.ArgumentParser(
        description="NPE inference on C_ell power spectra."
    )

    # Data
    parser.add_argument(
        "--sim-cls-file", type=Path,
        default=repo_cls / "simulations" / "sim_cls_bin4_nobaryons.npz",
        help="NPZ with training cls (fields: params, cls, ells).",
    )
    parser.add_argument(
        "--theory-fiducial", action="store_true",
        help=(
            "Use the theory C_ell as fiducial observation instead of the mean "
            "of simulated fiducial realizations."
        ),
    )
    parser.add_argument(
        "--fiducial-cls-file", type=Path,
        default=None,
        help=(
            "NPZ with fiducial cls (field: cls, ells). Fiducial obs = mean over rows. "
            "Defaults to sim_cls_bin4_fiducial.npz or theory_cls_bin4_fiducial.npz "
            "depending on --theory-fiducial."
        ),
    )
    parser.add_argument(
        "--lmax", type=float, default=None,
        help="Keep only ell bins with ell_center <= lmax. Default: use all bins.",
    )
    parser.add_argument(
        "--transfer-file", type=Path, default=None,
        help=(
            "Optional NPZ containing a 'doth_transfer' array (W_DoTH^2 * W_pix^2 "
            "at each ell). Used together with --transfer-threshold to mask bins."
        ),
    )
    parser.add_argument(
        "--transfer-threshold", type=float, default=None,
        help=(
            "Mask ell bins where |doth_transfer| < threshold * max(|doth_transfer|). "
            "Typical: 0.01 (1%% of peak). Combined with --lmax if both are set."
        ),
    )

    # Training
    parser.add_argument("--train", action="store_true",
                        help="Train a new model (otherwise load from checkpoint).")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(REPO_ROOT / "outputs" / "checkpoints"),
                        help="Directory for model checkpoints.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    # Sampling
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=1)

    # Preprocessing
    parser.add_argument("--variance-threshold", type=float, default=1e-10,
                        help="Drop features with variance below this threshold.")

    # Coverage test
    parser.add_argument("--run-coverage-test", action="store_true")
    parser.add_argument("--coverage-num-sims", type=int, default=100)
    parser.add_argument("--coverage-num-samples", type=int, default=1000)
    parser.add_argument("--coverage-bootstrap", action="store_true")
    parser.add_argument("--coverage-num-bootstrap", type=int, default=100)
    parser.add_argument("--coverage-seed", type=int, default=42)

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "outputs" / "plots"))
    parser.add_argument("--samples-dir", type=str,
                        default=str(REPO_ROOT / "outputs" / "samples"))

    # GPU
    parser.add_argument("--gpu", type=str, default="0")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def standardise(
    train: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subtract training mean and divide by training std, feature-wise.

    Returns standardised train array, standardised obs vector, mean, std.
    std is clipped to 1e-30 so division is always safe.
    """
    mu = train.mean(axis=0)
    sigma = train.std(axis=0).clip(1e-30)
    return (train - mu) / sigma, (obs - mu) / sigma, mu, sigma


def remove_zero_variance_features(
    data: np.ndarray, threshold: float = 1e-10, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    feature_variance = np.var(data, axis=0)
    mask = feature_variance > threshold
    n_removed = int(np.sum(~mask))
    if verbose:
        print(f"Variance filter: {n_removed}/{len(mask)} features removed "
              f"(threshold={threshold:.1e})")
    return data[:, mask], mask


# ---------------------------------------------------------------------------
# TARP coverage test
# ---------------------------------------------------------------------------

def run_tarp_coverage_test(posterior, data, params, args):
    print("\n" + "=" * 60)
    print("Running TARP Coverage Test")
    print("=" * 60)

    n_test = min(args.coverage_num_sims, data.shape[0])
    np.random.seed(args.coverage_seed)
    idx = np.random.choice(data.shape[0], size=n_test, replace=False)
    test_data = np.array(data[idx])
    test_params = np.array(params[idx])

    master_key = random.PRNGKey(args.coverage_seed)
    all_samples = []
    for i, x_obs in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_test}")
        sample_key, master_key = jax.random.split(master_key)
        s = posterior.sample(x=x_obs, num_samples=args.coverage_num_samples, key=sample_key)
        all_samples.append(np.array(s))

    all_samples = np.stack(all_samples, axis=1)  # (n_samples, n_sims, n_dims)
    ecp, alpha = get_tarp_coverage(
        samples=all_samples,
        theta=test_params,
        references="random",
        metric="euclidean",
        norm=True,
        bootstrap=args.coverage_bootstrap,
        num_bootstrap=args.coverage_num_bootstrap if args.coverage_bootstrap else 100,
        seed=args.coverage_seed,
    )
    print("TARP done.\n" + "=" * 60)
    return ecp, alpha


def plot_tarp_coverage(ecp, alpha, args, output_dir: str, stem: str) -> None:
    plt.figure(figsize=(6, 6))
    if args.coverage_bootstrap:
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std = np.std(ecp, axis=0)
        plt.plot(alpha, ecp_mean, "b-", lw=2, label="TARP")
        plt.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                         alpha=0.3, color="blue", label="Bootstrap")
    else:
        plt.plot(alpha, ecp, "b-", lw=2, label="TARP")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal")
    plt.xlabel("Credibility Level")
    plt.ylabel("Expected Coverage Probability")
    plt.title("TARP Coverage Diagnostic")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{stem}_tarp_coverage.pdf")
    plt.savefig(path, transparent=True, dpi=300)
    plt.close()
    print(f"Saved TARP plot: {path}")

    data_path = os.path.join(output_dir, f"{stem}_tarp_coverage_data.npz")
    if args.coverage_bootstrap:
        np.savez(data_path, ecp=ecp, alpha=alpha,
                 ecp_mean=np.mean(ecp, axis=0), ecp_std=np.std(ecp, axis=0))
    else:
        np.savez(data_path, ecp=ecp, alpha=alpha)
    print(f"Saved TARP data: {data_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    repo_cls = REPO_ROOT / "data" / "cls"
    if args.fiducial_cls_file is None:
        if args.theory_fiducial:
            args.fiducial_cls_file = repo_cls / "theory" / "theory_cls_bin4_fiducial_nside512_pixwin.npz"
        else:
            args.fiducial_cls_file = repo_cls / "simulations" / "sim_cls_bin4_fiducial.npz"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("JAX devices:", jax.devices())

    # --- Load training data ---------------------------------------------------
    sim_file = Path(args.sim_cls_file)
    if not sim_file.exists():
        raise FileNotFoundError(f"Sim cls file not found: {sim_file}")

    with np.load(sim_file, allow_pickle=True) as d:
        params = np.asarray(d["params"], dtype=float)   # (n_sims, 6)
        cls_train = np.asarray(d["cls"], dtype=float)   # (n_sims, n_ell)
        ells = np.asarray(d["ells"], dtype=float)       # (n_ell,)

    print(f"Loaded training data from {sim_file}")
    print(f"  params shape : {params.shape}")
    print(f"  cls shape    : {cls_train.shape}")
    print(f"  ell range    : [{ells.min():.1f}, {ells.max():.1f}]")

    # --- Load fiducial data ---------------------------------------------------
    fid_file = Path(args.fiducial_cls_file)
    if not fid_file.exists():
        raise FileNotFoundError(f"Fiducial cls file not found: {fid_file}")

    with np.load(fid_file, allow_pickle=True) as d:
        cls_fid = np.asarray(d["cls"], dtype=float)    # (n_perms, n_ell)
        ells_fid = np.asarray(d["ells"], dtype=float)

    print(f"Loaded fiducial data from {fid_file}")
    print(f"  cls shape : {cls_fid.shape}  ({cls_fid.shape[0]} realizations)")

    # Sanity check: ell grids must match
    if not np.allclose(ells, ells_fid, rtol=1e-5):
        raise ValueError("ell grids of sim and fiducial cls files do not match.")

    # --- ell selection --------------------------------------------------------
    ell_mask = np.ones(len(ells), dtype=bool)
    if args.lmax is not None:
        ell_mask &= ells <= float(args.lmax)
        print(f"ell selection (lmax={args.lmax:.0f}): "
              f"keeping {int(ell_mask.sum())}/{len(ells)} bins")
    if args.transfer_threshold is not None:
        if args.transfer_file is None:
            raise ValueError(
                "--transfer-threshold requires --transfer-file pointing to an NPZ "
                "with a 'doth_transfer' array."
            )
        with np.load(args.transfer_file, allow_pickle=True) as d:
            if "doth_transfer" not in d.files:
                raise ValueError(
                    f"{args.transfer_file} does not contain a 'doth_transfer' field."
                )
            transfer = np.asarray(d["doth_transfer"], dtype=float)
            ells_transfer = np.asarray(d["ells"], dtype=float)
        if not np.allclose(ells, ells_transfer, rtol=1e-5):
            raise ValueError("ell grid of transfer file does not match training cls.")
        thresh = float(args.transfer_threshold) * np.max(np.abs(transfer))
        tf_mask = np.abs(transfer) >= thresh
        ell_mask &= tf_mask
        print(f"transfer-fn cut (threshold={args.transfer_threshold:g} of peak |T|={np.max(np.abs(transfer)):.3e}): "
              f"after cut {int(ell_mask.sum())}/{len(ells)} bins")
    if ell_mask.all():
        ells_used = ells
        print(f"Using all {len(ells)} ell bins")
    else:
        cls_train = cls_train[:, ell_mask]
        cls_fid = cls_fid[:, ell_mask]
        ells_used = ells[ell_mask]
        print(f"Kept ell range : [{ells_used.min():.1f}, {ells_used.max():.1f}] "
              f"({len(ells_used)} bins)")

    # --- Fiducial observation: mean over realizations -------------------------
    # Exclude rows that are all-NaN (missing files)
    valid_fid = ~np.all(np.isnan(cls_fid), axis=1)
    fid_obs = np.nanmean(cls_fid[valid_fid], axis=0)   # (n_ell_used,)
    print(f"Fiducial observation: mean over {valid_fid.sum()} realizations")

    # --- Standardise ----------------------------------------------------------
    # Subtract training mean and divide by training std before any filtering.
    # This is essential for NPE and makes the variance threshold meaningful.
    cls_train, fid_obs, cls_mean, cls_std = standardise(cls_train, fid_obs)
    print(f"Standardised cls (mean/std computed from training set)")

    # --- Variance filtering ---------------------------------------------------
    cls_train, valid_mask = remove_zero_variance_features(
        cls_train, threshold=args.variance_threshold
    )
    fid_obs = fid_obs[valid_mask]
    print(f"Training datavector shape after filtering : {cls_train.shape}")
    print(f"Fiducial datavector shape after filtering : {fid_obs.shape}")

    if params.shape[0] != cls_train.shape[0]:
        raise ValueError(
            f"params ({params.shape[0]}) and cls ({cls_train.shape[0]}) row count mismatch."
        )

    # --- Build checkpoint name ------------------------------------------------
    lmax_tag = f"_lmax{int(args.lmax)}" if args.lmax is not None else ""
    tf_tag = (
        f"_tf{args.transfer_threshold:g}".replace(".", "p")
        if args.transfer_threshold is not None else ""
    )
    lmax_tag = lmax_tag + tf_tag
    sim_stem = sim_file.stem          # e.g. sim_cls_bin4_nobaryons
    fid_stem = fid_file.stem          # e.g. sim_cls_bin4_fiducial
    checkpoint_name = f"npe_cls_{sim_stem}{lmax_tag}"
    checkpoint_path = os.path.join(
        os.path.abspath(args.checkpoint_dir), checkpoint_name
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoint: {checkpoint_path}")

    # --- NPE ------------------------------------------------------------------
    params_jax = jnp.array(params)
    cls_jax = jnp.array(cls_train)

    if args.train:
        inference = NPE()
        inference = inference.append_simulations(params_jax, cls_jax)
        print(f"Training for {args.epochs} epochs ...")
        inference.train(
            checkpoint_path=checkpoint_path,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            training_batch_size=args.batch_size,
        )
        print("Training complete.")
    else:
        print("Loading existing model ...")
        n_params = params_jax.shape[1]
        n_features = cls_jax.shape[1]
        exmp_input = (jnp.zeros((1, n_params)), jnp.zeros((1, n_features)))
        # Checkpoints are saved under <base>/NDE_w_Standardization/version_N/
        # Find the latest version directory
        ckpt_base = Path(checkpoint_path) / "NDE_w_Standardization"
        if not ckpt_base.exists():
            print(f"Checkpoint directory not found: {ckpt_base}")
            print("Re-run with --train to train a new model.")
            return
        versions = sorted(ckpt_base.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
        if not versions:
            print(f"No versioned checkpoints found in {ckpt_base}")
            print("Re-run with --train to train a new model.")
            return
        ckpt_versioned = str(versions[-1])
        print(f"  Using checkpoint: {ckpt_versioned}")
        try:
            inference = NPE.load_from_checkpoints(ckpt_versioned, exmp_input)
        except Exception as exc:
            print(f"Failed to load model: {exc}")
            print("Re-run with --train to train a new model.")
            return

    posterior = inference.build_posterior()

    # --- Coverage test --------------------------------------------------------
    stem = f"cls_{sim_stem}_fid_{fid_stem}{lmax_tag}"
    if args.run_coverage_test:
        os.makedirs(args.output_dir, exist_ok=True)
        ecp, alpha = run_tarp_coverage_test(posterior, cls_jax, params_jax, args)
        plot_tarp_coverage(ecp, alpha, args, args.output_dir, stem)

    # --- Sample posterior -----------------------------------------------------
    fid_jax = jnp.array(fid_obs)
    master_key = random.PRNGKey(args.random_seed)
    sample_key, _ = jax.random.split(master_key)
    samples = posterior.sample(x=fid_jax, num_samples=args.num_samples, key=sample_key)
    print(f"Generated {args.num_samples} posterior samples.")

    # --- Plot -----------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    mc = MCSamples(samples=np.array(samples), names=PARAM_LABELS, label="cls NPE")
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4
    g.triangle_plot(
        [mc], filled=True,
        line_args=[{"color": "blue"}],
        contour_colors=["blue"],
        markers={lbl: val for lbl, val in zip(PARAM_LABELS, TRUE_PARAMS)},
    )
    plot_path = os.path.join(args.output_dir, f"posterior_{stem}.pdf")
    plt.savefig(plot_path, transparent=True)
    print(f"Saved plot: {plot_path}")

    # --- Save samples ---------------------------------------------------------
    os.makedirs(args.samples_dir, exist_ok=True)
    samples_path = os.path.join(args.samples_dir, f"samples_{stem}_npe.npy")
    np.save(samples_path, np.array(samples))
    print(f"Saved samples: {samples_path}")

    print("Done.")


if __name__ == "__main__":
    main()
