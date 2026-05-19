#!/usr/bin/env python3
"""
Quantify posterior agreement across the 2x2 (training, fiducial) design.

For each (theta, lmax) configuration the four contour files are:
    A: sim-trained,    sim-fid       (the "ground truth" SBI)
    B: sim-trained,    theory-fid    (off-diagonal: probes theory mean bias)
    C: theory-trained, sim-fid       (off-diagonal: probes Gaussian-cov surrogate)
    D: theory-trained, theory-fid    (theory-only self-consistency)

We report, per parameter, the shift of each off-diagonal posterior mean from
the diagonal A in units of A's std (i.e. delta_mu / sigma_A). Pass threshold
~0.3 sigma; >0.5 sigma is a warning.

Usage
-----
    python scripts/inference/compute_posterior_shifts.py [--samples-dir DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

PARAM_NAMES = ["Om", "sigma8", "w0", "H0", "ns", "Ob"]
TRUE_PARAMS = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> dict | None:
    """Extract (training, fiducial, theta, ratio, lmax) tags from a sample filename.

    Filename convention (see run_npe_inference_cls.py:300-353):
        samples_cls_{sim_stem}_fid_{fid_stem}{_lmaxN|}_npe.npy
    where sim_stem and fid_stem come from the input .npz stems.

    Returns None for files that don't match the DoTH C_ell sweep pattern.
    """
    if not name.endswith("_npe.npy"):
        return None
    if "doth" not in name:
        return None

    body = name[len("samples_cls_"):-len("_npe.npy")]
    # Split at "_fid_" -- both halves carry their own theta/ratio tags
    if "_fid_" not in body:
        return None
    sim_stem, rest = body.split("_fid_", 1)

    # Suffixes (any order at end): _lmaxN, _tfNNN (NNN uses 'p' for decimal point)
    tf_thresh = None
    m_tf = re.search(r"_tf([0-9p]+)$", rest)
    if m_tf:
        tf_thresh = float(m_tf.group(1).replace("p", "."))
        rest = rest[: m_tf.start()]

    lmax = None
    m_lmax = re.search(r"_lmax(\d+)$", rest)
    if m_lmax:
        lmax = int(m_lmax.group(1))
        rest = rest[: m_lmax.start()]

    fid_stem = rest

    # Identify training type
    if sim_stem.startswith("sim_doth_cls"):
        training = "sim"
    elif sim_stem.startswith("theory_doth_cls"):
        training = "theory"
    else:
        return None

    # Identify fiducial type
    if fid_stem.startswith("sim_doth_cls"):
        fiducial = "sim"
    elif fid_stem.startswith("theory_doth_cls"):
        fiducial = "theory"
    else:
        return None

    # Extract theta and ratio (present in both stems, take from sim stem)
    m_theta = re.search(r"theta([\d.]+)_ratio([\d.]+)", sim_stem)
    if not m_theta:
        return None
    theta = float(m_theta.group(1))
    ratio = float(m_theta.group(2))

    return dict(
        training=training,
        fiducial=fiducial,
        theta=theta,
        ratio=ratio,
        lmax=lmax,
        tf_thresh=tf_thresh,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) along the realization axis (axis 0), shape (n_params,)."""
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)


def sigma_shift(mu: np.ndarray, sigma_ref: np.ndarray, mu_ref: np.ndarray) -> np.ndarray:
    """Return (mu - mu_ref) / sigma_ref per parameter."""
    return (mu - mu_ref) / sigma_ref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse_config(
    files: dict[tuple[str, str], Path],
    theta: float, ratio: float, lmax: int | None, tf_thresh: float | None,
) -> None:
    """Print a per-configuration summary table."""
    label_lmax = "full ell" if lmax is None else f"lmax={lmax}"
    label_tf = "" if tf_thresh is None else f"  tf>={tf_thresh:g}"
    print(f"\n=== theta={theta:.1f} ratio={ratio:.1f}  {label_lmax}{label_tf} ===")

    summaries: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, p in files.items():
        s = np.load(p)
        summaries[key] = summarize(s)

    if ("sim", "sim") not in summaries:
        print("  [skip] no (sim, sim) reference for this config")
        return

    mu_A, sig_A = summaries[("sim", "sim")]

    # Header
    print(f"  {'param':<8} {'true':>9}   "
          f"{'A (s,s)':>15} {'B (s,t)':>15}   "
          f"{'C (t,s)':>15} {'D (t,t)':>15}   "
          f"|  shifts vs A in sigma_A units")
    print(f"  {'':<8} {'':>9}   "
          f"{'mu ± sig':>15} {'mu ± sig':>15}   "
          f"{'mu ± sig':>15} {'mu ± sig':>15}   "
          f"|  B  C  D")

    rows = []
    for i, name in enumerate(PARAM_NAMES):
        mu_A_i, sig_A_i = mu_A[i], sig_A[i]
        cells = []
        shifts = []
        for key, label in [
            (("sim", "sim"), "A"),
            (("sim", "theory"), "B"),
            (("theory", "sim"), "C"),
            (("theory", "theory"), "D"),
        ]:
            if key in summaries:
                mu_k, sig_k = summaries[key]
                cells.append(f"{mu_k[i]:.4f}±{sig_k[i]:.4f}")
                if label in ("B", "C", "D"):
                    shifts.append((mu_k[i] - mu_A_i) / sig_A_i)
            else:
                cells.append("—")
                if label in ("B", "C", "D"):
                    shifts.append(float("nan"))

        shift_str = "  ".join(
            f"{s:+.2f}" if np.isfinite(s) else "  — " for s in shifts
        )
        print(f"  {name:<8} {TRUE_PARAMS[i]:>9.4f}   "
              f"{cells[0]:>15} {cells[1]:>15}   "
              f"{cells[2]:>15} {cells[3]:>15}   |  {shift_str}")
        rows.append((name, mu_A_i, sig_A_i, *shifts))

    max_abs = max(abs(r[3]) for r in rows if np.isfinite(r[3]))
    max_abs_off = max(
        max(abs(r[3]), abs(r[4])) for r in rows
        if np.isfinite(r[3]) and np.isfinite(r[4])
    )
    print(f"  max |off-diagonal shift|: {max_abs_off:.2f} sigma_A")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-dir", type=Path,
        default=REPO_ROOT / "outputs" / "samples",
    )
    args = parser.parse_args()

    by_config: dict[tuple[float, float, int | None, float | None], dict[tuple[str, str], Path]] = {}
    for p in sorted(args.samples_dir.glob("*_npe.npy")):
        info = parse_filename(p.name)
        if info is None:
            continue
        cfg = (info["theta"], info["ratio"], info["lmax"], info["tf_thresh"])
        key = (info["training"], info["fiducial"])
        by_config.setdefault(cfg, {})[key] = p

    if not by_config:
        print("No DoTH NPE sample files found in", args.samples_dir)
        return

    print("Posterior-mean shifts across the 2x2 (training, fiducial) design.")
    print("Reference for sigma units is contour A = (sim training, sim fiducial).")
    print("A=sim-train+sim-fid, B=sim-train+theory-fid (off-diag),")
    print("C=theory-train+sim-fid (off-diag), D=theory-train+theory-fid.")
    print("Threshold: ~0.3 sigma pass, >0.5 sigma warning.")

    def cfg_sort_key(c):
        theta, ratio, lmax, tf = c
        return (
            theta, ratio,
            99999 if lmax is None else lmax,
            -1.0 if tf is None else tf,
        )

    for cfg in sorted(by_config.keys(), key=cfg_sort_key):
        analyse_config(by_config[cfg], *cfg)

    print()


if __name__ == "__main__":
    main()
