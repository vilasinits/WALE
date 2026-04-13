#!/usr/bin/env python3
"""Compare cosmoSLICS L1 simulation datavectors against theory predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wale.DatavectorDiagnostics import utc_timestamp, write_json

GROUP_CHOICES = ("all", "group_1_10", "group_11_25")
BIN_CHOICES = (4, 5)


def parse_scale_indices(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise ValueError("--scale-indices cannot be empty.")
    indices = np.array([int(v) for v in values], dtype=int)
    if np.any(indices < 0):
        raise ValueError(f"--scale-indices must be non-negative, got {indices.tolist()}")
    unique_sorted = np.unique(indices)
    return unique_sorted


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cosmoSLICS L1 NPZ datavectors with theory datavectors and "
            "compute residual diagnostics."
        )
    )
    parser.add_argument(
        "--sim-npz",
        type=str,
        required=True,
        help="Path to simulation NPZ from l1_norm_processing_cosmoslics_starlet.py",
    )

    parser.add_argument(
        "--theory-npz",
        type=str,
        default=None,
        help="Optional path to theory NPZ when theory arrays are in a separate file.",
    )
    parser.add_argument(
        "--theory-key-prefix",
        type=str,
        default=None,
        help=(
            "Prefix for theory arrays inside selected theory NPZ. "
            "Expected key format: <prefix>_bin4 / <prefix>_bin5."
        ),
    )

    parser.add_argument(
        "--bin",
        type=int,
        choices=BIN_CHOICES,
        required=True,
        help="Tomographic bin to compare (4 or 5).",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=GROUP_CHOICES,
        default="all",
        help="Optional row subset by realization group.",
    )
    parser.add_argument(
        "--scale-indices",
        type=str,
        default=None,
        help="Optional comma-separated starlet scale indices (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path for diagnostics.",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="Optional JSON summary manifest path.",
    )

    args = parser.parse_args()

    if (args.theory_npz is None) == (args.theory_key_prefix is None):
        parser.error(
            "Provide exactly one theory source option: either --theory-npz OR --theory-key-prefix."
        )

    try:
        args.scale_indices = parse_scale_indices(args.scale_indices)
    except ValueError as exc:
        parser.error(str(exc))
    return args



def ensure_2d_rows(arr: np.ndarray, key_name: str) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(
            f"Expected {key_name} to have at least 2 dimensions (rows, features), got {arr.shape}."
        )
    return arr.reshape(arr.shape[0], -1)



def select_rows_by_group(row_group: np.ndarray, group: str) -> np.ndarray:
    if group == "all":
        return np.ones(row_group.shape[0], dtype=bool)
    return row_group == group



def resolve_sim_array(npz_obj: np.lib.npyio.NpzFile, bin_id: int) -> np.ndarray:
    key = f"l1_bin{bin_id}"
    if key not in npz_obj:
        raise KeyError(f"Simulation NPZ missing required key '{key}'.")
    return np.asarray(npz_obj[key], dtype=np.float64)



def resolve_theory_array(
    npz_obj: np.lib.npyio.NpzFile,
    theory_path: Path,
    bin_id: int,
    prefix: str | None,
) -> tuple[np.ndarray, str]:
    if prefix is None:
        key = f"l1_bin{bin_id}"
    else:
        key = f"{prefix}_bin{bin_id}"

    if key not in npz_obj:
        available = ", ".join(sorted(npz_obj.files[:20]))
        raise KeyError(
            f"Theory NPZ '{theory_path}' missing key '{key}'. "
            f"Available keys (first 20): {available}"
        )
    return np.asarray(npz_obj[key], dtype=np.float64), key



def validate_shapes(
    sim_raw_shape: tuple[int, ...],
    theory_raw_shape: tuple[int, ...],
    sim_selected_shape: tuple[int, ...],
    theory_selected_shape: tuple[int, ...],
    context: str,
) -> None:
    if sim_selected_shape != theory_selected_shape:
        raise ValueError(
            "Theory/simulation shape mismatch after selection "
            f"({context}). Simulation selected shape={sim_selected_shape}, "
            f"theory selected shape={theory_selected_shape}. Raw shapes: "
            f"sim={sim_raw_shape}, theory={theory_raw_shape}."
        )



def compute_metrics(
    mean_sim: np.ndarray,
    mean_theory: np.ndarray,
    abs_residual: np.ndarray,
    rel_residual: np.ndarray,
) -> dict[str, float]:
    residual = mean_theory - mean_sim
    finite_rel = rel_residual[np.isfinite(rel_residual)]
    if finite_rel.size == 0:
        mean_abs_relative_error = float("nan")
        max_abs_relative_error = float("nan")
    else:
        mean_abs_relative_error = float(np.mean(np.abs(finite_rel)))
        max_abs_relative_error = float(np.max(np.abs(finite_rel)))

    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_relative_error": mean_abs_relative_error,
        "max_abs_relative_error": max_abs_relative_error,
        "mean_abs_residual": float(np.mean(abs_residual)),
    }



def main() -> None:
    args = parse_arguments()

    sim_path = Path(args.sim_npz).expanduser().resolve()
    theory_path = (
        Path(args.theory_npz).expanduser().resolve()
        if args.theory_npz is not None
        else sim_path
    )
    output_path = Path(args.output).expanduser().resolve()
    if args.manifest_output:
        manifest_path = Path(args.manifest_output).expanduser().resolve()
    else:
        manifest_path = None

    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation NPZ not found: {sim_path}")
    if not theory_path.exists():
        raise FileNotFoundError(f"Theory NPZ not found: {theory_path}")

    with np.load(sim_path, allow_pickle=False) as sim_npz:
        sim_arr_raw = resolve_sim_array(sim_npz, args.bin)

        if "row_group" in sim_npz:
            row_group = np.asarray(sim_npz["row_group"]).astype(str)
        else:
            row_group = np.array(["all"] * sim_arr_raw.shape[0], dtype=str)

        sim_centers_key = f"l1_bin{args.bin}_centers"
        sim_centers = (
            np.asarray(sim_npz[sim_centers_key], dtype=np.float64)
            if sim_centers_key in sim_npz
            else None
        )

    with np.load(theory_path, allow_pickle=False) as theory_npz:
        theory_arr_raw, theory_key = resolve_theory_array(
            theory_npz, theory_path, args.bin, args.theory_key_prefix
        )

    if sim_arr_raw.shape[0] != row_group.shape[0]:
        raise ValueError(
            "Simulation row_group length mismatch: "
            f"row_group={row_group.shape[0]}, l1_bin{args.bin} rows={sim_arr_raw.shape[0]}"
        )

    if sim_arr_raw.ndim != 3:
        raise ValueError(
            f"Expected simulation array to be 3D (rows, scales, features), got {sim_arr_raw.shape}"
        )
    if theory_arr_raw.ndim != 3:
        raise ValueError(
            f"Expected theory array to be 3D (rows, scales, features), got {theory_arr_raw.shape}"
        )
    if sim_arr_raw.shape[0] != theory_arr_raw.shape[0]:
        raise ValueError(
            "Theory/simulation row-count mismatch before selection: "
            f"sim rows={sim_arr_raw.shape[0]}, theory rows={theory_arr_raw.shape[0]} "
            f"(sim shape={sim_arr_raw.shape}, theory shape={theory_arr_raw.shape})."
        )
    if sim_arr_raw.shape[1:] != theory_arr_raw.shape[1:]:
        raise ValueError(
            "Theory/simulation per-row shape mismatch before selection: "
            f"sim per-row shape={sim_arr_raw.shape[1:]}, "
            f"theory per-row shape={theory_arr_raw.shape[1:]}. "
            f"Full shapes: sim={sim_arr_raw.shape}, theory={theory_arr_raw.shape}."
        )

    row_mask = select_rows_by_group(row_group, args.group)
    if not np.any(row_mask):
        raise ValueError(f"No rows selected for group '{args.group}'.")

    sim_selected = sim_arr_raw[row_mask]
    theory_selected = theory_arr_raw[row_mask]

    if args.scale_indices is not None:
        max_scale = sim_selected.shape[1] - 1
        if np.any(args.scale_indices > max_scale):
            raise ValueError(
                f"Scale indices {args.scale_indices.tolist()} out of range; "
                f"available scale indices are 0..{max_scale}."
            )
        sim_selected = sim_selected[:, args.scale_indices, :]
        theory_selected = theory_selected[:, args.scale_indices, :]

    validate_shapes(
        sim_raw_shape=sim_arr_raw.shape,
        theory_raw_shape=theory_arr_raw.shape,
        sim_selected_shape=sim_selected.shape,
        theory_selected_shape=theory_selected.shape,
        context=f"bin={args.bin}, group={args.group}, scales={args.scale_indices}",
    )

    sim_matrix = ensure_2d_rows(sim_selected, f"l1_bin{args.bin}")
    theory_matrix = ensure_2d_rows(theory_selected, theory_key)

    mean_sim = np.mean(sim_matrix, axis=0)
    mean_theory = np.mean(theory_matrix, axis=0)
    residual = mean_theory - mean_sim
    abs_residual = np.abs(residual)
    rel_residual = np.divide(
        residual,
        mean_sim,
        out=np.full_like(residual, np.nan, dtype=np.float64),
        where=np.abs(mean_sim) > 0.0,
    )

    metrics = compute_metrics(mean_sim, mean_theory, abs_residual, rel_residual)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        created_at_utc=np.array([utc_timestamp()]),
        sim_npz=np.array([str(sim_path)]),
        theory_npz=np.array([str(theory_path)]),
        theory_key=np.array([theory_key]),
        selected_bin=np.array([args.bin], dtype=np.int16),
        selected_group=np.array([args.group]),
        selected_scale_indices=(
            np.array([], dtype=np.int16)
            if args.scale_indices is None
            else args.scale_indices.astype(np.int16)
        ),
        n_rows_selected=np.array([sim_matrix.shape[0]], dtype=np.int32),
        n_features=np.array([mean_sim.size], dtype=np.int32),
        mean_sim_datavector=mean_sim.astype(np.float64),
        mean_theory_datavector=mean_theory.astype(np.float64),
        residual=residual.astype(np.float64),
        abs_residual=abs_residual.astype(np.float64),
        rel_residual=rel_residual.astype(np.float64),
        rmse=np.array([metrics["rmse"]], dtype=np.float64),
        mean_abs_relative_error=np.array(
            [metrics["mean_abs_relative_error"]], dtype=np.float64
        ),
        max_abs_relative_error=np.array(
            [metrics["max_abs_relative_error"]], dtype=np.float64
        ),
        mean_abs_residual=np.array([metrics["mean_abs_residual"]], dtype=np.float64),
        l1_centers=(
            np.array([], dtype=np.float64)
            if sim_centers is None
            else np.asarray(sim_centers, dtype=np.float64)
        ),
    )

    print(f"Saved comparison NPZ: {output_path}")
    print(
        "Metrics: "
        f"RMSE={metrics['rmse']:.6g}, "
        f"mean_abs_relative_error={metrics['mean_abs_relative_error']:.6g}, "
        f"max_abs_relative_error={metrics['max_abs_relative_error']:.6g}"
    )

    if manifest_path is not None:
        payload: dict[str, Any] = {
            "created_at_utc": utc_timestamp(),
            "script": "compare_theory_vs_cosmoslics_l1.py",
            "inputs": {
                "sim_npz": str(sim_path),
                "theory_npz": str(theory_path),
                "theory_key": theory_key,
                "selected_bin": int(args.bin),
                "selected_group": args.group,
                "selected_scale_indices": (
                    [] if args.scale_indices is None else args.scale_indices.astype(int).tolist()
                ),
            },
            "counts": {
                "rows_selected": int(sim_matrix.shape[0]),
                "n_features": int(mean_sim.size),
            },
            "metrics": metrics,
            "output_npz": str(output_path),
        }
        write_json(manifest_path, payload)
        print(f"Saved manifest JSON: {manifest_path}")


if __name__ == "__main__":
    main()
