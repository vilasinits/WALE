#!/usr/bin/env python3
"""
Compute wavelet L1 datavectors for cosmoSLICS mass maps using wl_stats_torch.

This script:
1. Audits the cosmoSLICS mass-map/catalogue structure.
2. Optionally computes n(z)-fingerprint diagnostics for realization groups.
3. Computes wavelet L1 datavectors for tomographic bins (default: 4,5).
4. Saves a canonical .npz bundle with datavectors and row-level metadata.

Realization groups are defined by cone index:
- group_1_10: cones 1..10
- group_11_25: cones 11..25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wale.DatavectorDiagnostics import (
    evaluate_l1_datavector_quality,
    summarize_quality_records,
    utc_timestamp,
    write_json,
)

COSMO_DIR_PATTERN = re.compile(r"^\d{2}_[af]$")
GROUP_1_10 = "group_1_10"
GROUP_11_25 = "group_11_25"
INDEX_COLUMN_NAMES = {"id", "index", "cosmo_index", "cosmology_index", "cosmo_id"}


@dataclass(frozen=True)
class RowRecord:
    row_key: str
    cosmo_index: int
    cosmo_label: str
    suffix: str
    cone: int
    group_label: str
    map_paths: dict[int, Path]
    catalog_path: Path | None
    catalog_rows: int | None


def parse_bins(bins_text: str) -> list[int]:
    bins = [int(x.strip()) for x in bins_text.split(",") if x.strip()]
    if not bins:
        raise ValueError("At least one bin must be provided.")
    for value in bins:
        if value < 1:
            raise ValueError(f"Invalid bin {value}. Bin numbers must be >= 1.")
    return sorted(set(bins))


def cones_for_group(group_choice: str) -> list[int]:
    if group_choice == "1-10":
        return list(range(1, 11))
    if group_choice == "11-25":
        return list(range(11, 26))
    if group_choice == "both":
        return list(range(1, 26))
    raise ValueError(f"Unsupported group choice: {group_choice}")


def group_for_cone(cone: int) -> str:
    if 1 <= cone <= 10:
        return GROUP_1_10
    if 11 <= cone <= 25:
        return GROUP_11_25
    raise ValueError(f"Cone index must be in [1, 25], got {cone}")


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_choice)


def resolve_dtype(dtype_choice: str) -> torch.dtype:
    if dtype_choice == "float32":
        return torch.float32
    if dtype_choice == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype '{dtype_choice}'")


def read_catalog_rows(catalog_path: Path) -> tuple[int, str | None]:
    warning_message = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fits.open(catalog_path, memmap=True) as hdul:
            nrows = int(hdul[1].header["NAXIS2"])
    if caught:
        warning_message = "; ".join(str(w.message) for w in caught)
    return nrows, warning_message


def discover_cosmo_labels(root: Path) -> list[str]:
    labels = [p.name for p in root.iterdir() if p.is_dir() and COSMO_DIR_PATTERN.match(p.name)]
    labels = sorted(labels)
    if not labels:
        raise FileNotFoundError(f"No cosmoSLICS directories found in {root}")
    return labels


def discover_rows(
    mass_maps_root: Path,
    catalog_root: Path,
    bins: list[int],
    group_choice: str,
) -> tuple[list[RowRecord], list[dict[str, Any]], dict[str, int]]:
    labels = discover_cosmo_labels(mass_maps_root)
    selected_cones = cones_for_group(group_choice)

    candidates: list[RowRecord] = []
    issues: list[dict[str, Any]] = []
    counters = {
        "rows_total": 0,
        "rows_candidate": 0,
        "rows_missing_maps": 0,
        "rows_missing_catalog": 0,
        "rows_catalog_warning": 0,
        "rows_catalog_error": 0,
    }

    for label in labels:
        cosmo_index = int(label.split("_")[0])
        suffix = label.split("_")[1]
        for cone in selected_cones:
            counters["rows_total"] += 1
            row_key = f"{label}_cone{cone:02d}"
            group_label = group_for_cone(cone)
            map_paths = {
                bin_id: mass_maps_root / label / f"kappa_map_cone{cone}_zbin{bin_id}.npy"
                for bin_id in bins
            }
            missing_bins = [bin_id for bin_id, path in map_paths.items() if not path.exists()]
            if missing_bins:
                counters["rows_missing_maps"] += 1
                issues.append(
                    {
                        "row_key": row_key,
                        "status": "missing_map",
                        "reason": f"missing bins: {missing_bins}",
                        "cosmo_label": label,
                        "cone": cone,
                    }
                )
                continue

            catalog_path = catalog_root / label / f"GalCatalog_LOS_cone{cone}.fits"
            if not catalog_path.exists():
                counters["rows_missing_catalog"] += 1
                issues.append(
                    {
                        "row_key": row_key,
                        "status": "missing_catalog",
                        "reason": f"missing catalogue: {catalog_path}",
                        "cosmo_label": label,
                        "cone": cone,
                    }
                )
                catalog_rows = None
            else:
                try:
                    catalog_rows, warn_msg = read_catalog_rows(catalog_path)
                    if warn_msg:
                        counters["rows_catalog_warning"] += 1
                        issues.append(
                            {
                                "row_key": row_key,
                                "status": "catalog_warning",
                                "reason": warn_msg,
                                "cosmo_label": label,
                                "cone": cone,
                            }
                        )
                except Exception as exc:
                    counters["rows_catalog_error"] += 1
                    catalog_rows = None
                    issues.append(
                        {
                            "row_key": row_key,
                            "status": "catalog_read_error",
                            "reason": str(exc),
                            "cosmo_label": label,
                            "cone": cone,
                        }
                    )

            candidates.append(
                RowRecord(
                    row_key=row_key,
                    cosmo_index=cosmo_index,
                    cosmo_label=label,
                    suffix=suffix,
                    cone=cone,
                    group_label=group_label,
                    map_paths=map_paths,
                    catalog_path=catalog_path if catalog_path.exists() else None,
                    catalog_rows=catalog_rows,
                )
            )
            counters["rows_candidate"] += 1

    return candidates, issues, counters


def _sample_nz_histogram(catalog_path: Path, edges: np.ndarray, sample_size: int) -> np.ndarray:
    with fits.open(catalog_path, memmap=True) as hdul:
        z_col = hdul[1].data["z_true"]
        n_rows = len(z_col)
        stride = max(1, n_rows // sample_size)
        z_sample = np.asarray(z_col[::stride], dtype=np.float64)
    hist, _ = np.histogram(z_sample, bins=edges)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        raise ValueError(f"Empty n(z) histogram for {catalog_path}")
    return hist / total


def compute_nz_fingerprint(
    catalog_root: Path,
    cosmo_labels: list[str],
    nz_max_files_per_group: int,
    nz_sample_size: int,
    nz_min: float,
    nz_max: float,
    nz_nbins: int,
) -> dict[str, Any]:
    edges = np.linspace(nz_min, nz_max, nz_nbins + 1, dtype=np.float64)
    profiles: dict[str, list[np.ndarray]] = {GROUP_1_10: [], GROUP_11_25: []}
    used_files: dict[str, list[str]] = {GROUP_1_10: [], GROUP_11_25: []}
    warnings_list: list[str] = []

    for label in cosmo_labels:
        for cone in range(1, 26):
            group_label = group_for_cone(cone)
            if len(used_files[group_label]) >= nz_max_files_per_group:
                continue
            catalog_path = catalog_root / label / f"GalCatalog_LOS_cone{cone}.fits"
            if not catalog_path.exists():
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hist = _sample_nz_histogram(
                    catalog_path=catalog_path,
                    edges=edges,
                    sample_size=nz_sample_size,
                )
            if caught:
                warnings_list.extend(str(w.message) for w in caught)
            profiles[group_label].append(hist)
            used_files[group_label].append(str(catalog_path))

        if all(
            len(used_files[group]) >= nz_max_files_per_group for group in (GROUP_1_10, GROUP_11_25)
        ):
            break

    summary: dict[str, Any] = {
        "edges": edges,
        "group_profiles": {},
        "used_files": used_files,
        "warnings": warnings_list,
    }
    for group in (GROUP_1_10, GROUP_11_25):
        if profiles[group]:
            arr = np.stack(profiles[group], axis=0)
            summary["group_profiles"][group] = {
                "count": int(arr.shape[0]),
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
            }
        else:
            summary["group_profiles"][group] = {
                "count": 0,
                "mean": np.zeros(nz_nbins, dtype=np.float64),
                "std": np.zeros(nz_nbins, dtype=np.float64),
            }
    return summary


def _is_integer_like(values: np.ndarray, atol: float = 1e-8) -> bool:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return False
    if not np.all(np.isfinite(values)):
        return False
    return np.allclose(values, np.rint(values), rtol=0.0, atol=atol)


def _parse_comment_header_names(path: Path) -> list[str] | None:
    last_comment_line: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                last_comment_line = stripped[1:].strip()
                continue
            break

    if not last_comment_line:
        return None

    tokens = [token.strip() for token in re.split(r"[,\s]+", last_comment_line) if token.strip()]
    return tokens or None


def _load_text_table(path: Path) -> tuple[np.ndarray, list[str] | None]:
    header_names = _parse_comment_header_names(path)
    suffix = path.suffix.lower()
    delimiter_candidates = [","] if suffix == ".csv" else [None, ",", ";", "\t"]
    tried: list[str] = []

    for delimiter in delimiter_candidates:
        delimiter_label = "whitespace" if delimiter is None else repr(delimiter)
        tried.append(delimiter_label)
        try:
            data = np.loadtxt(path, comments="#", delimiter=delimiter, dtype=np.float64)
        except Exception:
            continue

        array = np.asarray(data, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            continue
        if array.shape[1] == 0:
            raise ValueError(f"No columns found in parameter table {path}")
        return array, header_names

    raise ValueError(
        f"Could not parse text parameter table {path} using delimiters {tried}. "
        "Expected a numeric 2D table (optionally with '#' header comments)."
    )


def _extract_numeric_table(
    data: np.ndarray,
    source: str,
    header_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if data.dtype.names:
        field_names = list(data.dtype.names)
        index_field = next((name for name in field_names if name.lower() in INDEX_COLUMN_NAMES), None)
        value_fields = [name for name in field_names if name != index_field]
        if not value_fields:
            raise ValueError(f"No parameter columns found in structured table from {source}")
        table = np.column_stack([np.asarray(data[name], dtype=np.float64) for name in value_fields])
        index_values = None
        if index_field is not None:
            index_values = np.asarray(data[index_field], dtype=np.float64)
            if index_values.shape[0] != table.shape[0]:
                raise ValueError(f"Index length mismatch in structured table from {source}")
        return table, value_fields, index_values

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D numeric parameter array in {source}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Parameter array in {source} must be numeric, got dtype {array.dtype}")
    array = np.asarray(array, dtype=np.float64)
    ncols = array.shape[1]
    if ncols == 0:
        raise ValueError(f"Parameter array in {source} must have at least one column")

    names: list[str]
    index_values: np.ndarray | None = None

    if header_names is not None:
        if len(header_names) not in {ncols, ncols - 1}:
            raise ValueError(
                f"Header in {source} has {len(header_names)} columns, but data has {ncols} columns."
            )

        if len(header_names) == ncols and header_names[0].strip().lower() in INDEX_COLUMN_NAMES:
            index_values = np.asarray(array[:, 0], dtype=np.float64)
            array = array[:, 1:]
            names = header_names[1:]
        elif len(header_names) == ncols - 1:
            if not _is_integer_like(array[:, 0]):
                raise ValueError(
                    f"Header in {source} has {len(header_names)} parameter names for {ncols} data columns, "
                    "but first column is not integer-like ID/index."
                )
            index_values = np.asarray(array[:, 0], dtype=np.float64)
            array = array[:, 1:]
            names = header_names
        else:
            names = header_names
    else:
        names = [f"param_{idx}" for idx in range(ncols)]

    if array.shape[1] != len(names):
        raise ValueError(
            f"Parameter name count mismatch in {source}: {len(names)} names for {array.shape[1]} columns."
        )

    if array.shape[1] == 0:
        raise ValueError(f"No parameter columns found in {source}")

    return array, names, index_values


def _align_parameter_rows_by_index(
    table: np.ndarray,
    names: list[str],
    index_values: np.ndarray | None,
    expected_ncosmo: int,
    source: str,
) -> tuple[np.ndarray, list[str]]:
    if index_values is None and table.shape[1] >= 2 and _is_integer_like(table[:, 0]):
        candidate_ids = np.rint(table[:, 0]).astype(int)
        unique_ids = np.unique(candidate_ids)
        required_ids = np.arange(expected_ncosmo, dtype=int)
        if (
            np.all(candidate_ids >= 0)
            and unique_ids.size == candidate_ids.size
            and np.all(np.isin(required_ids, candidate_ids))
        ):
            index_values = np.asarray(table[:, 0], dtype=np.float64)
            table = table[:, 1:]
            if len(names) == table.shape[1] + 1:
                if all(re.fullmatch(r"param_\d+", name) for name in names):
                    names = [f"param_{idx}" for idx in range(table.shape[1])]
                else:
                    names = names[1:]
            else:
                names = [f"param_{idx}" for idx in range(table.shape[1])]

    if index_values is None:
        if table.shape[0] != expected_ncosmo:
            raise ValueError(
                f"Parameter table must have {expected_ncosmo} rows (one per cosmology), "
                f"got {table.shape[0]}."
            )
        return table, names

    index_values = np.asarray(index_values, dtype=np.float64)
    if index_values.ndim != 1 or index_values.shape[0] != table.shape[0]:
        raise ValueError(
            f"Index column in {source} has shape {index_values.shape}, expected ({table.shape[0]},)."
        )
    if not _is_integer_like(index_values):
        raise ValueError(f"Index column in {source} must be integer-like, got values {index_values[:5]}")

    ids = np.rint(index_values).astype(int)
    if np.any(ids < 0):
        bad = np.unique(ids[ids < 0])
        raise ValueError(f"Index column in {source} contains negative IDs: {bad.tolist()}")

    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    if duplicate_ids.size > 0:
        raise ValueError(f"Index column in {source} contains duplicate IDs: {duplicate_ids.tolist()}")

    order = np.argsort(ids)
    ids_sorted = ids[order]
    table_sorted = table[order]
    required_ids = np.arange(expected_ncosmo, dtype=int)

    missing_ids = required_ids[~np.isin(required_ids, ids_sorted)]
    if missing_ids.size > 0:
        raise ValueError(
            f"Parameter table {source} is missing required cosmology IDs "
            f"{missing_ids.tolist()} (expected IDs 0..{expected_ncosmo - 1})."
        )

    extra_ids = ids_sorted[~np.isin(ids_sorted, required_ids)]
    if extra_ids.size > 0:
        print(
            f"[params] Ignoring extra cosmology IDs in {source}: {extra_ids.tolist()} "
            f"(expected IDs 0..{expected_ncosmo - 1})."
        )

    kept_mask = np.isin(ids_sorted, required_ids)
    kept_ids = ids_sorted[kept_mask]
    kept_rows = table_sorted[kept_mask]
    row_by_id = {int(cosmo_id): row for cosmo_id, row in zip(kept_ids, kept_rows)}
    aligned_table = np.stack([row_by_id[int(cosmo_id)] for cosmo_id in required_ids], axis=0)
    return aligned_table, names


def load_cosmology_parameters(
    params_file: str | None,
    params_key: str | None,
    expected_ncosmo: int,
    param_names_override: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if params_file is None:
        empty_params = np.empty((expected_ncosmo, 0), dtype=np.float64)
        empty_names = np.array([], dtype=str)
        return empty_params, empty_names

    params_path = Path(params_file)
    if not params_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_path}")

    suffix = params_path.suffix.lower()
    header_names: list[str] | None = None

    if suffix == ".npy":
        loaded = np.load(params_path, allow_pickle=True)
        table, names, index_values = _extract_numeric_table(loaded, str(params_path))
    elif suffix == ".npz":
        with np.load(params_path, allow_pickle=True) as archive:
            keys = list(archive.keys())
            if not keys:
                raise ValueError(f"No arrays found in {params_path}")
            selected_key = params_key
            if selected_key is None:
                preferred = ["params", "cosmo_params", "parameters"]
                selected_key = next((key for key in preferred if key in archive), None)
                if selected_key is None:
                    two_d_numeric = [
                        key
                        for key in keys
                        if np.asarray(archive[key]).ndim == 2
                        and np.issubdtype(np.asarray(archive[key]).dtype, np.number)
                    ]
                    if len(two_d_numeric) != 1:
                        raise ValueError(
                            "Could not infer params key from npz. "
                            f"Available keys: {keys}. Use --params-key."
                        )
                    selected_key = two_d_numeric[0]
            if selected_key not in archive:
                raise ValueError(
                    f"Requested params key '{selected_key}' not present in {params_path}. "
                    f"Available keys: {keys}"
                )
            table, names, index_values = _extract_numeric_table(
                np.asarray(archive[selected_key]),
                f"{params_path}:{selected_key}",
            )
    elif suffix in {".dat", ".txt", ".csv"}:
        loaded, header_names = _load_text_table(params_path)
        table, names, index_values = _extract_numeric_table(
            loaded,
            str(params_path),
            header_names=header_names,
        )
    else:
        raise ValueError(
            f"Unsupported parameter file suffix '{params_path.suffix}'. "
            "Use .npy, .npz, .dat, .txt, or .csv."
        )

    table, names = _align_parameter_rows_by_index(
        table=np.asarray(table, dtype=np.float64),
        names=names,
        index_values=index_values,
        expected_ncosmo=expected_ncosmo,
        source=str(params_path),
    )

    if not names:
        names = [f"param_{idx}" for idx in range(table.shape[1])]
    if len(names) != table.shape[1]:
        raise ValueError(
            f"Inferred parameter names in {params_path} have length {len(names)} "
            f"for {table.shape[1]} parameter columns."
        )

    if param_names_override:
        override_names = [name.strip() for name in param_names_override.split(",") if name.strip()]
        if len(override_names) != table.shape[1]:
            raise ValueError(
                f"--param-names provided {len(override_names)} names, "
                f"but parameter table has {table.shape[1]} columns."
            )
        names = override_names

    return np.asarray(table, dtype=np.float64), np.array(names, dtype=str)


def configure_wl_stats_torch(wl_stats_path: str):
    wl_path = Path(wl_stats_path).expanduser().resolve()
    if str(wl_path) not in sys.path:
        sys.path.insert(0, str(wl_path))
    try:
        from wl_stats_torch import WLStatistics  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import wl_stats_torch from '{wl_path}'. "
            "Set --wl-stats-path to the package root (directory containing wl_stats_torch/)."
        ) from exc
    return WLStatistics


def compute_l1_from_map(
    map_path: Path,
    stats,
    noise_sigma: float,
    peak_min_snr: float,
    peak_max_snr: float,
    peak_nbins: int,
    l1_min_snr: float,
    l1_max_snr: float,
    l1_nbins: int,
    clamp_overflow: bool,
    l1_binning: str,
) -> tuple[np.ndarray, np.ndarray]:
    map_data = np.load(map_path, allow_pickle=False)
    if map_data.ndim != 2:
        raise ValueError(f"Expected 2D mass map at {map_path}, got shape {map_data.shape}")
    image = torch.from_numpy(np.asarray(map_data))

    results = stats.compute_all_statistics(
        image=image,
        noise_sigma=noise_sigma,
        mask=None,
        min_snr=peak_min_snr,
        max_snr=peak_max_snr,
        n_bins=peak_nbins,
        l1_nbins=l1_nbins,
        l1_min_snr=l1_min_snr,
        l1_max_snr=l1_max_snr,
        l1_binning=l1_binning,
        compute_mono=False,
        verbose=False,
        clamp_overflow=clamp_overflow,
    )
    l1_values = np.stack(
        [tensor.detach().cpu().numpy() for tensor in results["wavelet_l1_norms"]],
        axis=0,
    ).astype(np.float64)
    l1_bins = np.stack(
        [tensor.detach().cpu().numpy() for tensor in results["l1_bins"]],
        axis=0,
    ).astype(np.float64)
    return l1_values, l1_bins


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosmoSLICS wavelet L1 datavectors for bins 4/5 using wl_stats_torch."
    )
    parser.add_argument(
        "--catalog-root",
        type=str,
        default="/nas/tersenov/cosmoSLICS/cosmoSLICS",
        help="Root directory containing cosmoSLICS catalog directories (00_a, 00_f, ...).",
    )
    parser.add_argument(
        "--mass-maps-root",
        type=str,
        default="/nas/tersenov/cosmoSLICS/cosmoSLICS/mass_maps",
        help="Root directory containing mass maps grouped by cosmology/suffix.",
    )
    parser.add_argument(
        "--wl-stats-path",
        type=str,
        default="/home/tersenov/software/wl_stats_torch",
        help="Path to wl_stats_torch repository root.",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="4,5",
        help="Comma-separated tomographic bins to process (default: 4,5).",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["1-10", "11-25", "both"],
        default="1-10",
        help="Realization group by cone index. Default: 1-10.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path. Defaults to mass-maps root with descriptive filename.",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="Optional JSON manifest output path (default: output_npz with _manifest.json).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for number of candidate rows (useful for smoke tests).",
    )

    parser.add_argument("--n-scales", type=int, default=5, help="Number of wavelet scales.")
    parser.add_argument(
        "--wavelet-type",
        type=str,
        choices=["starlet", "doth"],
        default="starlet",
        help="Wavelet transform family used by wl_stats_torch.",
    )
    parser.add_argument(
        "--doth-base-radius",
        type=float,
        default=1.0,
        help="Base top-hat radius (pixels) for DoTH when --wavelet-type=doth.",
    )
    parser.add_argument("--pixel-arcmin", type=float, default=0.5, help="Pixel size in arcmin.")
    parser.add_argument("--noise-sigma", type=float, default=0.0146, help="Noise sigma for SNR.")
    parser.add_argument(
        "--peak-min-snr",
        type=float,
        default=-13.0,
        help="SNR min for wavelet peak histograms (not saved, used internally).",
    )
    parser.add_argument(
        "--peak-max-snr",
        type=float,
        default=13.0,
        help="SNR max for wavelet peak histograms (not saved, used internally).",
    )
    parser.add_argument(
        "--peak-nbins",
        type=int,
        default=40,
        help="Number of bins for peak histograms (not saved, used internally).",
    )
    parser.add_argument(
        "--l1-min-snr",
        type=float,
        default=-13.0,
        help="SNR min for L1 vectors.",
    )
    parser.add_argument(
        "--l1-max-snr",
        type=float,
        default=13.0,
        help="SNR max for L1 vectors.",
    )
    parser.add_argument("--l1-nbins", type=int, default=40, help="Number of SNR bins for L1.")
    parser.add_argument(
        "--l1-binning",
        type=str,
        choices=["auto", "snr", "coeff"],
        default="auto",
        help="Quantity used for L1 binning in wl_stats_torch.",
    )
    parser.add_argument(
        "--clamp-overflow",
        action="store_true",
        help="Clamp out-of-range values into edge bins (default: exclude out-of-range).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (e.g. 'auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Computation dtype passed to wl_stats_torch.",
    )
    parser.add_argument(
        "--clear-cuda-cache-every",
        type=int,
        default=0,
        help="If >0 and CUDA is used, clear CUDA cache every N processed rows.",
    )

    parser.add_argument(
        "--quality-policy",
        type=str,
        choices=["warn", "skip", "fail"],
        default="warn",
        help="How to handle problematic datavectors.",
    )
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=-1e-10,
        help="Minimum allowed L1 value before flagging as problematic.",
    )
    parser.add_argument(
        "--wiggle-fraction-threshold",
        type=float,
        default=0.85,
        help="Max sign-change fraction in gradients before flagging as wiggly.",
    )

    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Optional path to 25xN cosmology parameter table (.npy, .npz, .dat, .txt, .csv).",
    )
    parser.add_argument(
        "--params-key",
        type=str,
        default=None,
        help="Optional array key when --params-file points to .npz.",
    )
    parser.add_argument(
        "--param-names",
        type=str,
        default=None,
        help="Optional comma-separated parameter names overriding inferred names.",
    )

    parser.add_argument(
        "--skip-nz-fingerprint",
        action="store_true",
        help="Skip catalogue n(z)-fingerprint diagnostics.",
    )
    parser.add_argument(
        "--nz-max-files-per-group",
        type=int,
        default=10,
        help="Max number of catalogues sampled per group for n(z) fingerprint.",
    )
    parser.add_argument(
        "--nz-sample-size",
        type=int,
        default=300000,
        help="Approximate number of z samples per catalogue fingerprint.",
    )
    parser.add_argument("--nz-min", type=float, default=0.0, help="n(z) histogram min z.")
    parser.add_argument("--nz-max", type=float, default=3.5, help="n(z) histogram max z.")
    parser.add_argument(
        "--nz-nbins",
        type=int,
        default=70,
        help="Number of bins in n(z) fingerprint histograms.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run layout + n(z) audits and write manifest, without L1 processing.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    mass_maps_root = Path(args.mass_maps_root).expanduser().resolve()
    catalog_root = Path(args.catalog_root).expanduser().resolve()
    bins = parse_bins(args.bins)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        group_tag = args.group.replace("-", "_")
        bins_tag = "".join(str(value) for value in bins)
        output_name = f"cosmoslics_starlet_l1_group{group_tag}_bins{bins_tag}.npz"
        output_path = mass_maps_root / output_name

    if args.manifest_output:
        manifest_path = Path(args.manifest_output).expanduser().resolve()
    else:
        manifest_path = output_path.with_suffix("").with_name(output_path.stem + "_manifest.json")

    print(f"Mass maps root: {mass_maps_root}")
    print(f"Catalogue root: {catalog_root}")
    print(f"Group selection: {args.group} (cones {cones_for_group(args.group)[0]}..{cones_for_group(args.group)[-1]})")
    print(f"Bins: {bins}")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Device/dtype: {device} / {dtype}")
    print(
        f"Wavelet config: type={args.wavelet_type}, n_scales={args.n_scales}, "
        f"doth_base_radius={args.doth_base_radius}, l1_binning={args.l1_binning}"
    )

    if not mass_maps_root.exists():
        raise FileNotFoundError(f"Mass maps root not found: {mass_maps_root}")
    if not catalog_root.exists():
        raise FileNotFoundError(f"Catalogue root not found: {catalog_root}")

    candidates, issues, counters = discover_rows(
        mass_maps_root=mass_maps_root,
        catalog_root=catalog_root,
        bins=bins,
        group_choice=args.group,
    )
    print(
        "Discovered rows: "
        f"total={counters['rows_total']}, "
        f"candidate={counters['rows_candidate']}, "
        f"missing_maps={counters['rows_missing_maps']}, "
        f"missing_catalog={counters['rows_missing_catalog']}, "
        f"catalog_warnings={counters['rows_catalog_warning']}, "
        f"catalog_errors={counters['rows_catalog_error']}"
    )

    cosmo_labels = discover_cosmo_labels(mass_maps_root)
    nz_summary = None
    if not args.skip_nz_fingerprint:
        print("Computing n(z) fingerprints...")
        nz_summary = compute_nz_fingerprint(
            catalog_root=catalog_root,
            cosmo_labels=cosmo_labels,
            nz_max_files_per_group=args.nz_max_files_per_group,
            nz_sample_size=args.nz_sample_size,
            nz_min=args.nz_min,
            nz_max=args.nz_max,
            nz_nbins=args.nz_nbins,
        )
        for group in (GROUP_1_10, GROUP_11_25):
            profile = nz_summary["group_profiles"][group]
            print(f"  {group}: sampled_files={profile['count']}")

    if args.audit_only:
        audit_payload = {
            "created_at_utc": utc_timestamp(),
            "mode": "audit_only",
            "inputs": {
                "catalog_root": str(catalog_root),
                "mass_maps_root": str(mass_maps_root),
                "bins": bins,
                "group": args.group,
            },
            "counters": counters,
            "issue_count": len(issues),
            "issues": issues,
        }
        if nz_summary is not None:
            audit_payload["nz_summary"] = {
                "edges": nz_summary["edges"].tolist(),
                "group_profiles": {
                    group: {
                        "count": int(nz_summary["group_profiles"][group]["count"]),
                        "mean": nz_summary["group_profiles"][group]["mean"].tolist(),
                        "std": nz_summary["group_profiles"][group]["std"].tolist(),
                    }
                    for group in (GROUP_1_10, GROUP_11_25)
                },
                "used_file_counts": {
                    group: len(nz_summary["used_files"][group]) for group in (GROUP_1_10, GROUP_11_25)
                },
                "warnings": nz_summary["warnings"],
            }
        write_json(manifest_path, audit_payload)
        print(f"Audit manifest written to: {manifest_path}")
        return

    if args.max_rows is not None:
        candidates = candidates[: args.max_rows]
        print(f"Limiting processing to first {len(candidates)} candidate rows (--max-rows).")

    WLStatistics = configure_wl_stats_torch(args.wl_stats_path)
    stats = WLStatistics(
        n_scales=args.n_scales,
        device=device,
        pixel_arcmin=args.pixel_arcmin,
        dtype=dtype,
        wavelet_type=args.wavelet_type,
        doth_base_radius=args.doth_base_radius,
    )

    params_by_cosmo, param_names = load_cosmology_parameters(
        params_file=args.params_file,
        params_key=args.params_key,
        expected_ncosmo=25,
        param_names_override=args.param_names,
    )

    processed_rows: list[dict[str, Any]] = []
    l1_by_bin: dict[int, list[np.ndarray]] = {bin_id: [] for bin_id in bins}
    l1_centers_ref: dict[int, np.ndarray] = {}
    quality_records_by_bin: dict[int, list[dict[str, Any]]] = {bin_id: [] for bin_id in bins}
    status_counts = {
        "processed_ok": 0,
        "problematic_saved": 0,
        "skipped_quality": 0,
        "processing_errors": 0,
    }

    for idx, record in enumerate(tqdm(candidates, desc="Computing L1 datavectors")):
        row_l1: dict[int, np.ndarray] = {}
        row_centers: dict[int, np.ndarray] = {}
        quality_report: dict[int, dict[str, Any]] = {}

        try:
            for bin_id in bins:
                l1_values, l1_centers = compute_l1_from_map(
                    map_path=record.map_paths[bin_id],
                    stats=stats,
                    noise_sigma=args.noise_sigma,
                    peak_min_snr=args.peak_min_snr,
                    peak_max_snr=args.peak_max_snr,
                    peak_nbins=args.peak_nbins,
                    l1_min_snr=args.l1_min_snr,
                    l1_max_snr=args.l1_max_snr,
                    l1_nbins=args.l1_nbins,
                    clamp_overflow=args.clamp_overflow,
                    l1_binning=args.l1_binning,
                )
                row_l1[bin_id] = l1_values
                row_centers[bin_id] = l1_centers

                quality = evaluate_l1_datavector_quality(
                    l1_values,
                    negative_tolerance=args.negative_tolerance,
                    wiggle_fraction_threshold=args.wiggle_fraction_threshold,
                )
                quality_report[bin_id] = quality
                quality_records_by_bin[bin_id].append(quality)
        except Exception as exc:
            status_counts["processing_errors"] += 1
            issues.append(
                {
                    "row_key": record.row_key,
                    "status": "processing_error",
                    "reason": str(exc),
                    "cosmo_label": record.cosmo_label,
                    "cone": record.cone,
                }
            )
            continue

        problematic_bins = [
            bin_id for bin_id, quality in quality_report.items() if quality.get("problematic", False)
        ]
        if problematic_bins:
            reason_text = ", ".join(
                f"bin{bin_id}:{'/'.join(quality_report[bin_id].get('reasons', [])) or 'unknown'}"
                for bin_id in problematic_bins
            )
            if args.quality_policy == "fail":
                raise RuntimeError(
                    f"Quality policy 'fail' triggered for {record.row_key}. Reasons: {reason_text}"
                )
            if args.quality_policy == "skip":
                status_counts["skipped_quality"] += 1
                issues.append(
                    {
                        "row_key": record.row_key,
                        "status": "skipped_quality",
                        "reason": reason_text,
                        "cosmo_label": record.cosmo_label,
                        "cone": record.cone,
                    }
                )
                continue
            status_counts["problematic_saved"] += 1

        for bin_id in bins:
            if bin_id not in l1_centers_ref:
                l1_centers_ref[bin_id] = row_centers[bin_id]
            elif not np.allclose(l1_centers_ref[bin_id], row_centers[bin_id], rtol=0, atol=0):
                raise ValueError(
                    f"L1 bin centers mismatch for bin {bin_id} at row {record.row_key}. "
                    "Ensure consistent l1_min/l1_max/l1_nbins."
                )
            l1_by_bin[bin_id].append(row_l1[bin_id])

        processed_rows.append(
            {
                "row_key": record.row_key,
                "cosmo_index": record.cosmo_index,
                "cosmo_label": record.cosmo_label,
                "suffix": record.suffix,
                "cone": record.cone,
                "group_label": record.group_label,
                "catalog_rows": record.catalog_rows if record.catalog_rows is not None else -1,
                "map_paths": {bin_id: str(record.map_paths[bin_id]) for bin_id in bins},
                "quality": quality_report,
            }
        )
        status_counts["processed_ok"] += 1

        if args.clear_cuda_cache_every > 0 and device.type == "cuda":
            if (idx + 1) % args.clear_cuda_cache_every == 0:
                torch.cuda.empty_cache()

    if not processed_rows:
        raise RuntimeError(
            "No valid rows were processed. Check manifest issues for missing maps/errors."
        )

    row_cosmo_index = np.array([row["cosmo_index"] for row in processed_rows], dtype=np.int16)
    aligned_params = params_by_cosmo[row_cosmo_index]

    npz_payload: dict[str, Any] = {
        "created_at_utc": np.array([utc_timestamp()]),
        "catalog_root": np.array([str(catalog_root)]),
        "mass_maps_root": np.array([str(mass_maps_root)]),
        "wl_stats_path": np.array([str(Path(args.wl_stats_path).expanduser().resolve())]),
        "row_key": np.array([row["row_key"] for row in processed_rows], dtype=str),
        "row_cosmo_index": row_cosmo_index,
        "row_cosmo_label": np.array([row["cosmo_label"] for row in processed_rows], dtype=str),
        "row_suffix": np.array([row["suffix"] for row in processed_rows], dtype=str),
        "row_cone": np.array([row["cone"] for row in processed_rows], dtype=np.int16),
        "row_group": np.array([row["group_label"] for row in processed_rows], dtype=str),
        "row_catalog_rows": np.array([row["catalog_rows"] for row in processed_rows], dtype=np.int64),
        "selected_bins": np.array(bins, dtype=np.int16),
        "n_scales": np.array([args.n_scales], dtype=np.int16),
        "l1_nbins": np.array([args.l1_nbins], dtype=np.int16),
        "l1_min_snr": np.array([args.l1_min_snr], dtype=np.float64),
        "l1_max_snr": np.array([args.l1_max_snr], dtype=np.float64),
        "noise_sigma": np.array([args.noise_sigma], dtype=np.float64),
        "wavelet_type": np.array([args.wavelet_type], dtype=str),
        "doth_base_radius": np.array([args.doth_base_radius], dtype=np.float64),
        "l1_binning": np.array([args.l1_binning], dtype=str),
        "quality_policy": np.array([args.quality_policy], dtype=str),
        "params": aligned_params,
        "param_names": param_names,
    }
    for bin_id in bins:
        npz_payload[f"l1_bin{bin_id}"] = np.stack(l1_by_bin[bin_id], axis=0).astype(np.float32)
        npz_payload[f"l1_bin{bin_id}_centers"] = l1_centers_ref[bin_id].astype(np.float32)
        npz_payload[f"map_path_bin{bin_id}"] = np.array(
            [row["map_paths"][bin_id] for row in processed_rows], dtype=str
        )

    npz_payload["issue_row_key"] = np.array([issue["row_key"] for issue in issues], dtype=str)
    npz_payload["issue_status"] = np.array([issue["status"] for issue in issues], dtype=str)
    npz_payload["issue_reason"] = np.array([issue["reason"] for issue in issues], dtype=str)
    npz_payload["issue_cosmo_label"] = np.array(
        [issue.get("cosmo_label", "") for issue in issues], dtype=str
    )
    npz_payload["issue_cone"] = np.array([issue.get("cone", -1) for issue in issues], dtype=np.int16)

    for bin_id in bins:
        npz_payload[f"quality_bin{bin_id}_problematic"] = np.array(
            [
                bool(row["quality"][bin_id].get("problematic", False))
                for row in processed_rows
            ],
            dtype=np.bool_,
        )
        npz_payload[f"quality_bin{bin_id}_finite_ok"] = np.array(
            [bool(row["quality"][bin_id].get("finite_ok", False)) for row in processed_rows],
            dtype=np.bool_,
        )
        npz_payload[f"quality_bin{bin_id}_has_negative"] = np.array(
            [bool(row["quality"][bin_id].get("has_negative", False)) for row in processed_rows],
            dtype=np.bool_,
        )
        npz_payload[f"quality_bin{bin_id}_too_wiggly"] = np.array(
            [bool(row["quality"][bin_id].get("too_wiggly", False)) for row in processed_rows],
            dtype=np.bool_,
        )
        npz_payload[f"quality_bin{bin_id}_min_value"] = np.array(
            [float(row["quality"][bin_id].get("min_value", np.nan)) for row in processed_rows],
            dtype=np.float64,
        )
        npz_payload[f"quality_bin{bin_id}_max_wiggle_fraction"] = np.array(
            [
                float(row["quality"][bin_id].get("max_wiggle_fraction", np.nan))
                for row in processed_rows
            ],
            dtype=np.float64,
        )
        npz_payload[f"quality_bin{bin_id}_reasons"] = np.array(
            ["|".join(row["quality"][bin_id].get("reasons", [])) for row in processed_rows],
            dtype=str,
        )

    quality_summary_by_bin = {
        str(bin_id): summarize_quality_records(quality_records_by_bin[bin_id]) for bin_id in bins
    }
    rows_with_any_problematic = sum(
        1
        for row in processed_rows
        if any(
            row["quality"][bin_id].get("problematic", False)
            for bin_id in bins
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **npz_payload)

    manifest_payload: dict[str, Any] = {
        "created_at_utc": utc_timestamp(),
        "script": "l1_norm_processing_cosmoslics_starlet.py",
        "inputs": {
            "catalog_root": str(catalog_root),
            "mass_maps_root": str(mass_maps_root),
            "wl_stats_path": str(Path(args.wl_stats_path).expanduser().resolve()),
            "bins": bins,
            "group": args.group,
            "n_scales": args.n_scales,
            "pixel_arcmin": args.pixel_arcmin,
            "noise_sigma": args.noise_sigma,
            "wavelet_type": args.wavelet_type,
            "doth_base_radius": args.doth_base_radius,
            "l1_nbins": args.l1_nbins,
            "l1_min_snr": args.l1_min_snr,
            "l1_max_snr": args.l1_max_snr,
            "l1_binning": args.l1_binning,
            "quality_policy": args.quality_policy,
            "negative_tolerance": args.negative_tolerance,
            "wiggle_fraction_threshold": args.wiggle_fraction_threshold,
            "params_file": args.params_file,
            "params_key": args.params_key,
        },
        "counts": {
            **counters,
            **status_counts,
            "issues_total": len(issues),
            "rows_written": len(processed_rows),
        },
        "quality_summary": {
            "by_bin": quality_summary_by_bin,
            "rows_checked": len(processed_rows),
            "rows_with_any_problematic": rows_with_any_problematic,
        },
        "output_npz": str(output_path),
        "issue_preview": issues[:50],
    }
    if nz_summary is not None:
        manifest_payload["nz_summary"] = {
            "edges": nz_summary["edges"].tolist(),
            "group_profiles": {
                group: {
                    "count": int(nz_summary["group_profiles"][group]["count"]),
                    "mean": nz_summary["group_profiles"][group]["mean"].tolist(),
                    "std": nz_summary["group_profiles"][group]["std"].tolist(),
                }
                for group in (GROUP_1_10, GROUP_11_25)
            },
            "used_file_counts": {
                group: len(nz_summary["used_files"][group]) for group in (GROUP_1_10, GROUP_11_25)
            },
            "warnings": nz_summary["warnings"],
        }

    write_json(manifest_path, manifest_payload)

    print(f"Saved datavector bundle: {output_path}")
    print(f"Saved manifest: {manifest_path}")
    print(
        f"Rows written: {len(processed_rows)} | "
        f"Issues: {len(issues)} | "
        f"Quality problematic saved: {status_counts['problematic_saved']}"
    )


if __name__ == "__main__":
    main()
