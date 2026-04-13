"""Utilities for datavector naming, quality checks, and run metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def build_noise_suffix(add_noise: bool, noise_level: float) -> str:
    return f"_noisy_s{noise_level:.2f}" if add_noise else ""


def build_mask_suffix(apply_mask: bool, mask_area_sqdeg: float) -> str:
    if not apply_mask:
        return ""
    area_tag = int(round(mask_area_sqdeg))
    return f"_masked_{area_tag}sqdeg"


def build_top_hat_component_suffix(
    component: str,
    bin_number: int,
    theta: float,
    theta_ratio: float,
    dataset_suffix: str,
    add_noise: bool,
    noise_level: float,
    theta_idx: int | None = None,
) -> str:
    theta_str = (
        f"theta{theta:.1f}"
        if theta_idx is None
        else f"theta{theta:.1f}_scale{theta_idx}"
    )
    noise_suffix = build_noise_suffix(add_noise, noise_level)
    return f"_{component}_bin{bin_number}_{theta_str}_ratio{theta_ratio:.1f}{noise_suffix}_{dataset_suffix}.npy"


def build_top_hat_combined_filename(
    component: str,
    dataset_name: str,
    map_suffix: str,
    bin_number: int,
    scale_suffix: str,
    theta_ratio: float,
    add_noise: bool,
    noise_level: float,
) -> str:
    noise_suffix = build_noise_suffix(add_noise, noise_level)
    return (
        f"all_{component}_{dataset_name}_{map_suffix}_bin{bin_number}"
        f"{scale_suffix}_ratio{theta_ratio:.1f}{noise_suffix}.npy"
    )


def build_starlet_component_suffix(
    bin_number: int,
    dataset_suffix: str,
    add_noise: bool,
    noise_level: float,
    apply_mask: bool,
    mask_area_sqdeg: float,
) -> str:
    mask_suffix = build_mask_suffix(apply_mask, mask_area_sqdeg)
    noise_suffix = build_noise_suffix(add_noise, noise_level)
    return f"_l1_norms_starlet_bin{bin_number}{mask_suffix}{noise_suffix}_{dataset_suffix}.npy"


def build_starlet_combined_filename(
    dataset_name: str,
    map_suffix: str,
    bin_number: int,
    add_noise: bool,
    noise_level: float,
    apply_mask: bool,
    mask_area_sqdeg: float,
    min_snr: float,
    max_snr: float,
) -> str:
    mask_suffix = build_mask_suffix(apply_mask, mask_area_sqdeg)
    noise_suffix = build_noise_suffix(add_noise, noise_level)
    snr_suffix = f"_snr{min_snr:.0f}to{max_snr:.0f}"
    return (
        f"all_l1_norms_starlet_{dataset_name}_{map_suffix}_bin{bin_number}"
        f"{mask_suffix}{noise_suffix}{snr_suffix}.npy"
    )


def _wiggle_fraction(series: np.ndarray) -> float:
    if series.size < 5:
        return 0.0

    gradient = np.diff(series)
    amplitude = float(np.nanmax(np.abs(series))) if series.size else 0.0
    threshold = 1e-12 + 1e-6 * max(1.0, amplitude)
    significant = np.abs(gradient) > threshold
    gradient = gradient[significant]

    if gradient.size < 2:
        return 0.0

    signs = np.sign(gradient)
    sign_changes = np.count_nonzero(np.diff(signs) != 0)
    return float(sign_changes / (signs.size - 1))


def evaluate_l1_datavector_quality(
    datavector: np.ndarray,
    negative_tolerance: float = -1e-10,
    wiggle_fraction_threshold: float = 0.85,
) -> dict[str, Any]:
    arr = np.asarray(datavector, dtype=float)
    finite_ok = bool(np.isfinite(arr).all())
    min_value = float(np.nanmin(arr)) if arr.size else 0.0
    has_negative = min_value < negative_tolerance

    if arr.ndim <= 1:
        series_list = [arr]
    else:
        series_list = [row for row in arr.reshape(-1, arr.shape[-1])]

    wiggle_scores = [
        _wiggle_fraction(series) for series in series_list if series.size > 0
    ]
    max_wiggle_fraction = float(max(wiggle_scores)) if wiggle_scores else 0.0
    too_wiggly = max_wiggle_fraction > wiggle_fraction_threshold

    reasons: list[str] = []
    if not finite_ok:
        reasons.append("non_finite_values")
    if has_negative:
        reasons.append("negative_values")
    if too_wiggly:
        reasons.append("wiggly_shape")

    return {
        "problematic": bool(reasons),
        "reasons": reasons,
        "finite_ok": finite_ok,
        "min_value": min_value,
        "negative_tolerance": float(negative_tolerance),
        "has_negative": has_negative,
        "max_wiggle_fraction": max_wiggle_fraction,
        "wiggle_fraction_threshold": float(wiggle_fraction_threshold),
        "too_wiggly": too_wiggly,
    }


def summarize_quality_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    problematic = sum(1 for r in records if r.get("problematic"))
    non_finite = sum(1 for r in records if not r.get("finite_ok", True))
    negative = sum(1 for r in records if r.get("has_negative"))
    wiggly = sum(1 for r in records if r.get("too_wiggly"))
    return {
        "total_checked": total,
        "problematic": problematic,
        "non_finite": non_finite,
        "negative": negative,
        "wiggly": wiggly,
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
