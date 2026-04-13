import numpy as np

from wale.DatavectorDiagnostics import (
    build_starlet_combined_filename,
    build_starlet_component_suffix,
    build_top_hat_combined_filename,
    build_top_hat_component_suffix,
    evaluate_l1_datavector_quality,
)


def test_top_hat_component_suffix_matches_expected_contract():
    suffix = build_top_hat_component_suffix(
        component="l1_norm",
        bin_number=4,
        theta=30.0,
        theta_ratio=2.0,
        dataset_suffix="halofit",
        add_noise=True,
        noise_level=0.26,
        theta_idx=None,
    )
    assert suffix == "_l1_norm_bin4_theta30.0_ratio2.0_noisy_s0.26_halofit.npy"


def test_top_hat_combined_filename_matches_expected_contract():
    name = build_top_hat_combined_filename(
        component="l1_norms",
        dataset_name="halofit",
        map_suffix="baryonified",
        bin_number=2,
        scale_suffix="_theta15.0",
        theta_ratio=2.0,
        add_noise=False,
        noise_level=0.26,
    )
    assert name == "all_l1_norms_halofit_baryonified_bin2_theta15.0_ratio2.0.npy"


def test_starlet_filename_contracts():
    component_suffix = build_starlet_component_suffix(
        bin_number=1,
        dataset_suffix="fiducial",
        add_noise=True,
        noise_level=0.26,
        apply_mask=True,
        mask_area_sqdeg=14000.0,
    )
    assert (
        component_suffix
        == "_l1_norms_starlet_bin1_masked_14000sqdeg_noisy_s0.26_fiducial.npy"
    )

    combined_name = build_starlet_combined_filename(
        dataset_name="halofit",
        map_suffix="baryonified",
        bin_number=3,
        add_noise=False,
        noise_level=0.26,
        apply_mask=False,
        mask_area_sqdeg=14000.0,
        min_snr=-13.0,
        max_snr=13.0,
    )
    assert (
        combined_name == "all_l1_norms_starlet_halofit_baryonified_bin3_snr-13to13.npy"
    )


def test_quality_detects_negative_values():
    vec = np.array([0.1, 0.2, -0.05, 0.3])
    quality = evaluate_l1_datavector_quality(
        vec, negative_tolerance=-1e-10, wiggle_fraction_threshold=0.95
    )
    assert quality["problematic"] is True
    assert quality["has_negative"] is True
    assert "negative_values" in quality["reasons"]


def test_quality_detects_wiggly_shape():
    vec = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    quality = evaluate_l1_datavector_quality(
        vec, negative_tolerance=-1e-10, wiggle_fraction_threshold=0.5
    )
    assert quality["problematic"] is True
    assert quality["too_wiggly"] is True
    assert "wiggly_shape" in quality["reasons"]


def test_quality_flags_non_finite():
    vec = np.array([0.1, np.nan, 0.3])
    quality = evaluate_l1_datavector_quality(vec)
    assert quality["problematic"] is True
    assert quality["finite_ok"] is False
    assert "non_finite_values" in quality["reasons"]
