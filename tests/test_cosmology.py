import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath('../src'))
from wale.CosmologyModel import *



# pick some fiducial cosmology
COSMO = Cosmology_function(h=0.7, Oc=0.25, Ob=0.05, sigma8=0.8)

# 1) Expansion and Hubble rate
def test_E_at_zero_is_one():
    assert np.isclose(COSMO._E(0.0), 1.0)

def test_H_at_zero_equals_H0():
    assert np.isclose(COSMO.get_H(0.0), COSMO.H0)

# 2) Comoving distance χ(z)
def test_chi_zero_is_zero():
    assert np.isclose(COSMO.get_chi(0.0), 0.0)

def test_chi_monotonic_increasing():
    zs = np.linspace(0, 2, 5)
    chis = COSMO.get_chi(zs)
    assert np.all(np.diff(chis) > 0)

# 3) Inversion consistency: z(χ(z)) ≃ z
@pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 2.0])
def test_inversion_roundtrip(z):
    chi = COSMO.get_chi(z)
    z_recov = COSMO.get_z_from_chi(chi)
    assert np.isclose(z, z_recov, atol=1e-6)

# 4) Non-linear P(k): shape and positivity
def test_nonlinear_pk_default_shape_and_positive():
    # using default k-grid
    z = 0.5
    P = COSMO.get_nonlinear_pk(z)
    # should return exactly one value per entry in COSMO.k
    assert P.shape == (COSMO.nk,)
    assert np.all(P > 0)

# 5) Lensing weight (array version)
def test_lensing_weight_array_shape_and_edge_behavior():
    # compute chi_s
    chi_s = COSMO.get_chi(1.0)
    chis = np.array([0.0, chi_s, chi_s * 1.1])
    z_vals, W = COSMO.get_lensing_weight_array(chis, chi_s)

    # shape must match
    assert W.shape == chis.shape
    # at chi=0, weight is zero
    assert np.isclose(W[0], 0.0)
    # at chi=chi_source, factor (1 - chis/chi_source) == 0 → W == 0
    assert np.isclose(W[1], 0.0)
    # beyond the source plane, weight flips sign (implementation uses negative)
    assert W[2] < 0

# 6) nz-based lensing weight: positivity & shape
def test_nz_based_lensing_weight_positive_and_shape():
    # toy n(z)
    z_nz = np.linspace(0, 1, 1001)
    n_z = np.ones_like(z_nz)
    chis = np.linspace(0, COSMO.get_chi(1.0), 5)

    z_arr, W_nz = COSMO.get_lensing_weight_array_nz(chis, z_nz, n_z)
    # shape match
    assert W_nz.shape == chis.shape
    # weights non-negative for chis <= chi_source
    assert np.all(W_nz >= 0)

# 7) Edge‐cases & errors
def test_inversion_out_of_bounds_raises():
    chi_too_big = COSMO.get_chi(10.0) * 1.1
    with pytest.raises(ValueError):
        _ = COSMO.get_z_from_chi(chi_too_big, z_max=5.0)

def test_missing_normalization_parameters():
    with pytest.raises(ValueError):
        Cosmology_function(h=0.7, Oc=0.25, Ob=0.05)