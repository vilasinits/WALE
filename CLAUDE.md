# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

WALE (Wavelet ℓ₁-norm Estimator) is a Python package for predicting and analyzing one-point statistics of the wavelet ℓ₁-norm in cosmological weak lensing fields, based on Large Deviation Theory (LDT).

## Commands

### Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate wale_env

# Install package in editable mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_cosmology.py

# Run a single test function
pytest tests/test_cosmology.py::test_E_at_zero_is_one

# Run with coverage
pytest --cov=wale
```

### Documentation
```bash
cd docs && make html
```

### Code formatting
```bash
black src/
```

## Architecture

The package is in `src/wale/`. A typical prediction pipeline flows through these modules:

1. **`InitializeVariables.py`** — Entry point. `InitialiseVariables` class sets up cosmology, angular scales, lens-plane comoving distances, and lensing weights. Optionally draws P(k) covariance samples for variability studies.

2. **`CosmologyModel.py`** — `Cosmology_function` wraps PyCCL. Provides Hubble parameter, comoving distance, non-linear matter P(k) (HALOFIT), and lensing kernels. Supports CPL dark energy parameterization and both single-source and n(z)-based lensing weights.

3. **`VarianceCalculator.py`** — Computes linear and non-linear convergence variances σ² for a given filter, redshift slice, and scale. Handles both top-hat and starlet filters via Simpson's rule integration.

4. **`FilterFunctions.py`** — Defines filter window functions: top-hat (`top_hat_filter`), starlet (`starlet_filter`, B3 splines), and their analytical Hankel transforms. Used by the variance and PDF computations.

5. **`RateFunction.py`** — Implements the LDT action functions. `get_tau` maps density contrast to the LDT variable; `get_psi_2cell` computes the 2-cell action; `get_phi_projec_2cell` solves the projected saddle-point equations; `get_scaled_cgf` wraps the full SCGF computation.

6. **`CriticalPoints.py`** — `CriticalPointsFinder` locates critical points of the action function numerically using convolution stencils, marching squares (scikit-image), and root-finding. `find_critical_points_for_cosmo` is the high-level driver, with optional joblib parallelization.

7. **`ComputePDF.py`** — `computePDF` class assembles the prediction: takes variances and critical points, performs the Legendre transform and Bromwich integral (inverse Laplace transform) to produce the convergence field PDF.

8. **`CommonUtils.py`** — Shared utilities: pixel window functions, Fourier coordinate helpers, moment computation, L1 norm extraction from histograms, and Limber-approximation variance (`compute_sigma_kappa_squared`).

9. **`CovarianceMatrix.py`** — `get_covariance` estimates the P(k) covariance matrix using the i-trispectrum model (Gualdi et al. 2021) and optionally generates Monte Carlo P(k) realizations for propagating cosmological variability.

10. **`LoadSimulations.py`** — Loads simulation data, applies wavelet smoothing in Fourier space, computes PDFs and L1 norms across realizations, and returns ensemble averages/standard deviations for comparison with theory.

## Key Dependencies

- **PyCCL** — cosmological calculations (P(k), distances, growth factors)
- **CAMB / CLASS** — Boltzmann solvers for power spectra
- **JAX** — numerical computation (used in rate function and variance integrals)
- **numba** — JIT compilation for performance-critical loops
- **mpmath** — high-precision spherical Bessel function evaluations in `FilterFunctions`
- **scikit-image** — marching squares contour extraction in `CriticalPoints`
- **joblib** — optional parallelization in `CriticalPoints`

## Data

- `data/` — pre-computed results and simulation outputs (`.npz`, `.npy` files)
- `results/` — filter output arrays per tomographic bin
- `notebooks/` — exploratory Jupyter notebooks demonstrating pipeline usage
