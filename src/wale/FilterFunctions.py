import math as _math
import numpy as np
import jax
import jax.numpy as jnp
from functools import lru_cache
import mpmath as mp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Pure-JAX Bessel J0 and J1 via power series + asymptotic expansion.
#
# Replacing jax.pure_callback (which requires a host–device round-trip on
# every call) with a polynomial evaluation that runs entirely in XLA.
#
# Power series (accurate for |x| ≤ 12, 28 terms):
#   J0(x) = Σ  (-1)^m / (m!)^2   * (x/2)^(2m)
#   J1(x) = Σ  (-1)^m / (m!(m+1)!) * (x/2)^(2m+1)
#
# Asymptotic (|x| > 12, two-term):
#   J0(x) ≈ sqrt(2/(πx)) cos(x − π/4)
#   J1(x) ≈ sqrt(2/(πx)) [cos(x−3π/4)(1−15/(128x²)) − sin(x−3π/4) 3/(8x)]
# ---------------------------------------------------------------------------

_N_BESSEL = 28  # terms; accurate to ~1e-15 for |x| ≤ 12

_J0_COEFFS = jnp.array(
    [(-1.0) ** m / float(_math.factorial(m)) ** 2 for m in range(_N_BESSEL)],
    dtype=jnp.float64,
)
_J1_COEFFS = jnp.array(
    [(-1.0) ** m / (float(_math.factorial(m)) * float(_math.factorial(m + 1)))
     for m in range(_N_BESSEL)],
    dtype=jnp.float64,
)

# Same coefficients as plain numpy arrays (for the scalar numpy path)
_J0_COEFFS_NP = np.array(
    [(-1.0) ** m / float(_math.factorial(m)) ** 2 for m in range(_N_BESSEL)]
)
_J1_COEFFS_NP = np.array(
    [(-1.0) ** m / (float(_math.factorial(m)) * float(_math.factorial(m + 1)))
     for m in range(_N_BESSEL)]
)


def _j0(x):
    """
    J0(x) — pure JAX, no host callbacks.
    Power series for |x| ≤ 12; leading asymptotic for |x| > 12.
    JAX auto-differentiates through both branches.
    """
    absx = jnp.abs(x)
    t = absx * absx * 0.25  # (x/2)^2

    # Horner evaluation of Σ c_m * t^m (unrolled by JIT)
    p = _J0_COEFFS[-1]
    for c in _J0_COEFFS[-2::-1]:
        p = p * t + c

    safe = jnp.where(absx < 1e-30, jnp.ones_like(absx), absx)
    asym = jnp.sqrt(2.0 / (jnp.pi * safe)) * jnp.cos(safe - jnp.pi / 4.0)
    return jnp.where(absx < 1e-30, jnp.ones_like(x), jnp.where(absx <= 12.0, p, asym))


def _j1(x):
    """
    J1(x) — pure JAX, no host callbacks.
    Power series for |x| ≤ 12; two-term asymptotic for |x| > 12.
    """
    absx = jnp.abs(x)
    t = absx * absx * 0.25

    p = _J1_COEFFS[-1]
    for c in _J1_COEFFS[-2::-1]:
        p = p * t + c
    poly_val = (absx * 0.5) * p

    safe = jnp.where(absx < 1e-30, jnp.ones_like(absx), absx)
    asym = jnp.sqrt(2.0 / (jnp.pi * safe)) * (
        jnp.cos(safe - 0.75 * jnp.pi) * (1.0 - 15.0 / (128.0 * safe ** 2))
        - jnp.sin(safe - 0.75 * jnp.pi) * (3.0 / (8.0 * safe))
    )
    result = jnp.where(absx <= 12.0, poly_val, asym)
    return jnp.where(absx < 1e-30, jnp.zeros_like(x), result * jnp.sign(x))


# ---------------------------------------------------------------------------
# Numpy scalar versions (used in VarianceCalculator's fast scalar path)
# ---------------------------------------------------------------------------

def _j1_numpy(x):
    """Numpy J1 via the same polynomial — no scipy dependency."""
    absx = np.abs(x)
    t = absx * absx * 0.25
    p = np.polyval(_J1_COEFFS_NP[::-1], t)   # uses numpy.polyval
    # Actually, let's use Horner manually for consistency:
    p2 = _J1_COEFFS_NP[-1]
    for c in _J1_COEFFS_NP[-2::-1]:
        p2 = p2 * t + c
    poly_val = (absx * 0.5) * p2
    # asymptotic for large x
    safe = np.where(absx < 1e-30, 1.0, absx)
    asym = np.sqrt(2.0 / (np.pi * safe)) * (
        np.cos(safe - 0.75 * np.pi) * (1.0 - 15.0 / (128.0 * safe ** 2))
        - np.sin(safe - 0.75 * np.pi) * (3.0 / (8.0 * safe))
    )
    result = np.where(absx <= 12.0, poly_val, asym)
    return np.where(absx < 1e-30, 0.0, result * np.sign(x))


# ---------------------------------------------------------------------------
# Filter window functions (JAX)
# ---------------------------------------------------------------------------

def top_hat_filter(k, R):
    """Top-hat filter W(kR) = 2 J1(kR) / (kR). Safe at kR → 0."""
    kR = k * R
    safe_kR = jnp.where(jnp.abs(kR) < 1e-30, jnp.ones_like(kR), kR)
    return jnp.where(jnp.abs(kR) < 1e-30, jnp.ones_like(kR), 2.0 * _j1(safe_kR) / safe_kR)


def top_hat_filter_numpy(k, R):
    """Numpy version of top-hat filter — for fast scalar path."""
    kR = k * R
    safe_kR = np.where(np.abs(kR) < 1e-30, np.ones_like(kR), kR)
    return np.where(np.abs(kR) < 1e-30, np.ones_like(kR), 2.0 * _j1_numpy(safe_kR) / safe_kR)


def b3_1D_ft(x):
    """B3-spline 1D Fourier factor sin(x/2)/(x/2). Safe at x=0 (limit = 1)."""
    safe_x = jnp.where(jnp.abs(x) < 1e-30, jnp.ones_like(x), x)
    sinc_val = jnp.where(
        jnp.abs(x) < 1e-30,
        jnp.ones_like(x),
        jnp.sin(safe_x / 2.0) / (safe_x / 2.0),
    )
    return sinc_val ** 4.0


def b3_2D_ft(x, y):
    return b3_1D_ft(x) * b3_1D_ft(y)


def starlet_filter(k, R):
    """Isotropic starlet (B3-spline) filter in Fourier space."""
    return b3_2D_ft(k * R, k * R)


def starlet_filter_numpy(k, R):
    """Numpy version of starlet filter — for fast scalar path."""
    kR = k * R
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(np.abs(kR / 2) < 1e-30, 1.0, np.sin(kR / 2) / (kR / 2))
    return (s ** 4) ** 2  # b3_1D_ft(kR)^2


def get_W2D_FL(window_radius, map_shape, filter_type, **kwargs):
    """2D Fourier-space window function for a square map."""
    N = map_shape[0]
    dx = N / N
    kx = np.fft.fftshift(np.fft.fftfreq(N, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(N, dx))
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k_grid = jnp.asarray(2.0 * np.pi * np.sqrt(k2))
    ind = int(N / 2)
    k_grid = k_grid.at[ind, ind].set(1e-7)
    if filter_type == "tophat":
        return top_hat_filter(k_grid, window_radius)
    elif filter_type == "starlet":
        return starlet_filter(k_grid, window_radius)


# ---------------------------------------------------------------------------
# Analytical Hankel transform utilities (mpmath/numpy — NOT in hot path).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def S_scalar(n: int, b: float) -> float:
    from scipy import special as sp
    if n < -1:
        raise ValueError("n cannot be smaller than -1.")
    J0 = sp.j0(b)
    J1 = sp.j1(b)
    if n == 0:
        return b * J1
    elif n == -1:
        return b * float(mp.hyp1f2(0.5, 1, 1.5, -(b ** 2) / 4))
    else:
        return b ** (n + 1) * J1 + n * b ** n * J0 - n ** 2 * S_scalar(n - 2, b)


def S(n: int, b):
    b = np.asarray(b)
    if b.ndim == 0:
        return S_scalar(n, float(b))
    return np.vectorize(lambda x: S_scalar(n, float(x)))(b)


def uHat_starlet_analytical(eta, R):
    """Analytical Hankel transform of the starlet U-filter (mpmath, not JAX)."""
    eta = np.asarray(eta) * R
    eta_safe = np.clip(eta, 2e-2, 100)
    b_half = 0.5 * eta_safe
    b_one = eta_safe
    b_two = 2.0 * eta_safe

    factor1 = (
        0.125 * eta_safe ** 3 * S(0, b_half)
        - 0.75 * eta_safe ** 2 * S(1, b_half)
        + 1.5 * eta_safe * S(2, b_half)
        - S(3, b_half)
    )
    factor2 = (
        eta_safe ** 3 * S(0, b_one)
        - 3 * eta_safe ** 2 * S(1, b_one)
        + 3 * eta_safe * S(2, b_one)
        - S(3, b_one)
    )
    factor3 = (
        8 * eta_safe ** 3 * S(0, b_two)
        - 12 * eta_safe ** 2 * S(1, b_two)
        + 6 * eta_safe * S(2, b_two)
        - S(3, b_two)
    )
    return (
        (2 * np.pi)
        * (-128 / 9 * factor1 + 4 * factor2 - 1 / 9 * factor3)
        / eta_safe ** 5
    )
