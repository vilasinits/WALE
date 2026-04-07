import functools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .FilterFunctions import top_hat_filter, starlet_filter, top_hat_filter_numpy, starlet_filter_numpy


# ---------------------------------------------------------------------------
# Simpson weights pre-computation (numpy, once per k-grid)
# ---------------------------------------------------------------------------

def _simpson_weights_numpy(x):
    """
    Non-uniform composite Simpson weights for grid x (numpy, not JAX).
    Integral ≈ sum(w * f).  Called once at Variance construction.
    """
    n = len(x)
    h = np.diff(x)
    w = np.zeros(n)
    n_pairs = (n - 1) // 2
    for i in range(n_pairs):
        h0, h1 = h[2 * i], h[2 * i + 1]
        c = (h0 + h1) / 6.0
        w[2 * i]     += c * (2.0 - h1 / h0)
        w[2 * i + 1] += c * (h0 + h1) ** 2 / (h0 * h1)
        w[2 * i + 2] += c * (2.0 - h0 / h1)
    if n % 2 == 0:          # leftover trapezoidal interval
        w[-2] += 0.5 * h[-1]
        w[-1] += 0.5 * h[-1]
    return w


# ---------------------------------------------------------------------------
# _simpson_jax: kept for ComputePDF (still used there)
# ---------------------------------------------------------------------------

def _simpson_jax(y, x):
    """
    Non-uniform composite Simpson's rule, integrating over the last axis.
    Works for 1D (n,), 2D (m, n) and 3D (m, p, n) inputs.
    """
    n = y.shape[-1]
    h = jnp.diff(x)

    if n < 2:
        return jnp.zeros(y.shape[:-1])
    if n == 2:
        return 0.5 * h[0] * (y[..., 0] + y[..., 1])

    n_pairs = (n - 1) // 2
    h0 = h[0::2][:n_pairs]
    h1 = h[1::2][:n_pairs]
    y0 = y[..., 0 : 2 * n_pairs : 2]
    y1 = y[..., 1 : 2 * n_pairs + 1 : 2]
    y2 = y[..., 2 : 2 * n_pairs + 2 : 2]

    result = jnp.sum(
        (h0 + h1) / 6.0
        * (
            y0 * (2.0 - h1 / h0)
            + y1 * (h0 + h1) ** 2 / (h0 * h1)
            + y2 * (2.0 - h0 / h1)
        ),
        axis=-1,
    )
    if n % 2 == 0:
        result = result + 0.5 * h[-1] * (y[..., -2] + y[..., -1])
    return result


# ---------------------------------------------------------------------------
# JIT-compiled σ² — uses pre-computed base weights, just a dot product
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2,))
def _sigma2_jit(k, bw, filter_type, R1, R2):
    """
    σ²(R₁, R₂) = sum(W(kR₁) · W(kR₂) · bw)

    bw = k · P(k) · simpson_weights / (2π), pre-computed per (z, pk).
    Compiled once per filter_type; all subsequent calls are a single XLA op.
    """
    if filter_type == "tophat":
        w1 = top_hat_filter(k, R1)
        w2 = top_hat_filter(k, R2)
    else:
        w1 = starlet_filter(k, R1)
        w2 = starlet_filter(k, R2)
    return jnp.sum(w1 * w2 * bw)


# ---------------------------------------------------------------------------
# Variance class
# ---------------------------------------------------------------------------

class Variance:
    """
    Compute linear and nonlinear convergence variances.

    σ²(R₁, R₂, z) = (1/2π) ∫ k P(k,z) W(kR₁) W(kR₂) dk

    Pre-computes Simpson weights and base-weight arrays (k·P(k)·w_simp/(2π))
    at construction so that each nonlinear_sigma2 call reduces to a single
    JIT-compiled dot product.
    """

    def __init__(self, cosmo, filter_type, pk):
        self.cosmo = cosmo
        self.filter_type = filter_type
        self.pk = pk

        k = cosmo.k                                        # (nk,) numpy
        self._k_jax = jnp.asarray(k, dtype=jnp.float64)

        # Simpson integration weights for this k-grid (numpy, once)
        simp_w = _simpson_weights_numpy(k)                 # (nk,) numpy
        self._simp_w = simp_w

        # Base weights per redshift: bw[z] = k · pk[z] · simp_w / (2π)
        self._bw_numpy = {
            z: k * p * simp_w / (2.0 * np.pi)
            for z, p in pk.items()
        }
        self._bw_jax = {
            z: jnp.asarray(bw, dtype=jnp.float64)
            for z, bw in self._bw_numpy.items()
        }

        # Pre-warm JIT so compilation happens at construction, not first use
        _z0 = next(iter(self._bw_jax))
        _sigma2_jit(
            self._k_jax, self._bw_jax[_z0], filter_type,
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )

    def nonlinear_sigma2(self, redshift, R1, R2=None, **kwargs):
        """σ²(R₁, R₂, z) via JIT-compiled dot product."""
        if R2 is None:
            R2 = R1
        if "pk" in kwargs:
            bw = jnp.asarray(
                self.cosmo.k * kwargs["pk"] * self._simp_w / (2.0 * np.pi),
                dtype=jnp.float64,
            )
        else:
            bw = self._bw_jax[redshift]
        return _sigma2_jit(self._k_jax, bw, self.filter_type, R1, R2)

    def get_sig_slice(self, z, R1, R2):
        """σ²(R₁) + σ²(R₂) − 2σ²(R₁, R₂)  [2-cell / annulus variance]"""
        return (
            self.nonlinear_sigma2(z, R1)
            + self.nonlinear_sigma2(z, R2)
            - 2.0 * self.nonlinear_sigma2(z, R1, R2)
        )

    def get_sig_slice_1cell(self, z, R1):
        """σ²(R₁, R₁)  [single-cell variance at radius R₁]"""
        return self.nonlinear_sigma2(z, R1)
