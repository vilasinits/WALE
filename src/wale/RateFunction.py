import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .FilterFunctions import top_hat_filter, starlet_filter


# ---------------------------------------------------------------------------
# LDT density → τ mapping   (Bernardeau 1994, Boyle+2021 Eq. 1)
#
# τ(ρ) = ν(1 − ρ^{−1/ν}),   ν = 1.4  (spherical-collapse approximation)
# ---------------------------------------------------------------------------

def get_tau(rho):
    """
    LDT variable τ(ρ = 1+δ).

        τ(ρ) = ν ( 1 − ρ^{−1/ν} ),   ν = 1.4

    Parameters
    ----------
    rho : float or array_like
        Local density ρ = 1 + δ.
    """
    nu = 1.4
    return nu * (1.0 - rho ** (-1.0 / nu))


# ---------------------------------------------------------------------------
# Pointwise rate-function evaluators
# Pure JAX (no @jit) — JIT is applied by the projection functions via vmap.
# ---------------------------------------------------------------------------

def _psi_1cell(k, bw_l, chi, delta, theta1, filter_type):
    """
    1-cell linear rate function at a single (χ, δ) point.  [Boyle+2021 Eq. 4]

        ψ_l(δ) = τ(1+δ)² / (2 σ²_l(R_lag))

    R_lag = χ √max(1+δ, 0.01) θ₁  (mass-conservation Lagrangian radius)
    σ²_l  = Σ_k W(k R_lag)² · bw_l   with  bw_l = k P_lin w_simp / (2π)

    Parameters
    ----------
    k     : (nk,) wavenumber array [Mpc⁻¹]
    bw_l  : (nk,) k · P_lin(k,z) · w_simp / (2π)  for this z-slice
    chi   : scalar  comoving distance [Mpc]
    delta : scalar  density contrast δ
    theta1: scalar  angular scale θ₁ [rad]
    filter_type : 'tophat' or 'starlet'
    """
    R    = chi * jnp.sqrt(jnp.maximum(1.0 + delta, 0.01)) * theta1
    tau  = get_tau(1.0 + delta)
    W    = top_hat_filter(k, R) if filter_type == "tophat" else starlet_filter(k, R)
    sig2 = jnp.sum(W ** 2 * bw_l)
    return tau ** 2 / (2.0 * sig2)


def _psi_2cell(k, bw, chi, d1, d2, theta1, theta2, recal, filter_type):
    """
    2-cell rate function at a single (χ, δ₁, δ₂) point.  [Boyle+2021 Eq. 6]

        ψ(δ₁, δ₂) = (τᵀ Σ⁻¹ τ) · recal / 2

    where  τ = (τ(1+δ₁), τ(1+δ₂))  and  Σ = [[σ₁₁, σ₁₂], [σ₁₂, σ₂₂]]
    is the 2-cell convergence covariance at the Lagrangian radii.
    Expanding the 2×2 inverse analytically:
        τᵀ Σ⁻¹ τ = (σ₁₁ τ₂² − 2 σ₁₂ τ₁ τ₂ + σ₂₂ τ₁²) / det(Σ)

    Parameters
    ----------
    k, bw  : (nk,) wavenumber and nonlinear base-weight arrays
    chi    : scalar  comoving distance [Mpc]
    d1, d2 : scalars  density contrasts δ₁, δ₂
    theta1, theta2 : angular scales [rad]
    recal  : empirical recalibration factor  σ²_LDT / σ²_sim
    filter_type : 'tophat' or 'starlet'
    """
    R1   = chi * jnp.sqrt(jnp.maximum(1.0 + d1, 0.01)) * theta1
    R2   = chi * jnp.sqrt(jnp.maximum(1.0 + d2, 0.01)) * theta2
    tau1 = get_tau(1.0 + d1)
    tau2 = get_tau(1.0 + d2)
    if filter_type == "tophat":
        W1, W2 = top_hat_filter(k, R1), top_hat_filter(k, R2)
    else:
        W1, W2 = starlet_filter(k, R1), starlet_filter(k, R2)
    sig11 = jnp.sum(W1 ** 2 * bw)
    sig22 = jnp.sum(W2 ** 2 * bw)
    sig12 = jnp.sum(W1 * W2 * bw)
    det   = sig11 * sig22 - sig12 ** 2
    return (sig11 * tau2 ** 2 - 2.0 * sig12 * tau1 * tau2 + sig22 * tau1 ** 2) * recal / (det * 2.0)


# ---------------------------------------------------------------------------
# JAX-native Newton solvers
#
# Both use jax.lax.scan over λ (warm-starting: δ*(λ_j) seeds δ*(λ_{j+1}))
# and jax.lax.fori_loop for the inner Newton iterations (fixed iteration count,
# required for XLA compilation).  Callers vmap these over the χ dimension.
# ---------------------------------------------------------------------------

def _newton_1cell_scan(k, bw_l, chi, r, w, theta1, filter_type, lam_arr, n_iter):
    """
    Warm-started 1D Newton: δ*(λ) for all λ at one χ-slice.

    Saddle condition  [Boyle+2021 Eq. NLPhi assembly]:
        ψ'_l(δ*) = λ · W(χ) · r,   r = σ²_nl(R₀) / σ²_l(R₀)

    f(δ) = ψ'_l(δ) − λ W r = 0  solved by Newton:
        δ ← clip(δ − f(δ)/f'(δ),  −0.99, 4.5)

    Both f and f' are approximated by central finite differences on ψ_l.

    Parameters
    ----------
    k, bw_l   : (nk,) wavenumber and linear base-weight arrays
    chi       : scalar  comoving distance [Mpc]
    r         : scalar  σ²_nl(R₀) / σ²_l(R₀)
    w         : scalar  lensing weight W(χ)
    theta1    : scalar  angular scale θ₁ [rad]
    filter_type : 'tophat' or 'starlet'  (Python str, static at trace time)
    lam_arr   : (nlambda,) λ values (must be ascending for good warm-starting)
    n_iter    : int  Newton steps per λ (Python int, static)

    Returns
    -------
    delta_zeros : (nlambda,) δ*(λ)
    """
    deld = 1e-6

    def psi(d):
        return _psi_1cell(k, bw_l, chi, d, theta1, filter_type)

    def solve_one_lam(d_prev, lam_j):
        target = lam_j * w * r   # ψ'_l(δ*) = λ W r

        def newton_step(d, _):
            dpsi  = (psi(d + deld) - psi(d - deld)) / (2.0 * deld)
            d2psi = (psi(d + deld) - 2.0 * psi(d) + psi(d - deld)) / deld ** 2
            return jnp.clip(d - (dpsi - target) / (d2psi + 1e-30), -0.99, 4.5), None

        d_star, _ = jax.lax.scan(newton_step, d_prev, None, length=n_iter)
        return d_star, d_star

    _, delta_zeros = jax.lax.scan(solve_one_lam, jnp.float64(0.0), lam_arr)
    return delta_zeros   # (nlambda,)


def _newton_2cell_scan(k, bw, chi, w, theta1, theta2, recal, filter_type, lam_arr, n_iter):
    """
    Warm-started 2D Newton: (δ₁*, δ₂*)(λ) for all λ at one χ-slice.

    Saddle conditions  [Boyle+2021 Eq. 7]:
        ∂ψ/∂δ₁ = −λ W,   ∂ψ/∂δ₂ = +λ W

    Residual (absorbing the FD step deld into A so units match):
        F₁ = 2A + [ψ(δ₁+deld,δ₂) − ψ(δ₁−deld,δ₂)] = 0,  A = λ W deld
        F₂ = −2A + [ψ(δ₁,δ₂+deld) − ψ(δ₁,δ₂−deld)] = 0

    Jacobian via central finite differences; 2×2 system solved with
    jnp.linalg.solve (small diagonal regulariser avoids exact-singularity NaN).

    Parameters
    ----------
    k, bw  : (nk,) wavenumber and nonlinear base-weight arrays
    chi    : scalar  comoving distance [Mpc]
    w      : scalar  lensing weight W(χ)
    theta1, theta2 : angular scales [rad]
    recal  : empirical recalibration factor
    filter_type : 'tophat' or 'starlet'
    lam_arr : (nlambda,) λ values
    n_iter  : int  Newton steps per λ

    Returns
    -------
    delta_zeros : (nlambda, 2)  — (δ₁*, δ₂*) for each λ
    """
    deld  = 1e-6
    h_jac = 1e-6   # Jacobian finite-difference step

    def psi(d1, d2):
        return _psi_2cell(k, bw, chi, d1, d2, theta1, theta2, recal, filter_type)

    def residual(state, A):
        d1, d2 = state[0], state[1]
        F1 = 2.0 * A + (psi(d1 + deld, d2) - psi(d1 - deld, d2))
        F2 = -2.0 * A + (psi(d1, d2 + deld) - psi(d1, d2 - deld))
        return jnp.array([F1, F2])

    def newton_step_2d(state, A):
        fval = residual(state, A)
        e0   = jnp.array([h_jac, 0.0])
        e1   = jnp.array([0.0, h_jac])
        J    = jnp.stack([
            (residual(state + e0, A) - residual(state - e0, A)) / (2.0 * h_jac),
            (residual(state + e1, A) - residual(state - e1, A)) / (2.0 * h_jac),
        ], axis=-1)
        # Small diagonal regulariser prevents NaN when J is near-singular
        J_reg = J + 1e-15 * jnp.eye(2, dtype=jnp.float64)
        return state + jnp.linalg.solve(J_reg, -fval)

    def solve_one_lam(state_prev, lam_j):
        A         = lam_j * w * deld
        new_state = jax.lax.fori_loop(
            0, n_iter, lambda _, s: newton_step_2d(s, A), state_prev
        )
        return new_state, new_state

    _, delta_zeros = jax.lax.scan(
        solve_one_lam, jnp.zeros(2, dtype=jnp.float64), lam_arr
    )
    return delta_zeros   # (nlambda, 2)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers used by CriticalPoints.py
# ---------------------------------------------------------------------------

def get_psi_2cell(variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """
    2-cell LDT action ψ(δ₁, δ₂).  [Boyle+2021 Eq. 6]

    Thin wrapper around _psi_2cell for CriticalPoints.py, which passes
    a Variance object rather than raw k/bw arrays.
    """
    k   = variance._k_jax
    bw  = variance._bw_jax[z]
    return _psi_2cell(
        k, bw,
        jnp.asarray(chi,    dtype=jnp.float64).squeeze(),
        jnp.asarray(delta1, dtype=jnp.float64).reshape(()),
        jnp.asarray(delta2, dtype=jnp.float64).reshape(()),
        theta1, theta2, recal, variance.filter_type,
    )


def get_psi_derivative_delta1(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂ψ/∂δ₁ via central finite differences."""
    return (get_psi_2cell(variance, chi, recal, z, delta1 + deld, delta2, theta1, theta2)
            - get_psi_2cell(variance, chi, recal, z, delta1 - deld, delta2, theta1, theta2)) / (2.0 * deld)


def get_psi_derivative_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂ψ/∂δ₂ via central finite differences."""
    return (get_psi_2cell(variance, chi, recal, z, delta1, delta2 + deld, theta1, theta2)
            - get_psi_2cell(variance, chi, recal, z, delta1, delta2 - deld, theta1, theta2)) / (2.0 * deld)


def get_psi_2nd_derivative_delta1(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂²ψ/∂δ₁²."""
    psi_p = get_psi_2cell(variance, chi, recal, z, delta1 + deld, delta2, theta1, theta2)
    psi_m = get_psi_2cell(variance, chi, recal, z, delta1 - deld, delta2, theta1, theta2)
    psi_0 = get_psi_2cell(variance, chi, recal, z, delta1,        delta2, theta1, theta2)
    return (psi_p - 2.0 * psi_0 + psi_m) / deld ** 2


def get_psi_2nd_derivative_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂²ψ/∂δ₂²."""
    psi_p = get_psi_2cell(variance, chi, recal, z, delta1, delta2 + deld, theta1, theta2)
    psi_m = get_psi_2cell(variance, chi, recal, z, delta1, delta2 - deld, theta1, theta2)
    psi_0 = get_psi_2cell(variance, chi, recal, z, delta1, delta2,        theta1, theta2)
    return (psi_p - 2.0 * psi_0 + psi_m) / deld ** 2


def get_psi_mixed_derivative_delta1_delta2(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    """∂²ψ/∂δ₁∂δ₂ via 4-point central stencil."""
    h  = deld
    pp  = get_psi_2cell(variance, chi, recal, z, delta1 + h, delta2 + h, theta1, theta2)
    pm  = get_psi_2cell(variance, chi, recal, z, delta1 + h, delta2 - h, theta1, theta2)
    mp_ = get_psi_2cell(variance, chi, recal, z, delta1 - h, delta2 + h, theta1, theta2)
    mm  = get_psi_2cell(variance, chi, recal, z, delta1 - h, delta2 - h, theta1, theta2)
    return (pp - pm - mp_ + mm) / (4.0 * h ** 2)


def psi_derivative_determinant(deld, delta1, delta2, z, variance, chi, recal, theta1, theta2):
    """det(Hψ) = ∂²ψ/∂δ₁² · ∂²ψ/∂δ₂² − (∂²ψ/∂δ₁∂δ₂)²."""
    p11 = get_psi_2nd_derivative_delta1(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2)
    p22 = get_psi_2nd_derivative_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2)
    p12 = get_psi_mixed_derivative_delta1_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2)
    return p11 * p22 - p12 ** 2


# ---------------------------------------------------------------------------
# Projected 2-cell SCGF  φ(λ)  [Boyle+2021 Eq. 8]
# ---------------------------------------------------------------------------

def get_phi_projec_2cell(
    theta1, theta2, zarr, chis, dchis, w, y, recal, variance, **kwargs
):
    """
    Projected 2-cell scaled cumulant generating function:

        φ(λ) = ∫ dχ  max_{δ₁,δ₂} { λ W(χ)(−δ₁+δ₂) − ψ(δ₁,δ₂;χ) }

    Assembled slice by slice:
        φ(λ) = dχ Σ_i [ λ W_i (−δ₁* + δ₂*) − ψ(δ₁*, δ₂*; χ_i) ]

    Parameters
    ----------
    theta1, theta2 : float  angular scales [rad]
    zarr  : (nchi,) redshifts
    chis  : (nchi,) comoving distances [Mpc]
    dchis : float   uniform Δχ step
    w     : (nchi,) lensing weights W(χ)
    y     : (nlambda,) λ-grid
    recal : float   σ²_LDT / σ²_sim recalibration factor
    variance : Variance  built from nonlinear P(k)
    n_iter : int, optional  Newton steps per λ (default 30)
    """
    n_iter = int(kwargs.get("n_iter", 30))

    k           = variance._k_jax
    filter_type = variance.filter_type
    bw_stack    = jnp.stack([variance._bw_jax[z] for z in zarr])   # (nchi, nk)

    chis_jax = jnp.asarray(chis, dtype=jnp.float64)
    w_jax    = jnp.asarray(w,    dtype=jnp.float64)
    lam_arr  = jnp.asarray(y,    dtype=jnp.float64)

    # Solve saddle (δ₁*, δ₂*)(χ, λ): vmap over χ-slices, scan over λ
    solve_all   = jax.vmap(
        lambda bw_i, chi_i, w_i: _newton_2cell_scan(
            k, bw_i, chi_i, w_i, theta1, theta2, recal, filter_type, lam_arr, n_iter
        )
    )
    delta_zeros = solve_all(bw_stack, chis_jax, w_jax)   # (nchi, nlambda, 2)
    d1 = delta_zeros[..., 0]   # (nchi, nlambda)
    d2 = delta_zeros[..., 1]

    # ψ(δ₁*, δ₂*) for all (χ, λ) pairs: vmap over χ, vmap over λ
    psi_grid = jax.vmap(
        lambda bw_i, chi_i, d1_i, d2_i: jax.vmap(
            lambda dd1, dd2: _psi_2cell(
                k, bw_i, chi_i, dd1, dd2, theta1, theta2, recal, filter_type
            )
        )(d1_i, d2_i)
    )(bw_stack, chis_jax, d1, d2)   # (nchi, nlambda)

    # φ(λ) = dχ [ λ (w · annulus) − 1 · psi_grid ]
    # annulus[i, j] = −δ₁*(i,j) + δ₂*(i,j)
    phi = dchis * (
        lam_arr * jnp.einsum("i,ij->j", w_jax, -d1 + d2)
        - jnp.sum(psi_grid, axis=0)
    )
    return phi


def get_scaled_cgf(theta1, theta2, zarr, chis, dchis, lensing_weight, y, recal, variance):
    """Wrapper: 2-cell SCGF φ(λ)."""
    return get_phi_projec_2cell(
        theta1, theta2, zarr, chis, dchis, lensing_weight, y, recal, variance
    )


# ---------------------------------------------------------------------------
# Projected 1-cell SCGF  φ(λ)  with nonlinear rescaling  [Boyle+2021 Eq. NLPhi]
# ---------------------------------------------------------------------------

def get_phi_projec_1cell(
    theta1, zarr, chis, dchis, w, y, variance_linear, variance_nonlinear, **kwargs
):
    """
    Projected 1-cell SCGF with per-slice nonlinear rescaling:

        φ_nl(λ) = ∫ dχ  max_δ { λ W(χ) δ − ψ_l(δ;χ) / r(χ) }

    where  r(χ) = σ²_nl(R₀) / σ²_l(R₀)  at the Eulerian scale R₀ = χθ₁.
    Assembled slice by slice:
        φ(λ) = dχ Σ_i [ λ W_i δ*(i,j) − ψ_l(δ*(i,j)) / r_i ]

    Parameters
    ----------
    theta1   : float  angular scale θ₁ [rad]
    zarr     : (nchi,) redshifts
    chis     : (nchi,) comoving distances [Mpc]
    dchis    : float   Δχ step
    w        : (nchi,) lensing weights W(χ)
    y        : (nlambda,) λ-grid
    variance_linear    : Variance  built from linear P(k)
    variance_nonlinear : Variance  built from nonlinear Halofit P(k)
    n_iter : int, optional  Newton steps per λ (default 50)
    """
    n_iter = int(kwargs.get("n_iter", 50))

    nchi        = len(chis)
    k           = variance_linear._k_jax
    filter_type = variance_linear.filter_type

    bw_l_stack = jnp.stack([variance_linear._bw_jax[z] for z in zarr])   # (nchi, nk)

    # Per-slice rescaling ratio r(χ) = σ²_nl(R₀) / σ²_l(R₀)
    # evaluated at the fixed Eulerian scale R₀ = χ θ₁  (not Lagrangian scale)
    r_arr = jnp.array([
        float(variance_nonlinear.nonlinear_sigma2(zarr[i], float(chis[i]) * theta1))
        / float(variance_linear.nonlinear_sigma2(zarr[i], float(chis[i]) * theta1))
        for i in range(nchi)
    ])

    chis_jax = jnp.asarray(chis, dtype=jnp.float64)
    w_jax    = jnp.asarray(w,    dtype=jnp.float64)
    lam_arr  = jnp.asarray(y,    dtype=jnp.float64)

    # Solve saddle δ*(χ, λ): vmap over χ-slices, scan over λ
    solve_all   = jax.vmap(
        lambda bw_i, chi_i, r_i, w_i: _newton_1cell_scan(
            k, bw_i, chi_i, r_i, w_i, theta1, filter_type, lam_arr, n_iter
        )
    )
    delta_zeros = solve_all(bw_l_stack, chis_jax, r_arr, w_jax)   # (nchi, nlambda)

    # ψ_l(δ*) for all (χ, λ) pairs
    psi_grid = jax.vmap(
        lambda bw_i, chi_i, deltas_i: jax.vmap(
            lambda d: _psi_1cell(k, bw_i, chi_i, d, theta1, filter_type)
        )(deltas_i)
    )(bw_l_stack, chis_jax, delta_zeros)   # (nchi, nlambda)

    # φ(λ) = dχ Σ_i [ λ W_i δ*(i,j) − ψ_l(δ*(i,j)) / r_i ]
    phi = dchis * (
        lam_arr * jnp.einsum("i,ij->j", w_jax, delta_zeros)
        - jnp.einsum("i,ij->j", 1.0 / r_arr, psi_grid)
    )
    return phi


def get_scaled_cgf_1cell(
    theta1, zarr, chis, dchis, lensing_weight, y, variance_linear, variance_nonlinear
):
    """Wrapper: 1-cell SCGF φ(λ) with linear rate function + NL rescaling."""
    return get_phi_projec_1cell(
        theta1, zarr, chis, dchis, lensing_weight, y,
        variance_linear, variance_nonlinear,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper for backward compatibility
# ---------------------------------------------------------------------------

def get_psi_1cell(variance, chi, recal, z, delta, theta1):
    """
    1-cell LDT action ψ(δ) as a Variance-object wrapper.

        ψ_l(δ) = τ(1+δ)² · recal / (2 σ²_l(R_lag))

    For the physical 1-cell rate function use recal=1.0.
    recal is kept in the signature for API compatibility.
    """
    return _psi_1cell(
        variance._k_jax,
        variance._bw_jax[z],
        jnp.asarray(chi,   dtype=jnp.float64).squeeze(),
        jnp.asarray(delta, dtype=jnp.float64).reshape(()),
        theta1,
        variance.filter_type,
    ) * recal


# ---------------------------------------------------------------------------
# Deprecated stub
# ---------------------------------------------------------------------------

def get_lambda_crit_1cell(*args, **kwargs):
    """
    Deprecated: use find_lambda_range_1cell from wale.CriticalPoints.

    The old implementation used max|ψ'|/|W| over the full δ-grid, dominated
    by the singularity near δ=−1, returning an incorrectly large bound.
    The replacement locates the genuine turning point on the positive branch
    and returns a symmetric range ±λ_crit · safety_factor.
    """
    raise NotImplementedError(
        "get_lambda_crit_1cell is deprecated. "
        "Use find_lambda_range_1cell from wale.CriticalPoints."
    )
