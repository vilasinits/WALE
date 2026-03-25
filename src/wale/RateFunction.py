import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# LDT density → τ mapping
# ---------------------------------------------------------------------------

def get_tau(rho):
    """
    Large-deviation theory mapping τ(ρ).

    Parameters
    ----------
    rho : float or array_like
        Local density ρ = 1 + δ.

    Returns
    -------
    tau : same type as input
    """
    nu = 1.4
    return nu * (1.0 - rho ** (-1.0 / nu))


# ---------------------------------------------------------------------------
# 2-cell rate function ψ(δ₁, δ₂)
# ---------------------------------------------------------------------------

def get_psi_2cell(variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """
    2-cell LDT action ψ(δ₁, δ₂).

    All inputs are scalars (or (1,)-arrays that are squeezed internally).
    Returns a JAX scalar.

    Parameters
    ----------
    variance : Variance
        Provides nonlinear_sigma2.
    chi : float or array
        Comoving distance (Mpc).  Squeezed to a scalar internally.
    recal : float
        Empirical recalibration factor.
    z : float
        Redshift (used as dict key into variance.pk).
    delta1, delta2 : float
        Density contrasts in each cell.
    theta1, theta2 : float
        Angular scales (radians) of the two cells.
    """
    d1 = jnp.asarray(delta1, dtype=jnp.float64).reshape(())
    d2 = jnp.asarray(delta2, dtype=jnp.float64).reshape(())
    chi_s = jnp.asarray(chi, dtype=jnp.float64).squeeze()

    tau1 = get_tau(1.0 + d1)
    tau2 = get_tau(1.0 + d2)

    R1 = chi_s * jnp.sqrt(jnp.maximum(1.0 + d1, 0.01)) * theta1
    R2 = chi_s * jnp.sqrt(jnp.maximum(1.0 + d2, 0.01)) * theta2

    sig12 = variance.nonlinear_sigma2(redshift=z, R1=R1, R2=R2)
    sig11 = variance.nonlinear_sigma2(redshift=z, R1=R1, R2=R1)
    sig22 = variance.nonlinear_sigma2(redshift=z, R1=R2, R2=R2)

    det = sig11 * sig22 - sig12 ** 2
    psi = (
        (sig11 * tau2 ** 2 - 2.0 * sig12 * tau1 * tau2 + sig22 * tau1 ** 2)
        * recal
        / (det * 2.0)
    )
    return jnp.squeeze(psi)


# ---------------------------------------------------------------------------
# Partial derivatives of ψ (central finite differences)
# ---------------------------------------------------------------------------

def get_psi_derivative_delta1(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂ψ/∂δ₁ via central finite differences."""
    psi_p = get_psi_2cell(variance, chi, recal, z, delta1 + deld, delta2, theta1, theta2)
    psi_m = get_psi_2cell(variance, chi, recal, z, delta1 - deld, delta2, theta1, theta2)
    return (psi_p - psi_m) / (2.0 * deld)


def get_psi_derivative_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂ψ/∂δ₂ via central finite differences."""
    psi_p = get_psi_2cell(variance, chi, recal, z, delta1, delta2 + deld, theta1, theta2)
    psi_m = get_psi_2cell(variance, chi, recal, z, delta1, delta2 - deld, theta1, theta2)
    return (psi_p - psi_m) / (2.0 * deld)


def get_psi_2nd_derivative_delta1(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂²ψ/∂δ₁² via second-order central finite differences."""
    psi_p = get_psi_2cell(variance, chi, recal, z, delta1 + deld, delta2, theta1, theta2)
    psi_m = get_psi_2cell(variance, chi, recal, z, delta1 - deld, delta2, theta1, theta2)
    psi_0 = get_psi_2cell(variance, chi, recal, z, delta1, delta2, theta1, theta2)
    return (psi_p - 2.0 * psi_0 + psi_m) / (deld ** 2)


def get_psi_2nd_derivative_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2):
    """∂²ψ/∂δ₂² via second-order central finite differences."""
    psi_p = get_psi_2cell(variance, chi, recal, z, delta1, delta2 + deld, theta1, theta2)
    psi_m = get_psi_2cell(variance, chi, recal, z, delta1, delta2 - deld, theta1, theta2)
    psi_0 = get_psi_2cell(variance, chi, recal, z, delta1, delta2, theta1, theta2)
    return (psi_p - 2.0 * psi_0 + psi_m) / (deld ** 2)


def get_psi_mixed_derivative_delta1_delta2(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    """∂²ψ/∂δ₁∂δ₂ via central finite differences."""
    h = deld
    pp = get_psi_2cell(variance, chi, recal, z, delta1 + h, delta2 + h, theta1, theta2)
    pm = get_psi_2cell(variance, chi, recal, z, delta1 + h, delta2 - h, theta1, theta2)
    mp_ = get_psi_2cell(variance, chi, recal, z, delta1 - h, delta2 + h, theta1, theta2)
    mm = get_psi_2cell(variance, chi, recal, z, delta1 - h, delta2 - h, theta1, theta2)
    return (pp - pm - mp_ + mm) / (4.0 * h * h)


def psi_derivative_determinant(deld, delta1, delta2, z, variance, chi, recal, theta1, theta2):
    """
    det(Hψ) = ∂²ψ/∂δ₁² · ∂²ψ/∂δ₂² − (∂²ψ/∂δ₁∂δ₂)²
    """
    psi_11 = get_psi_2nd_derivative_delta1(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2)
    psi_22 = get_psi_2nd_derivative_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2)
    psi_12 = get_psi_mixed_derivative_delta1_delta2(deld, variance, chi, recal, z, delta1, delta2, theta1, theta2)
    return psi_11 * psi_22 - psi_12 ** 2


# ---------------------------------------------------------------------------
# Newton solver for 2D systems (replaces scipy.optimize.root)
# ---------------------------------------------------------------------------

def _newton_2d(f, x0, n_iter=15, tol=1e-11):
    """
    Solve 2D system f(x) = 0 using Newton's method.

    The 2×2 Jacobian is estimated via central finite differences (4 evaluations
    of f per Newton step), which avoids any JAX tracing through f.

    Parameters
    ----------
    f : callable
        f(x) → jnp array of shape (2,).
    x0 : array_like
        Initial guess, shape (2,).
    n_iter : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance on ‖f(x)‖.

    Returns
    -------
    x : jnp array, shape (2,)
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    h = 1e-6
    e0 = jnp.array([h, 0.0])
    e1 = jnp.array([0.0, h])

    for _ in range(n_iter):
        fval = f(x)
        if float(jnp.linalg.norm(fval)) < tol:
            break
        # Central-difference Jacobian: 4 evaluations
        col0 = (f(x + e0) - f(x - e0)) / (2.0 * h)
        col1 = (f(x + e1) - f(x - e1)) / (2.0 * h)
        J = jnp.stack([col0, col1], axis=-1)   # (2, 2)
        try:
            dx = jnp.linalg.solve(J, -fval)
        except Exception:
            break
        x = x + dx

    return x


# ---------------------------------------------------------------------------
# Projected 2-cell SCGF φ(y) via saddle-point equations
# ---------------------------------------------------------------------------

def get_phi_projec_2cell(
    theta1, theta2, zarr, chis, dchis, w, y, recal, variance, **kwargs
):
    """
    Projected 2-cell φ(y) computed by solving saddle-point equations.

    The root-finding step uses a Newton solver with central-difference
    Jacobian (replaces scipy.optimize.root).  Warm-starting over the y-grid
    is preserved for robustness.

    Parameters
    ----------
    theta1, theta2 : float
        Angular scales (radians).
    zarr : array_like
        Redshifts at each chi slice.
    chis : array_like
        Comoving distances (Mpc).
    dchis : float
        Integration step Δχ.
    w : array_like
        Lensing weights W(χ).
    y : array_like
        SCGF slope values.
    recal : float
        Recalibration factor.
    variance : Variance
        Provides nonlinear_sigma2.
    deld : float, optional
        Finite-difference step (default 1e-6).
    n_iter : int, optional
        Newton iterations per solve (default 30).
    verbose : bool, optional
        Print progress (default False).
    """
    deld = kwargs.get("deld", 1e-6)
    n_iter = kwargs.get("n_iter", 30)
    verbose = kwargs.get("verbose", False)

    nchi = len(chis)
    ny = len(y)

    def to_solve(delta, A, chi_val, z_val):
        """Saddle-point residual — returns jnp array of shape (2,)."""
        d1, d2 = delta[0], delta[1]
        psi_d1_fwd = get_psi_2cell(variance, chi_val, recal, z_val, d1 + deld, d2, theta1, theta2)
        psi_d1_bwd = get_psi_2cell(variance, chi_val, recal, z_val, d1 - deld, d2, theta1, theta2)
        psi_d2_fwd = get_psi_2cell(variance, chi_val, recal, z_val, d1, d2 + deld, theta1, theta2)
        psi_d2_bwd = get_psi_2cell(variance, chi_val, recal, z_val, d1, d2 - deld, theta1, theta2)
        eq1 = jnp.squeeze(2.0 * A + (psi_d1_fwd - psi_d1_bwd))
        eq2 = jnp.squeeze(-2.0 * A + (psi_d2_fwd - psi_d2_bwd))
        return jnp.array([eq1, eq2])

    # delta_zeros[i, j] = (δ₁*, δ₂*) at chi[i], y[j]
    delta_zeros = np.zeros((nchi, ny, 2))

    for i in range(nchi):
        if verbose:
            print(f"Iteration {i * 100 / nchi:.1f} %", end="\r")
        chi_val = np.array([chis[i]])
        z_val = zarr[i]

        for j in range(ny):
            A = y[j] * w[i] * deld
            x0 = jnp.zeros(2) if j == 0 else jnp.asarray(delta_zeros[i, j - 1])

            sol = _newton_2d(
                lambda delta, _A=A, _chi=chi_val, _z=z_val: to_solve(delta, _A, _chi, _z),
                x0,
                n_iter=n_iter,
            )
            delta_zeros[i, j] = np.array(sol)

    # Assemble φ(y) = ∫ dχ [y w(χ)(−δ₁* + δ₂*) − ψ(δ₁*, δ₂*)]
    phi_proj = []
    for i in range(ny):
        phi_ = 0.0
        for j in range(nchi):
            phi_ += (
                y[i] * w[j] * (-delta_zeros[j, i, 0] + delta_zeros[j, i, 1])
                - get_psi_2cell(
                    variance,
                    chi=np.array([chis[j]]),
                    recal=recal,
                    z=zarr[j],
                    delta1=delta_zeros[j, i, 0],
                    delta2=delta_zeros[j, i, 1],
                    theta1=theta1,
                    theta2=theta2,
                )
            ) * dchis
        phi_proj.append(phi_)

    return jnp.asarray(phi_proj)


def get_scaled_cgf(theta1, theta2, zarr, chis, dchis, lensing_weight, y, recal, variance):
    """
    Wrapper: compute the scaled cumulant generating function (SCGF) φ(y).
    """
    return get_phi_projec_2cell(
        theta1, theta2, zarr, chis, dchis, lensing_weight, y, recal, variance
    )
