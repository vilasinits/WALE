import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

jax.config.update("jax_enable_x64", True)

from wale.RateFunction import get_scaled_cgf, get_scaled_cgf_1cell


# ---------------------------------------------------------------------------
# JAX RK4 ODE integrator (replaces scipy.integrate.solve_ivp)
# ---------------------------------------------------------------------------

def _rk4_integrate(ode, y0, t0, t1, n_steps):
    """
    Fixed-step RK4 integration using jax.lax.scan.

    Parameters
    ----------
    ode : callable
        ode(t, y) → dy/dt, where y is a real jnp array of shape (2,).
    y0 : jnp array, shape (2,)
        Initial state.
    t0, t1 : float
        Integration interval.
    n_steps : int
        Number of uniform steps.

    Returns
    -------
    t_vals : jnp array, shape (n_steps + 1,)
    ys : jnp array, shape (n_steps + 1, 2)
        Solution at each t.
    """
    dt = (t1 - t0) / n_steps
    t_eval = jnp.linspace(t0, t1 - dt, n_steps)

    def rk4_step(y, t):
        k1 = ode(t, y)
        k2 = ode(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = ode(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = ode(t + dt,       y + dt * k3)
        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_new, y_new

    _, ys = jax.lax.scan(rk4_step, y0, t_eval)
    # Prepend initial state
    ys_full = jnp.concatenate([y0[None, :], ys], axis=0)  # (n_steps+1, 2)
    t_vals = jnp.linspace(t0, t1, n_steps + 1)
    return t_vals, ys_full


class computePDF:
    """
    Compute the convergence PDF via Large Deviation Theory.

    1. Solve saddle-point equations → SCGF φ(λ).
    2. Legendre-transform to get the polynomial approximation p(τ).
    3. Integrate the implicit ODE dτ/dt = i p'(τ) / (1 − it p''(τ)) along
       the imaginary λ-axis using JAX RK4 (replaces scipy solve_ivp).
    4. Evaluate the Bromwich integral for all κ simultaneously via jax.vmap.
    """

    def __init__(self, variables, variance, kappa=None, plot_scgf=False,
                 single_cell=False, variance_linear=None):
        """
        Parameters
        ----------
        variance : Variance
            For 2-cell: built from nonlinear P(k).
            For 1-cell: also the nonlinear Variance (used for the per-slice
            rescaling ratio r = σ²_nl/σ²_l).
        variance_linear : Variance or None
            For 1-cell only: Variance built from the **linear** P(k).
            Required when single_cell=True; ignored for 2-cell.
        """
        self.variables      = variables
        self.plot_scgf      = plot_scgf
        self.variance       = variance          # nonlinear
        self.variance_linear = variance_linear  # linear (1-cell only)
        self.single_cell    = single_cell
        if kappa is not None:
            self.kappa = kappa
        else:
            edges = np.linspace(-0.06, 0.06, 801)
            self.kappa = 0.5 * (edges[:-1] + edges[1:])
        self.pdf_values, self.kappa_values = self.compute_pdf_values()

    def get_scgf(self):
        """Compute the SCGF φ(λ) on the lambda grid."""
        if self.single_cell:
            if self.variance_linear is None:
                raise ValueError(
                    "computePDF: variance_linear must be provided for single_cell=True. "
                    "Build it with Variance(cosmo, filter_type, pk=cosmo.plin)."
                )
            scgf = get_scaled_cgf_1cell(
                self.variables.theta1_radian,
                self.variables.redshifts,
                self.variables.chis,
                self.variables.dchi,
                self.variables.lensingweights,
                self.variables.lambdas,
                self.variance_linear,    # linear variance → rate function
                self.variance,           # nonlinear variance → rescaling ratio
            )
        else:
            scgf = get_scaled_cgf(
                self.variables.theta1_radian,
                self.variables.theta2_radian,
                self.variables.redshifts,
                self.variables.chis,
                self.variables.dchi,
                self.variables.lensingweights,
                self.variables.lambdas,
                self.variables.recal_value,
                self.variance,
            )
        return scgf

    def compute_phi_values(self):
        """
        Compute φ(λ) on the imaginary axis via the implicit-function ODE.

        Steps
        -----
        1. Spline the real-λ SCGF, extract dφ/dλ.
        2. Fit degree-5 polynomial p to (τ_eff, dφ/dλ) data.
        3. Integrate the IFT ODE with JAX RK4 to obtain τ(it) for t=0..N.
        4. φ(it) = it · p(τ(it)) − τ(it)²/2.

        Returns
        -------
        lambda_new : jnp complex array, shape (N,)
        phi_values : jnp complex array, shape (N,)
        """
        scgf = self.get_scgf()
        scgf_1d = np.asarray(scgf).reshape(len(self.variables.lambdas), -1)[:, 0]

        # Trim non-finite SCGF values (appear near critical-λ boundaries where
        # the Newton saddle-point solver diverges; safe to drop edge points).
        lambdas_all = np.asarray(self.variables.lambdas)
        finite_mask = np.isfinite(scgf_1d)
        if not np.all(finite_mask):
            n_bad = int(np.sum(~finite_mask))
            print(f"Warning: {n_bad}/{len(scgf_1d)} SCGF values are non-finite; trimming.")
        lambdas = lambdas_all[finite_mask]
        scgf_1d = scgf_1d[finite_mask]

        scgf_spline = CubicSpline(lambdas, scgf_1d)
        dscgf = scgf_spline(lambdas, 1)

        if self.plot_scgf:
            plt.figure(figsize=(4, 4))
            plt.plot(lambdas, scgf_1d)
            plt.show()

        # Build Legendre-transform coordinate: τ_eff = sign(λ) √(2(λ dφ/dλ - φ))
        raw = 2.0 * (lambdas * dscgf - scgf_1d)
        tau_effective = np.sqrt(np.maximum(raw, 0.0))
        x_data = np.sign(lambdas) * tau_effective
        y_data = dscgf

        # Degree-5 polynomial p such that p'(τ) ≈ λ  (the saddle-point relation)
        coeffs = np.polyfit(x_data, y_data, 5)
        # Derivative coefficients (numpy poly1d convention: descending powers)
        dp_c = np.polyder(coeffs)   # degree 4
        d2p_c = np.polyder(dp_c)    # degree 3

        # Convert to complex128 for evaluation at complex τ
        p_c_jax   = jnp.asarray(coeffs,  dtype=jnp.complex128)
        dp_c_jax  = jnp.asarray(dp_c,   dtype=jnp.complex128)
        d2p_c_jax = jnp.asarray(d2p_c,  dtype=jnp.complex128)

        N = 60000
        lambda_new = 1j * jnp.arange(N, dtype=jnp.float64)  # λ = it, t=0..N-1

        # ODE: dτ/dt = i p'(τ) / (1 − it p''(τ))
        # τ = y[0] + i y[1], integrated as a real 2-vector.
        def ode_rhs(t, y):
            tau = y[0] + 1j * y[1]
            lam = 1j * t
            dp_val  = jnp.polyval(dp_c_jax,  tau)
            d2p_val = jnp.polyval(d2p_c_jax, tau)
            denom   = 1.0 - lam * d2p_val
            dtau    = jnp.where(
                jnp.abs(denom) < 1e-30,
                0.0 + 0.0j,
                1j * dp_val / denom,
            )
            return jnp.array([dtau.real, dtau.imag])

        _, ys = _rk4_integrate(ode_rhs, jnp.zeros(2), 0.0, float(N - 1), N - 1)
        taus = ys[:, 0] + 1j * ys[:, 1]  # (N,) complex

        phi_values = lambda_new * jnp.polyval(p_c_jax, taus) - taus ** 2 / 2.0
        return lambda_new, phi_values

    def compute_pdf_values(self):
        """
        Bromwich integral for all κ simultaneously using jax.vmap.

        P(κ) = Im[ ∫ exp(−λκ + φ(λ)) dλ ] / π,   λ = it

        The trapezoidal weights include a Lanczos σ-factor sinc(t/N) which
        tapers the integrand smoothly to zero at the truncation point t=N,
        suppressing the Gibbs-like oscillations that cause O(1e-14) negative
        PDF values in the tails.  The taper is negligible in the bulk
        (sinc(t/N) ≈ 1 for t ≪ N) so it does not bias the PDF.
        After integration, any residual negative values are clipped to zero —
        physically necessary since a PDF cannot be negative.

        Returns
        -------
        pdf_values : np.ndarray, shape (nkappa,)
        kappa_values : array
        """
        lambda_new, phi_values = self.compute_phi_values()
        kappa_values = self.kappa

        N = len(lambda_new)

        # Trapezoidal weights on the imaginary axis
        delta_lambda = 1j  # step size
        trap_weights = jnp.ones(N, dtype=jnp.complex128) * delta_lambda
        trap_weights = trap_weights.at[0].set(delta_lambda * 0.5)
        trap_weights = trap_weights.at[-1].set(delta_lambda * 0.5)

        # Lanczos σ-factor: sinc(t/N) tapers the truncated sum to zero at t=N,
        # eliminating ringing from the hard cutoff.
        # jnp.sinc(x) = sin(πx)/(πx), so sinc(t/N) → 1 at t=0, → 0 at t=N.
        t_arr = jnp.arange(N, dtype=jnp.float64)
        lanczos = jnp.sinc(t_arr / N)
        trap_weights = trap_weights * lanczos

        kappa_jax = jnp.asarray(kappa_values, dtype=jnp.float64)

        @jax.jit
        def _bromwich(kappa_arr):
            """Vectorised Bromwich integral over all κ values."""
            def _pdf_one(kappa):
                integrand = jnp.exp(-lambda_new * kappa + phi_values) * trap_weights
                return jnp.imag(jnp.sum(integrand) / jnp.pi)

            return jax.vmap(_pdf_one)(kappa_arr)

        pdf_values = np.maximum(np.array(_bromwich(kappa_jax)), 0.0)
        return list(pdf_values), kappa_values
