import functools
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.path import Path

from scipy.ndimage import convolve, gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import brentq

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    Parallel = None
    delayed = None
    _HAVE_JOBLIB = False

from .RateFunction import (
    get_tau,
    get_psi_2cell,
    get_psi_derivative_delta1,
    get_psi_derivative_delta2,
    _psi_2cell,
)
from .FilterFunctions import top_hat_filter, starlet_filter
from .VarianceCalculator import _simpson_jax


@functools.partial(jax.jit, static_argnums=(6,))
def _psi_grid_jit(k, bw, R1_arr, R2_arr, tau1, tau2, filter_type, recal_value):
    """
    JIT-compiled ψ(δ₁, δ₂) grid.

    bw = k · P(k) · simpson_weights / (2π), pre-computed per (z, pk).
    Uses matmul for sig12 to avoid allocating a (ngrid, ngrid, nk) array.
    Compiled once per filter_type; all z-slice calls reuse the same XLA graph.
    """
    if filter_type == "tophat":
        W1 = top_hat_filter(k[None, :], R1_arr[:, None])   # (ngrid, nk)
        W2 = top_hat_filter(k[None, :], R2_arr[:, None])
    else:
        W1 = starlet_filter(k[None, :], R1_arr[:, None])
        W2 = starlet_filter(k[None, :], R2_arr[:, None])

    sig11 = jnp.sum(W1 ** 2 * bw[None, :], axis=-1)        # (ngrid,)
    sig22 = jnp.sum(W2 ** 2 * bw[None, :], axis=-1)        # (ngrid,)
    sig12 = (W1 * bw[None, :]) @ W2.T                       # (ngrid, ngrid) via BLAS

    S11 = sig11[:, None]
    S22 = sig22[None, :]
    det = S11 * S22 - sig12 ** 2
    psi = (
        (S11 * tau2 ** 2 - 2.0 * sig12 * tau1 * tau2 + S22 * tau1 ** 2)
        * recal_value / (det * 2.0)
    )
    return psi


@functools.partial(jax.jit, static_argnums=(3,))
def _psi_grid_1cell_jit(k, bw, R_arr, filter_type, tau_arr, recal_value):
    """
    JIT-compiled ψ₁(δ) on a 1D δ-grid.

    bw = k · P(k) · simpson_weights / (2π), pre-computed per (z, pk).
    Compiled once per filter_type; all z-slice calls reuse the same XLA graph.

    Parameters
    ----------
    k : jnp array, shape (nk,)
    bw : jnp array, shape (nk,)
        Pre-computed k·P(k)·w_simp/(2π) for this z-slice.
    R_arr : jnp array, shape (ngrid,)
        Lagrangian radii R = χ·√max(1+δ, 0.01)·θ₁ for each δ.
    filter_type : str
        "tophat" or "starlet" (static — determines JIT compilation branch).
    tau_arr : jnp array, shape (ngrid,)
        τ(ρ) evaluated at ρ = 1+δ for each δ in the grid.
    recal_value : float
        Empirical recalibration factor.

    Returns
    -------
    psi : jnp array, shape (ngrid,)
        ψ₁(δ) = τ(ρ)²·recal / (2 σ²(R, R)) for each δ.
    """
    if filter_type == "tophat":
        W = top_hat_filter(k[None, :], R_arr[:, None])   # (ngrid, nk)
    else:
        W = starlet_filter(k[None, :], R_arr[:, None])   # (ngrid, nk)

    sig11 = jnp.sum(W ** 2 * bw[None, :], axis=-1)       # (ngrid,)
    psi   = tau_arr ** 2 * recal_value / (2.0 * sig11)   # (ngrid,)
    return psi


@functools.partial(jax.jit, static_argnums=(4,))
def _g_batch_jax(k, bw, chi, recal, filter_type, theta1, theta2, d1_arr, d2_arr, lw):
    """
    Vectorised g(d1, d2) = (∂ψ/∂δ₁ + ∂ψ/∂δ₂) / W for all points on a contour.

    g = 0 is the stationarity condition on the det(Hψ)=0 catastrophe manifold
    (i.e. the critical-point condition for the 2-cell SCGF).

    Replaces the n_samples Python loop in _roots_along_path with one JIT+vmap call.

    Parameters
    ----------
    k, bw    : (nk,) wavenumber and base-weight arrays for this z-slice
    chi      : scalar  comoving distance [Mpc]
    recal    : scalar  recalibration factor
    filter_type : 'tophat' or 'starlet' (static)
    theta1, theta2 : angular scales [rad]
    d1_arr, d2_arr : (n_samples,)  contour coordinates
    lw       : scalar  lensing weight W(χ)
    """
    deld = 1e-6

    def g_one(d1, d2):
        dpsi1 = (_psi_2cell(k, bw, chi, d1 + deld, d2, theta1, theta2, recal, filter_type)
                 - _psi_2cell(k, bw, chi, d1 - deld, d2, theta1, theta2, recal, filter_type)) / (2.0 * deld)
        dpsi2 = (_psi_2cell(k, bw, chi, d1, d2 + deld, theta1, theta2, recal, filter_type)
                 - _psi_2cell(k, bw, chi, d1, d2 - deld, theta1, theta2, recal, filter_type)) / (2.0 * deld)
        return (dpsi1 + dpsi2) / lw

    return jax.vmap(g_one)(d1_arr, d2_arr)


class CriticalPointsFinder:
    r"""
    Fast critical-points finder using:
      1) Convolution stencils for Hessian determinant of ψ on a grid.
      2) Marching-squares (matplotlib.contour) to get det(Hψ)=0 curve(s).
      3) Root-finding along those curves for g(d1,d2) = dψ/dd1 + dψ/dd2.

    Notes
    -----
    - This is faster and more reliable than uniform fine grids + scattered splines.
    - It evaluates ψ(d1,d2) once per z on a coarse/moderate grid, then refines *only
      where needed* (along the zero-determinant contour).
    """

    def __init__(self, variables, lw, z, chis, ngrid=50, plot=False,
                 smooth_sigma=0.0,  # small smoothing of det(H) to tame jagged edges (0 = off)
                 ):
        self.variables = variables
        self.plot = plot
        self.smooth_sigma = float(smooth_sigma)
        # print(f"       Setting ngrid = {ngrid}. Increase for accuracy; runtime ~ O(ngrid^2).")

        self.delta1_vals = np.linspace(-0.99, 1.99, ngrid)
        self.delta2_vals = np.linspace(-0.99, 1.99, ngrid)
        self.D1, self.D2 = np.meshgrid(self.delta1_vals, self.delta2_vals, indexing="ij")

        self.dx = float(self.delta1_vals[1] - self.delta1_vals[0])
        self.dy = float(self.delta2_vals[1] - self.delta2_vals[0])

        self.lw = np.asarray(lw)
        self.z = np.asarray(z)
        self.chis = np.asarray(chis)

        # sanity alignment
        if not (len(self.lw) == len(self.z) == len(self.chis)):
            raise ValueError("lw, z, and chis must have the same length (aligned slices).")

        # prefetch constants
        self.recal_value = self.variables.recal_value
        self.theta1 = self.variables.theta1_radian
        self.theta2 = self.variables.theta2_radian
        self.h = self.variables.cosmo.h

        # 2nd-derivative central-difference stencils
        self._Dxx = np.array([[0, 0, 0],
                              [1,-2, 1],
                              [0, 0, 0]], dtype=float) / (self.dx*self.dx)
        self._Dyy = np.array([[0, 1, 0],
                              [0,-2, 0],
                              [0, 1, 0]], dtype=float) / (self.dy*self.dy)
        # mixed derivative (∂²/∂x∂y)
        self._Dxy = np.array([[ 1, 0,-1],
                              [ 0, 0, 0],
                              [-1, 0, 1]], dtype=float) / (4*self.dx*self.dy)

        # Pre-compute τ grids — fixed for this finder (same δ grid for every z-slice)
        self._tau1 = jnp.asarray(get_tau(1.0 + self.D1))  # (ngrid, ngrid)
        self._tau2 = jnp.asarray(get_tau(1.0 + self.D2))

    # ---------- core numeric helpers ----------

    def _psi_grid_batch(self, variance):
        """
        Compute ψ(δ₁, δ₂) for ALL z-slices simultaneously via vmap.

        Replaces the per-slice Python loop with one fused XLA kernel over
        the (nchi, ngrid, ngrid) grid.

        Returns
        -------
        psi_all : np.ndarray, shape (nchi, ngrid, ngrid)
        """
        k        = variance._k_jax
        bw_stack = jnp.stack([variance._bw_jax[z] for z in self.z])   # (nchi, nk)
        R1_stack = jnp.stack([                                          # (nchi, ngrid)
            jnp.asarray(chi * np.sqrt(np.maximum(1.0 + self.delta1_vals, 0.01)) * self.theta1)
            for chi in self.chis
        ])
        R2_stack = jnp.stack([                                          # (nchi, ngrid)
            jnp.asarray(chi * np.sqrt(np.maximum(1.0 + self.delta2_vals, 0.01)) * self.theta2)
            for chi in self.chis
        ])
        psi_all = jax.vmap(
            lambda bw_i, R1_i, R2_i: _psi_grid_jit(
                k, bw_i, R1_i, R2_i,
                self._tau1, self._tau2,
                variance.filter_type, self.recal_value,
            )
        )(bw_stack, R1_stack, R2_stack)
        return np.asarray(psi_all, dtype=float)   # (nchi, ngrid, ngrid)

    def _psi_grid(self, variance, chi_value, z):
        """
        Evaluate ψ(δ₁, δ₂) on the full (ngrid × ngrid) grid.
        Returns from the pre-computed batch cache when available.
        """
        if hasattr(self, '_psi_cache') and z in self._psi_cache:
            return self._psi_cache[z]
        k  = variance._k_jax
        bw = variance._bw_jax[z]
        R1_arr = jnp.asarray(
            chi_value * np.sqrt(np.maximum(1.0 + self.delta1_vals, 0.01)) * self.theta1
        )
        R2_arr = jnp.asarray(
            chi_value * np.sqrt(np.maximum(1.0 + self.delta2_vals, 0.01)) * self.theta2
        )
        psi = _psi_grid_jit(
            k, bw, R1_arr, R2_arr,
            self._tau1, self._tau2,
            variance.filter_type, self.recal_value,
        )
        return np.asarray(psi, dtype=float)

    def _det_hessian(self, psi):
        """Compute det(Hψ) via convolution stencils."""
        psi_xx = convolve(psi, self._Dxx, mode="nearest")
        psi_yy = convolve(psi, self._Dyy, mode="nearest")
        psi_xy = convolve(psi, self._Dxy, mode="nearest")
        detH = psi_xx*psi_yy - psi_xy*psi_xy
        if self.smooth_sigma > 0:
            detH = gaussian_filter(detH, sigma=self.smooth_sigma, mode="nearest")
        return detH

    # def _zero_contours(self, detH):
    #     """
    #     Extract zero-level contours from detH using matplotlib's marching squares
    #     without leaving a figure behind.
    #     Returns a list of arrays of shape (k, 2) with columns [d1, d2].
    #     """
    #     fig, ax = plt.subplots()
    #     try:
    #         CS = ax.contour(self.delta1_vals, self.delta2_vals, detH.T, levels=[0.0])
    #         paths = []
    #         if CS.collections and CS.collections[0].get_paths():
    #             for p in CS.collections[0].get_paths():
    #                 v = p.vertices  # (n, 2) as [[x(d1), y(d2)], ...]
    #                 paths.append(v.copy())
    #         return paths
    #     finally:
    #         plt.close(fig)
    def _zero_contours(self, detH):
        """
        Extract zero-level contours using scikit-image (no figures created).
        """
        import numpy as np
        from skimage import measure

        # detH is on grid (d1, d2). We want contours at 0.
        # skimage expects array indexed as [row(y), col(x)], so transpose if needed.
        contours = measure.find_contours(detH.T, level=0.0)
        # contours are sequences of (row, col) in pixel coords; map them to d1/d2 axes
        d1 = self.delta1_vals
        d2 = self.delta2_vals
        paths = []
        for c in contours:
            # c[:,0] ~ row index over d2; c[:,1] ~ col index over d1
            y = np.interp(c[:,0], np.arange(len(d2)), d2)
            x = np.interp(c[:,1], np.arange(len(d1)), d1)
            paths.append(np.column_stack([x, y]))
        return paths


    def _g(self, d1, d2, variance, lw, z, chi_value, deld=1e-6):
        dpsi_d1 = get_psi_derivative_delta1(
            deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
        )
        dpsi_d2 = get_psi_derivative_delta2(
            deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
        )
        # .item() extracts a Python float from a 0-d or single-element array,
        # which is required for NumPy 2.x (float() no longer accepts 1-D arrays).
        return np.asarray(dpsi_d1 + dpsi_d2).flat[0] / lw

    def _roots_along_path(
        self,
        path_xy,
        variance,
        lw,
        z,
        chi_value,
        n_samples=100,
        flip_sign=True,
        return_mode="derivative",
        deld=1e-6,
    ):
        """
        Find roots of g(d1,d2)=0 restricted to the given det(Hψ)=0 contour 'path_xy' (n×2).
        Parameterize by arclength, sample g, bracket sign changes, refine with brentq.
        Returns list of critical values; by default returns -(dψ/dδ1)*h/lw to match original code.
        """
        if path_xy is None or len(path_xy) < 2:
            return []

        x = np.asarray(path_xy[:, 0], dtype=float)
        y = np.asarray(path_xy[:, 1], dtype=float)

        # arclength parameterization
        ds = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate(([0.0], np.cumsum(ds)))
        L = float(s[-1])
        if not np.isfinite(L) or L <= 0.0:
            return []

        fx = interp1d(s, x, kind="linear", assume_sorted=True)
        fy = interp1d(s, y, kind="linear", assume_sorted=True)

        # define the sampling grid *before* using it
        n_samples = int(max(16, n_samples))
        ts = np.linspace(0.0, L, n_samples)

        # helper: g along contour
        def g_at_t(tt):
            d1 = float(fx(tt))
            d2 = float(fy(tt))
            return self._g(d1, d2, variance, lw, z, chi_value, deld=deld)

        # helper: scaled dψ/dδ1 along contour (to match your original return scale)
        def d1_deriv_at_t(tt):
            d1 = float(fx(tt))
            d2 = float(fy(tt))
            val = get_psi_derivative_delta1(
                deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
            ) * 1 / lw
            return np.asarray(val).flat[0]

        # Sample g at all contour points simultaneously via vmap (_g_batch_jax).
        # This replaces n_samples sequential calls with one fused XLA kernel.
        d1_arr = jnp.asarray([float(fx(t)) for t in ts], dtype=jnp.float64)
        d2_arr = jnp.asarray([float(fy(t)) for t in ts], dtype=jnp.float64)
        gvals = np.asarray(_g_batch_jax(
            variance._k_jax,
            variance._bw_jax[z],
            jnp.float64(chi_value),
            self.recal_value,
            variance.filter_type,
            self.theta1,
            self.theta2,
            d1_arr,
            d2_arr,
            float(lw),
        ), dtype=float)

        crits = []
        for k in range(ts.size - 1):
            g1, g2 = gvals[k], gvals[k + 1]
            if not (np.isfinite(g1) and np.isfinite(g2)):
                continue

            if g1 == 0.0:
                t0 = ts[k]
                val = -d1_deriv_at_t(t0) if return_mode == "derivative" else -float(fx(t0))
                crits.append(val)
                continue

            if g1 * g2 < 0.0:  # sign change -> refine with brentq
                try:
                    root_t = brentq(lambda tt: g_at_t(tt), ts[k], ts[k + 1], maxiter=200)
                    val = -d1_deriv_at_t(root_t) if return_mode == "derivative" else -float(fx(root_t))
                    crits.append(val)
                except ValueError:
                    # bracketing failed due to numerical issues; skip this interval
                    pass

        return crits


    # ---------- public API ----------

    def get_critical_points(self, variance, lw, z, chi_value, target_levels=1,
                        return_mode="derivative"):
        """
        Compute critical points for one (lw, z, chi_value) triple.
        Returns a Python list of critical values (can be empty).
        """
        # 1) ψ grid
        psi = self._psi_grid(variance, chi_value, z)

        # 2) det(Hψ)
        detH = self._det_hessian(psi)

        # 3) Zero contours
        paths = self._zero_contours(detH)
        if len(paths) == 0:
            if self.plot:
                print(f"   z={z:.3f}: no det(Hψ)=0 contour found.")
            return []

        # Use the longest contours first (often the main, physically relevant one)
        paths.sort(key=lambda v: 0.0 if len(v) < 2 else np.sum(np.hypot(np.diff(v[:,0]), np.diff(v[:,1]))),
                   reverse=True)
        if target_levels is not None and target_levels > 0:
            paths = paths[:target_levels]

        # 4) Find roots of g along each contour
        all_crits = []
        for P in paths:
            crits = self._roots_along_path(P, variance, lw, z, chi_value,
                                        return_mode=return_mode)
            all_crits.extend(crits)
        return all_crits


def find_smallest_pair(critical_values):
    """
    From an array-like of critical points, finds the pair with the smallest Euclidean distance.
    Here critical_values is typically 1-D; this function supports 1-D or 2-D points.
    """
    cv = np.asarray(critical_values)
    if cv.size == 0:
        return None

    if cv.ndim == 1:
        # 1-D case: pair with smallest absolute difference
        cv_sorted = np.sort(cv)
        if cv_sorted.size < 2:
            return None
        diffs = np.diff(cv_sorted)
        k = np.argmin(np.abs(diffs))
        return (cv_sorted[k], cv_sorted[k+1])

    # 2-D points case
    num_points = cv.shape[0]
    if num_points < 2:
        return None

    best_pair = None
    best_dist = np.inf
    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(cv[i] - cv[j])
            if dist < best_dist:
                best_dist = dist
                best_pair = (cv[i], cv[j])
    return best_pair


def _as_scalar(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.ndim > 0 else float(x)

def _process_one_z(i, z_i, lw_i, chi_i, finder, variance, return_mode):
    lw_s = _as_scalar(lw_i)  # ensure scalar lensing weight per z
    crits = finder.get_critical_points(variance, lw=lw_s, z=z_i, chi_value=chi_i,
                                       return_mode=return_mode)
    if len(crits) == 0:
        return []
    crits = np.asarray(crits, dtype=float)
    crits = crits[np.isfinite(crits)]
    return crits.tolist()


def find_critical_points_for_cosmo(
    variables,
    variance,
    ngrid_critical=90,
    plot=False,
    min_z=1,
    max_z=4,
    smooth_sigma=0.0,
    parallel=False,
    n_jobs=-1,
    return_mode="derivative",
):
    """
    High-level driver that returns (smallest_positive, largest_negative)
    aggregated across z in [min_z:max_z].

    Parameters
    ----------
    variables : object
        Must provide attributes:
          - lensingweights, redshifts, chis (indexable, aligned),
          - recal_value, theta1_radian, theta2_radian, cosmo.h
    variance : any
        Passed through to ψ and derivative functions.
    ngrid_critical : int
        Grid resolution (per axis). 60–120 is often a good range.
    plot : bool
        Keep for API compatibility; plotting of g-curves is not produced here.
    min_z, max_z : int
        Slice range (Python-style: start inclusive, end exclusive).
    smooth_sigma : float
        Optional Gaussian smoothing (in grid pixels) applied to det(Hψ) before contouring.
    parallel : bool
        If True and joblib is available, compute slices in parallel.
    n_jobs : int
        Joblib workers (ignored if parallel=False or joblib not installed).

    Returns
    -------
    (smallest_positive, largest_negative)
    """
    # print("   Finding critical points (optimized)...")


    # Align slices for lw, z, chis (IMPORTANT!)
    lw_slice = variables.lensingweights[min_z:max_z]
    z_slice = variables.redshifts[min_z:max_z]
    chi_slice = variables.chis[min_z:max_z]

    finder = CriticalPointsFinder(
        variables,
        lw=lw_slice,
        z=z_slice,
        chis=chi_slice,
        ngrid=ngrid_critical,
        plot=plot,
        smooth_sigma=smooth_sigma,
    )

    # Pre-compute ψ for all z-slices in one vmap call, cache on the finder.
    # _psi_grid() will return from the cache instead of calling _psi_grid_jit per slice.
    psi_all = finder._psi_grid_batch(variance)
    finder._psi_cache = {z: psi_all[i] for i, z in enumerate(finder.z)}

    # Per-z processing
    if parallel and _HAVE_JOBLIB:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_one_z)(i, z_i, lw_i, chi_i, finder, variance, return_mode)
            for i, (z_i, lw_i, chi_i) in enumerate(zip(finder.z, finder.lw, finder.chis))
        )
    else:
        results = [
            _process_one_z(i, z_i, lw_i, chi_i, finder, variance, return_mode)
            for i, (z_i, lw_i, chi_i) in enumerate(zip(finder.z, finder.lw, finder.chis))
        ]

    # Flatten and clean
    flat = np.array([c for sub in results for c in sub], dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        print("  Warning: No critical points found in the specified redshift range.")
        return None, None

    positive = flat[flat > 0]
    negative = flat[flat < 0]

    smallest_positive = np.min(positive) if positive.size else None
    largest_negative = np.max(negative) if negative.size else None
    # print("       Smallest positive / largest negative:",
        #   smallest_positive, "/", largest_negative)

    return smallest_positive, largest_negative


def find_lambda_range_1cell(variables, variance_linear, variance_nonlinear,
                            safety_factor=0.85, delta_pos_max=10.0, ngrid=500):
    """
    1-cell critical λ finder using linear rate function + per-slice rescaling.

    Physics
    -------
    The effective saddle-point condition (after Eq. NLPhi, Boyle et al. 2021):

        r_j · W(χ_j) · λ = ψ'_l(δ*)

    where r_j = σ²_nl(R₀_j) / σ²_l(R₀_j) at the Eulerian scale R₀ = χθ₁,
    and ψ_l uses the **linear** variance at the Lagrangian scale.

    The effective λ(δ) on the positive branch is:

        λ(δ) = ψ'_l(δ) / (r_j · W(χ_j))

    Its turning point (dλ/dδ = 0) gives λ_crit.  The range is symmetric
    (see find_lambda_range_1cell docstring of the previous version for
    the justification).

    Parameters
    ----------
    variables : InitialiseVariables
        Must provide: theta1_radian, redshifts, chis, lensingweights.
    variance_linear : Variance
        Built from the **linear** power spectrum (pk=cosmo.plin).
    variance_nonlinear : Variance
        Built from the nonlinear Halofit power spectrum (pk=cosmo.pnl).
    safety_factor : float
        Fraction of λ_crit (default 0.85).
    delta_pos_max : float
        Upper δ limit for positive-branch scan (default 10.0).
    ngrid : int
        Grid points on positive branch (default 500).

    Returns
    -------
    lambda_min, lambda_max : float
        Symmetric bounds (−λ_crit · safety_factor, +λ_crit · safety_factor).
    """
    theta1 = variables.theta1_radian
    zarr   = np.asarray(variables.redshifts)
    chis   = np.asarray(variables.chis)
    lw     = np.asarray(variables.lensingweights)

    delta_pos = np.linspace(1e-4, delta_pos_max, ngrid)
    tau_pos   = jnp.asarray(get_tau(1.0 + delta_pos), dtype=jnp.float64)
    k         = variance_linear._k_jax   # same k-grid for both variances

    lambda_crit_pos_list = []

    for i, (chi, z, w) in enumerate(zip(chis, zarr, lw)):
        if abs(w) < 1e-12:
            continue

        w   = float(w)

        # ψ_l on positive branch: use linear bw, recal=1 (no simulation rescaling)
        bw_l = variance_linear._bw_jax[z]
        R_pos = jnp.asarray(chi * np.sqrt(1.0 + delta_pos) * theta1, dtype=jnp.float64)
        psi_l = np.asarray(
            _psi_grid_1cell_jit(k, bw_l, R_pos, variance_linear.filter_type, tau_pos, 1.0)
        )

        # Per-slice rescaling ratio r = σ²_nl(R₀) / σ²_l(R₀) at Eulerian scale
        R0 = float(chi * theta1)
        sig_nl = float(variance_nonlinear.nonlinear_sigma2(z, R0))
        sig_l  = float(variance_linear.nonlinear_sigma2(z, R0))
        r      = sig_nl / sig_l

        # Effective λ(δ) = ψ'_l(δ) / (r · W)
        # From saddle condition: λ W = ψ'_l(δ*)/r  →  λ = ψ'_l / (r W)
        dpsi_l  = np.gradient(psi_l, delta_pos)
        lam_pos = dpsi_l / (r * w)

        # First sign-change in dλ/dδ: peak = λ_crit for this slice
        dlam  = np.diff(lam_pos)
        peaks = np.where((dlam[:-1] > 0) & (dlam[1:] <= 0))[0]
        if peaks.size > 0:
            lambda_crit_pos_list.append(lam_pos[peaks[0] + 1])
        else:
            lambda_crit_pos_list.append(lam_pos[-1])

    if len(lambda_crit_pos_list) == 0:
        raise ValueError(
            "find_lambda_range_1cell: no χ-slice with |W| > 1e-12. "
            "Check lensingweights."
        )

    lam_crit = float(np.min(lambda_crit_pos_list))
    bound    = lam_crit * safety_factor
    return -bound, bound
