# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline, UnivariateSpline

# from .RateFunction import (
#     get_psi_2cell,
#     get_psi_derivative_delta1,
#     get_psi_derivative_delta2,
# )


# # --- Fast stencil (Hessian) computations ---
# def det_hessian(psi, dx, dy):
#     # Central differences
#     Dxx = np.array([[0, 0, 0],
#                     [1, -2, 1],
#                     [0, 0, 0]]) / dx**2
#     Dyy = np.array([[0, 1, 0],
#                     [0, -2, 0],
#                     [0, 1, 0]]) / dy**2
#     Dxy = np.array([[1, 0, -1],
#                     [0, 0, 0],
#                     [-1, 0, 1]]) / (4*dx*dy)
#     psi_xx = fftconvolve(psi, Dxx, mode="same")
#     psi_yy = fftconvolve(psi, Dyy, mode="same")
#     psi_xy = fftconvolve(psi, Dxy, mode="same")
#     return psi_xx * psi_yy - psi_xy * psi_xy

# # --- Main fast grid/contour/critical-point pipeline ---
# def find_critical_points_single(
#     variables, variance, lw, z, chi_value,
#     ngrid=100, deld=1e-5, contour_level=0.0, plot=True
# ):
#     d1_vals = np.linspace(-0.99, 1.99, ngrid)
#     d2_vals = np.linspace(-0.99, 1.99, ngrid)
#     D1, D2 = np.meshgrid(d1_vals, d2_vals, indexing='ij')
#     dx = d1_vals[1] - d1_vals[0]
#     dy = d2_vals[1] - d2_vals[0]

#     # Compute ψ on full grid
#     psi = get_psi_2cell(
#         variance, chi_value, variables.recal_value, z, D1, D2,
#         variables.theta1_radian, variables.theta2_radian
#     )

#     # Compute determinant of Hessian
#     detH = det_hessian(psi, dx, dy)

#     # Extract contours of det(H)=0
#     paths = find_contours(detH, contour_level)
#     crit_pts = []
#     for path in paths:
#         # Interpolate indices to (d1, d2) space
#         d1_path = np.interp(path[:, 0], np.arange(ngrid), d1_vals)
#         d2_path = np.interp(path[:, 1], np.arange(ngrid), d2_vals)
#         for d1, d2 in zip(d1_path, d2_path):
#             deriv_sum = (
#                 get_psi_derivative_delta1(
#                     deld, variance, chi_value, variables.recal_value, z, d1, d2,
#                     variables.theta1_radian, variables.theta2_radian
#                 )
#                 +
#                 get_psi_derivative_delta2(
#                     deld, variance, chi_value, variables.recal_value, z, d1, d2,
#                     variables.theta1_radian, variables.theta2_radian
#                 )
#             ) * variables.cosmo.h / lw
#             if np.abs(deriv_sum) < 1e-2:  # For demo; use proper root-finding as needed
#                 crit_pts.append((d1, d2))
#     crit_pts = np.array(crit_pts)

#     if plot:
#         plt.figure(figsize=(8, 8))
#         plt.contour(D1, D2, detH, levels=[contour_level], colors='b')
#         if crit_pts.size:
#             plt.scatter(crit_pts[:, 0], crit_pts[:, 1], c='r', s=10, label='critical pts')
#         plt.xlabel(r'$\delta_1$')
#         plt.ylabel(r'$\delta_2$')
#         plt.legend()
#         plt.title("Critical points: all physical parameters injected")
#         plt.show()
#     return crit_pts


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from scipy.ndimage import convolve, gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Optional: parallelization; code runs without joblib too
# at module top
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    Parallel = None
    delayed = None
    _HAVE_JOBLIB = False


from .RateFunction import (
    get_psi_2cell,
    get_psi_derivative_delta1,
    get_psi_derivative_delta2,
)


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

    # ---------- core numeric helpers ----------

    def _psi_grid(self, variance, chi_value, z):
        """
        Evaluate ψ(d1,d2) on the whole grid in one go if broadcasting is supported;
        otherwise fall back to a fast loop.
        """
        try:
            # Many NumPy-aware functions will broadcast over arrays directly
            psi = get_psi_2cell(
                variance, chi_value, self.recal_value, z,
                self.D1, self.D2, self.theta1, self.theta2
            )
            psi = np.asarray(psi, dtype=float)
            if psi.shape != self.D1.shape:
                raise ValueError("Broadcasted shape mismatch for get_psi_2cell.")
            return psi
        except Exception:
            # Fallback loop (still in C via ndindex + direct writes; faster than np.vectorize)
            psi = np.empty_like(self.D1, dtype=float)
            it = np.ndindex(self.D1.shape)
            for i, j in it:
                psi[i, j] = float(get_psi_2cell(
                    variance, chi_value, self.recal_value, z,
                    self.D1[i, j], self.D2[i, j], self.theta1, self.theta2
                ))
            return psi

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
        return (dpsi_d1 + dpsi_d2) * 1 / lw  #self.h / lw                                   ##################### removed h

    def _roots_along_path(
        self,
        path_xy,
        variance,
        lw,
        z,
        chi_value,
        n_samples=256,
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
            ) * 1 / lw   #self.h / lw                               ##################### removed h                         
            return float(val)

        # sample g
        gvals = np.empty_like(ts, dtype=float)
        for k in range(ts.size):
            gvals[k] = g_at_t(ts[k])

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
    parallel=True,
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


