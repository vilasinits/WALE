# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.interpolate import CubicSpline, UnivariateSpline

# # from .RateFunction import (
# #     get_psi_2cell,
# #     get_psi_derivative_delta1,
# #     get_psi_derivative_delta2,
# # )


# # class CriticalPointsFinder:
# #     r"""
# #     A class designed to identify critical points where the rate function's convexity changes in a cosmological context.
# #     This is achieved through analyzing the Hessian matrix of the rate function across a grid of values,
# #     identifying zero crossings in its determinant to locate changes in convexity.

# #     The rate function :math:`I(x)` characterizes the exponential decay rate of the probabilities of certain outcomes
# #     as the system size increases. The rate function is required to be convex, which ensures that the study of rare
# #     events through large deviation principles can be approached effectively through minimization techniques.

# #     **Cumulant Generating Function and Legendre-Fenchel Transform**
# #     The CGF, denoted by :math:`\Lambda(\theta)`, is foundational for deriving the rate function through the Legendre-Fenchel transform.
# #     This transform connects the CGF and the rate function as follows:

# #     .. math::

# #         I(x) = \sup_{\theta} \{ \theta x - \Lambda(\theta) \}

# #     This equation ensures that the rate function :math:`I(x)` is convex, inheriting this property from the convex CGF :math:`\Lambda(\theta)`.
# #     The supremum operation over :math:`\theta` highlights that :math:`I(x)` represents the tightest upper bound
# #     of the linear functions defined by :math:`\theta x - \Lambda(\theta)`.

# #     **Convexity of the Rate Function**
# #     The convexity of the rate function :math:`I(x)` implies the following inequality for any two points :math:`x_1` and :math:`x_2`
# #     in its domain and any :math:`\lambda \in [0, 1]`:

# #     .. math::

# #         I(\lambda x_1 + (1 - \lambda)x_2) \leq \lambda I(x_1) + (1 - \lambda) I(x_2)

# #     This inequality defines the convexity of the rate function, critical for analyzing rare events in large deviation theory.

# #     In this method, we use the determinant of the Hessian of the rate function to locate points where it vanishes.
# #     These points help identify the values of :math:`\lambda` used in our subsequent calculations.
# #     """

# #     def __init__(
# #         self,
# #         variables,
# #         lw,
# #         z,
# #         chis,
# #         ngrid=50,
# #         plot=False,
# #     ):
# #         """
# #         Initializes the CriticalPointsFinder with cosmology and variance objects,
# #         and optionally configures plotting.

# #         Parameters:
# #             variables (VariablesGenerator): An instance containing all necessary cosmological parameters and variables.
# #             ngrid (int): The number of grid points to use for delta value calculations.
# #             plot (bool): Flag to enable plotting of critical points.
# #         """
# #         self.variables = variables
# #         self.plot = plot
# #         print(
# #             f"       Setting ngrid = {ngrid}. Increase this for more accuracy, but note that computation becomes slower!"
# #         )
# #         self.delta1_vals = np.linspace(-0.99, 1.99, ngrid)
# #         self.delta2_vals = np.linspace(-0.99, 1.99, ngrid)
# #         self.D1, self.D2 = np.meshgrid(
# #             self.delta1_vals, self.delta2_vals, indexing="ij"
# #         )
# #         self.lw = lw
# #         self.z = z
# #         self.chis = chis

# #     def get_hessian(self, x):
# #         """Calculates the Hessian matrix of a function."""
# #         x_grad = np.gradient(x)
# #         hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
# #         for k, grad_k in enumerate(x_grad):
# #             tmp_grad = np.gradient(grad_k)
# #             for l, grad_kl in enumerate(tmp_grad):
# #                 hessian[k, l, :, :] = grad_kl
# #         return hessian

# #     def find_zero_crossing_point(
# #         self, x1, y1, x2, y2, determinant_value1, determinant_value2
# #     ):
# #         """Finds the zero crossing point between two points based on the determinant values."""
# #         t = abs(determinant_value1) / (
# #             abs(determinant_value1) + abs(determinant_value2)
# #         )
# #         zero_crossing_x = x1 + t * (x2 - x1)
# #         zero_crossing_y = y1 + t * (y2 - y1)
# #         return zero_crossing_x, zero_crossing_y

# #     def find_zero_crossings(self, determinant):
# #         """Identifies zero crossings in the determinant grid."""
# #         zero_crossings = []
# #         for i in range(determinant.shape[0] - 1):
# #             for j in range(determinant.shape[1] - 1):
# #                 if determinant[i, j] * determinant[i, j + 1] <= 0:
# #                     newx, newy = self.find_zero_crossing_point(
# #                         self.D1[i, j],
# #                         self.D2[i, j],
# #                         self.D1[i, j + 1],
# #                         self.D2[i, j + 1],
# #                         determinant[i, j],
# #                         determinant[i, j + 1],
# #                     )
# #                     zero_crossings.append((newx, newy))
# #                 if determinant[i, j] * determinant[i + 1, j] <= 0:
# #                     newx, newy = self.find_zero_crossing_point(
# #                         self.D1[i, j],
# #                         self.D2[i, j],
# #                         self.D1[i + 1, j],
# #                         self.D2[i + 1, j],
# #                         determinant[i, j],
# #                         determinant[i + 1, j],
# #                     )
# #                     zero_crossings.append((newx, newy))
# #         return zero_crossings

# #     def get_critical_points(self, variance, lw, z, chi_value):
# #         """Calculates critical points for the given redshift z and plots them if requested."""
# #         recal_value = self.variables.recal_value
# #         theta1 = self.variables.theta1_radian
# #         theta2 = self.variables.theta2_radian

# #         deld = 1e-8
# #         rate_function = np.vectorize(
# #             lambda d1, d2: get_psi_2cell(
# #                 variance,
# #                 chi_value,
# #                 recal_value,
# #                 z,
# #                 d1,
# #                 d2,
# #                 theta1,
# #                 theta2,
# #             )
# #         )(self.D1, self.D2)

# #         hessian = np.array(self.get_hessian(rate_function))
# #         determinants = np.array(
# #             (hessian[0, 0, :, :] * hessian[1, 1, :, :])
# #             - (hessian[0, 1, :, :] * hessian[1, 0, :, :])
# #         )
# #         zero_crossings = np.array(self.find_zero_crossings(determinants))
# #         drf1, drf2 = [], []
# #         for x, y in zero_crossings:
# #             drf1.append(
# #                 get_psi_derivative_delta1(
# #                     deld,
# #                     variance,
# #                     chi_value,
# #                     recal_value,
# #                     z,
# #                     x,
# #                     y,
# #                     theta1,
# #                     theta2,
# #                 )
# #                 * self.variables.cosmo.h
# #                 / lw
# #             )
# #             drf2.append(
# #                 get_psi_derivative_delta2(
# #                     deld,
# #                     variance,
# #                     chi_value,
# #                     recal_value,
# #                     z,
# #                     x,
# #                     y,
# #                     theta1,
# #                     theta2,
# #                 )
# #                 * self.variables.cosmo.h
# #                 / lw
# #             )
# #         drf1, drf2 = np.array(drf1), np.array(drf2)

# #         sorted_indices = np.argsort(drf1[:, 0])
# #         # Sort drf1 and drf2 using the sorted indices
# #         sorted_drf1 = drf1[sorted_indices, 0]
# #         sorted_drf2 = drf2[sorted_indices, 0]

# #         drf_spline = CubicSpline(sorted_drf1[:], sorted_drf2[:])
# #         drf1_new = np.linspace(-1000, 3000, 200)
# #         drf2_new = drf_spline(drf1_new)
# #         sum_derivatives = drf1_new + drf2_new
# #         # Fit spline to the sum of derivatives
# #         spline1 = UnivariateSpline(drf1_new, sum_derivatives, s=0)
# #         sorted_indices = np.argsort(drf2_new)
# #         # Find the value of x where the spline is 0
# #         critical_points1 = spline1.roots()

# #         # print(
# #         #     "The approximate critical points at redshift z: ",
# #         #     z,
# #         #     " are: ",
# #         #     -critical_points1,
# #         # )
# #         if self.plot:
# #             plt.plot(drf1_new, sum_derivatives, label=z)
# #             plt.scatter(critical_points1, spline1(critical_points1), color="r")
# #             plt.xlim(-1000, 2000)
# #             plt.ylim(-1000, 2000)
# #             plt.grid(visible=True, which="both", axis="both")
# #             plt.legend()
# #         return [-x for x in critical_points1]


# # def find_smallest_pair(critical_values):
# #     """
# #     Finds the pair of points with the smallest Euclidean distance between them from a set of critical values.

# #     Parameters:
# #         critical_values (numpy.ndarray): An array of critical points.

# #     Returns:
# #         tuple: The pair of points with the smallest distance and their Euclidean distance.
# #     """
# #     num_points = critical_values.shape[0]
# #     if num_points < 2:
# #         return None, float("inf")  # No pair exists

# #     smallest_distance = float("inf")
# #     smallest_pair = None

# #     for i in range(num_points - 1):
# #         for j in range(i + 1, num_points):
# #             distance = np.linalg.norm(critical_values[i] - critical_values[j])
# #             if distance < smallest_distance:
# #                 smallest_distance = distance
# #                 smallest_pair = (critical_values[i], critical_values[j])

# #     return smallest_pair


# # def find_critical_points_for_cosmo(
# #     variables, variance, ngrid_critical=90, plot=False, min_z=1, max_z=4
# # ):
# #     """
# #     Finds critical points in the lensing potential based on the provided variables.

# #     Args:
# #         variables: Object containing necessary cosmological variables (e.g., lensingweights, redshifts, chis).
# #         variance: Variance or smoothing parameter needed for critical point computation.
# #         ngrid_critical (int, optional): Grid resolution for critical point search. Defaults to 90.
# #         plot (bool, optional): Whether to plot the results. Defaults to False.
# #         max_nz (int, optional): Maximum number of redshift slices to use. Defaults to 4.

# #     Returns:
# #         tuple: Smallest positive and largest negative critical point values.
# #     """
# #     print("   Finding critical points...")

# #     criticalpoints = CriticalPointsFinder(
# #         variables,
# #         ngrid=ngrid_critical,
# #         lw=variables.lensingweights[min_z:max_z],
# #         z=variables.redshifts[min_z:max_z],
# #         chis=variables.chis,
# #         plot=plot,
# #     )

# #     critical_values_list = []

# #     for i, z_crit in enumerate(criticalpoints.z):
# #         crit_vals = criticalpoints.get_critical_points(
# #             variance,
# #             lw=criticalpoints.lw[i],
# #             z=z_crit,
# #             chi_value=criticalpoints.chis[i],
# #         )
# #         if crit_vals is not None and len(crit_vals) >= 2:
# #             critical_values_list.append(crit_vals[:2])

# #     if not critical_values_list:
# #         print("  Warning: No critical points found in the specified redshift range.")
# #         return None, None

# #     # Flatten values
# #     flat_values = []
# #     for item in critical_values_list:
# #         if isinstance(item, (np.ndarray, list)):
# #             flat_values.extend(np.ravel(item))
# #         else:
# #             flat_values.append(item)

# #     flat_values = np.array(flat_values)
# #     flat_values = flat_values[~np.isnan(flat_values)]

# #     # Compute smallest positive and largest negative values
# #     positive_values = flat_values[flat_values > 0]
# #     negative_values = flat_values[flat_values < 0]

# #     smallest_positive = np.min(positive_values) if positive_values.size > 0 else None
# #     largest_negative = np.max(negative_values) if negative_values.size > 0 else None

# #     print(
# #         "       Smallest distance pair of critical points:",
# #         smallest_positive,
# #         largest_negative,
# #     )

# #     return smallest_positive, largest_negative


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import convolve, gaussian_filter
# from scipy.interpolate import interp1d
# from scipy.optimize import brentq

# # Optional: parallelization; code runs without joblib too
# try:
#     from joblib import Parallel, delayed
#     _HAVE_JOBLIB = True
# except Exception:
#     _HAVE_JOBLIB = True

# from .RateFunction import (
#     get_psi_2cell,
#     get_psi_derivative_delta1,
#     get_psi_derivative_delta2,
# )


# class CriticalPointsFinder:
#     r"""
#     Fast critical-points finder using:
#       1) Convolution stencils for Hessian determinant of ψ on a grid.
#       2) Marching-squares (matplotlib.contour) to get det(Hψ)=0 curve(s).
#       3) Root-finding along those curves for g(d1,d2) = dψ/dd1 + dψ/dd2.

#     Notes
#     -----
#     - This is faster and more reliable than uniform fine grids + scattered splines.
#     - It evaluates ψ(d1,d2) once per z on a coarse/moderate grid, then refines *only
#       where needed* (along the zero-determinant contour).
#     """

#     def __init__(self, variables, lw, z, chis, ngrid=50, plot=False,
#                  smooth_sigma=0.0,  # small smoothing of det(H) to tame jagged edges (0 = off)
#                  ):
#         self.variables = variables
#         self.plot = plot
#         self.smooth_sigma = float(smooth_sigma)

#         print(f"       Setting ngrid = {ngrid}. Increase for accuracy; runtime ~ O(ngrid^2).")
#         self.delta1_vals = np.linspace(-0.99, 1.99, ngrid)
#         self.delta2_vals = np.linspace(-0.99, 1.99, ngrid)
#         self.D1, self.D2 = np.meshgrid(self.delta1_vals, self.delta2_vals, indexing="ij")

#         self.dx = float(self.delta1_vals[1] - self.delta1_vals[0])
#         self.dy = float(self.delta2_vals[1] - self.delta2_vals[0])

#         self.lw = np.asarray(lw)
#         self.z = np.asarray(z)
#         self.chis = np.asarray(chis)

#         # sanity alignment
#         if not (len(self.lw) == len(self.z) == len(self.chis)):
#             raise ValueError("lw, z, and chis must have the same length (aligned slices).")

#         # prefetch constants
#         self.recal_value = self.variables.recal_value
#         self.theta1 = self.variables.theta1_radian
#         self.theta2 = self.variables.theta2_radian
#         self.h = self.variables.cosmo.h

#         # 2nd-derivative central-difference stencils
#         self._Dxx = np.array([[0, 0, 0],
#                               [1,-2, 1],
#                               [0, 0, 0]], dtype=float) / (self.dx*self.dx)
#         self._Dyy = np.array([[0, 1, 0],
#                               [0,-2, 0],
#                               [0, 1, 0]], dtype=float) / (self.dy*self.dy)
#         # mixed derivative (∂²/∂x∂y)
#         self._Dxy = np.array([[ 1, 0,-1],
#                               [ 0, 0, 0],
#                               [-1, 0, 1]], dtype=float) / (4*self.dx*self.dy)

#     # ---------- core numeric helpers ----------

#     def _psi_grid(self, variance, chi_value, z):
#         """
#         Evaluate ψ(d1,d2) on the whole grid in one go if broadcasting is supported;
#         otherwise fall back to a fast loop.
#         """
#         try:
#             # Many NumPy-aware functions will broadcast over arrays directly
#             psi = get_psi_2cell(
#                 variance, chi_value, self.recal_value, z,
#                 self.D1, self.D2, self.theta1, self.theta2
#             )
#             psi = np.asarray(psi, dtype=float)
#             if psi.shape != self.D1.shape:
#                 raise ValueError("Broadcasted shape mismatch for get_psi_2cell.")
#             return psi
#         except Exception:
#             # Fallback loop (still in C via ndindex + direct writes; faster than np.vectorize)
#             psi = np.empty_like(self.D1, dtype=float)
#             it = np.ndindex(self.D1.shape)
#             for i, j in it:
#                 psi[i, j] = float(get_psi_2cell(
#                     variance, chi_value, self.recal_value, z,
#                     self.D1[i, j], self.D2[i, j], self.theta1, self.theta2
#                 ))
#             return psi

#     def _det_hessian(self, psi):
#         """Compute det(Hψ) via convolution stencils."""
#         psi_xx = convolve(psi, self._Dxx, mode="nearest")
#         psi_yy = convolve(psi, self._Dyy, mode="nearest")
#         psi_xy = convolve(psi, self._Dxy, mode="nearest")
#         detH = psi_xx*psi_yy - psi_xy*psi_xy
#         if self.smooth_sigma > 0:
#             detH = gaussian_filter(detH, sigma=self.smooth_sigma, mode="nearest")
#         return detH

#     def _zero_contours(self, detH):
#         """
#         Extract zero-level contours from detH using matplotlib's marching squares
#         without leaving a figure behind.
#         Returns a list of arrays of shape (k, 2) with columns [d1, d2].
#         """
#         fig, ax = plt.subplots()
#         try:
#             CS = ax.contour(self.delta1_vals, self.delta2_vals, detH.T, levels=[0.0])
#             paths = []
#             if CS.collections and CS.collections[0].get_paths():
#                 for p in CS.collections[0].get_paths():
#                     v = p.vertices  # (n, 2) as [[x(d1), y(d2)], ...]
#                     paths.append(v.copy())
#             return paths
#         finally:
#             plt.close(fig)

#     def _g(self, d1, d2, variance, lw, z, chi_value, deld=1e-6):
#         dpsi_d1 = get_psi_derivative_delta1(
#             deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
#         )
#         dpsi_d2 = get_psi_derivative_delta2(
#             deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
#         )
#         return (dpsi_d1 + dpsi_d2) * self.h / lw

#     def _roots_along_path(
#         self,
#         path_xy,
#         variance,
#         lw,
#         z,
#         chi_value,
#         n_samples=256,
#         flip_sign=True,
#         return_mode="derivative",
#         deld=1e-6,
#     ):
#         """
#         Find roots of g(d1,d2)=0 restricted to the given det(Hψ)=0 contour 'path_xy' (n×2).
#         Parameterize by arclength, sample g, bracket sign changes, refine with brentq.
#         Returns list of critical values; by default returns -(dψ/dδ1)*h/lw to match original code.
#         """
#         if path_xy is None or len(path_xy) < 2:
#             return []

#         x = np.asarray(path_xy[:, 0], dtype=float)
#         y = np.asarray(path_xy[:, 1], dtype=float)

#         # arclength parameterization
#         ds = np.hypot(np.diff(x), np.diff(y))
#         s = np.concatenate(([0.0], np.cumsum(ds)))
#         L = float(s[-1])
#         if not np.isfinite(L) or L <= 0.0:
#             return []

#         fx = interp1d(s, x, kind="linear", assume_sorted=True)
#         fy = interp1d(s, y, kind="linear", assume_sorted=True)

#         # define the sampling grid *before* using it
#         n_samples = int(max(16, n_samples))
#         ts = np.linspace(0.0, L, n_samples)

#         # helper: g along contour
#         def g_at_t(tt):
#             d1 = float(fx(tt))
#             d2 = float(fy(tt))
#             return self._g(d1, d2, variance, lw, z, chi_value, deld=deld)

#         # helper: scaled dψ/dδ1 along contour (to match your original return scale)
#         def d1_deriv_at_t(tt):
#             d1 = float(fx(tt))
#             d2 = float(fy(tt))
#             val = get_psi_derivative_delta1(
#                 deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
#             ) * self.h / lw
#             return float(val)

#         # sample g
#         gvals = np.empty_like(ts, dtype=float)
#         for k in range(ts.size):
#             gvals[k] = g_at_t(ts[k])

#         crits = []
#         for k in range(ts.size - 1):
#             g1, g2 = gvals[k], gvals[k + 1]
#             if not (np.isfinite(g1) and np.isfinite(g2)):
#                 continue

#             if g1 == 0.0:
#                 t0 = ts[k]
#                 val = -d1_deriv_at_t(t0) if return_mode == "derivative" else -float(fx(t0))
#                 crits.append(val)
#                 continue

#             if g1 * g2 < 0.0:  # sign change -> refine with brentq
#                 try:
#                     root_t = brentq(lambda tt: g_at_t(tt), ts[k], ts[k + 1], maxiter=200)
#                     val = -d1_deriv_at_t(root_t) if return_mode == "derivative" else -float(fx(root_t))
#                     crits.append(val)
#                 except ValueError:
#                     # bracketing failed due to numerical issues; skip this interval
#                     pass

#         return crits


#     # ---------- public API ----------

#     def get_critical_points(self, variance, lw, z, chi_value, target_levels=1,
#                         return_mode="derivative"):
#         """
#         Compute critical points for one (lw, z, chi_value) triple.
#         Returns a Python list of critical values (can be empty).
#         """
#         # 1) ψ grid
#         psi = self._psi_grid(variance, chi_value, z)

#         # 2) det(Hψ)
#         detH = self._det_hessian(psi)

#         # 3) Zero contours
#         paths = self._zero_contours(detH)
#         if len(paths) == 0:
#             if self.plot:
#                 print(f"   z={z:.3f}: no det(Hψ)=0 contour found.")
#             return []

#         # Use the longest contours first (often the main, physically relevant one)
#         paths.sort(key=lambda v: 0.0 if len(v) < 2 else np.sum(np.hypot(np.diff(v[:,0]), np.diff(v[:,1]))),
#                    reverse=True)
#         if target_levels is not None and target_levels > 0:
#             paths = paths[:target_levels]

#         # 4) Find roots of g along each contour
#         all_crits = []
#         for P in paths:
#             crits = self._roots_along_path(P, variance, lw, z, chi_value,
#                                         return_mode=return_mode)
#             all_crits.extend(crits)
#         return all_crits

#         # Optional quickplot (g vs arclength would need extra plumbing; here we skip)
#         # return all_crits


# def find_smallest_pair(critical_values):
#     """
#     From an array-like of critical points, finds the pair with the smallest Euclidean distance.
#     Here critical_values is typically 1-D; this function supports 1-D or 2-D points.
#     """
#     cv = np.asarray(critical_values)
#     if cv.size == 0:
#         return None

#     if cv.ndim == 1:
#         # 1-D case: pair with smallest absolute difference
#         cv_sorted = np.sort(cv)
#         if cv_sorted.size < 2:
#             return None
#         diffs = np.diff(cv_sorted)
#         k = np.argmin(np.abs(diffs))
#         return (cv_sorted[k], cv_sorted[k+1])

#     # 2-D points case
#     num_points = cv.shape[0]
#     if num_points < 2:
#         return None

#     best_pair = None
#     best_dist = np.inf
#     for i in range(num_points - 1):
#         for j in range(i + 1, num_points):
#             dist = np.linalg.norm(cv[i] - cv[j])
#             if dist < best_dist:
#                 best_dist = dist
#                 best_pair = (cv[i], cv[j])
#     return best_pair


# def _as_scalar(x):
#     x = np.asarray(x)
#     return float(np.mean(x)) if x.ndim > 0 else float(x)

# def _process_one_z(i, z_i, lw_i, chi_i, finder, variance, return_mode):
#     lw_s = _as_scalar(lw_i)  # ensure scalar lensing weight per z
#     crits = finder.get_critical_points(variance, lw=lw_s, z=z_i, chi_value=chi_i,
#                                        return_mode=return_mode)
#     if len(crits) == 0:
#         return []
#     crits = np.asarray(crits, dtype=float)
#     crits = crits[np.isfinite(crits)]
#     return crits.tolist()


# def find_critical_points_for_cosmo(
#     variables,
#     variance,
#     ngrid_critical=90,
#     plot=False,
#     min_z=1,
#     max_z=4,
#     smooth_sigma=0.0,
#     parallel=True,
#     n_jobs=-1,
#     return_mode="derivative", 
# ):
#     """
#     High-level driver that returns (smallest_positive, largest_negative)
#     aggregated across z in [min_z:max_z].

#     Parameters
#     ----------
#     variables : object
#         Must provide attributes:
#           - lensingweights, redshifts, chis (indexable, aligned),
#           - recal_value, theta1_radian, theta2_radian, cosmo.h
#     variance : any
#         Passed through to ψ and derivative functions.
#     ngrid_critical : int
#         Grid resolution (per axis). 60–120 is often a good range.
#     plot : bool
#         Keep for API compatibility; plotting of g-curves is not produced here.
#     min_z, max_z : int
#         Slice range (Python-style: start inclusive, end exclusive).
#     smooth_sigma : float
#         Optional Gaussian smoothing (in grid pixels) applied to det(Hψ) before contouring.
#     parallel : bool
#         If True and joblib is available, compute slices in parallel.
#     n_jobs : int
#         Joblib workers (ignored if parallel=False or joblib not installed).

#     Returns
#     -------
#     (smallest_positive, largest_negative)
#     """
#     print("   Finding critical points (optimized)...")

#     # Align slices for lw, z, chis (IMPORTANT!)
#     lw_slice = variables.lensingweights[min_z:max_z]
#     z_slice = variables.redshifts[min_z:max_z]
#     chi_slice = variables.chis[min_z:max_z]

#     finder = CriticalPointsFinder(
#         variables,
#         lw=lw_slice,
#         z=z_slice,
#         chis=chi_slice,
#         ngrid=ngrid_critical,
#         plot=plot,
#         smooth_sigma=smooth_sigma,
#     )

#     # Per-z processing
#     if parallel and _HAVE_JOBLIB:
#         results = Parallel(n_jobs=n_jobs, prefer="threads")(
#             delayed(_process_one_z)(i, z_i, lw_i, chi_i, finder, variance, return_mode)
#             for i, (z_i, lw_i, chi_i) in enumerate(zip(finder.z, finder.lw, finder.chis))
#         )
#     else:
#         results = [
#             _process_one_z(i, z_i, lw_i, chi_i, finder, variance, return_mode)
#             for i, (z_i, lw_i, chi_i) in enumerate(zip(finder.z, finder.lw, finder.chis))
#         ]

#     # Flatten and clean
#     flat = np.array([c for sub in results for c in sub], dtype=float)
#     flat = flat[np.isfinite(flat)]
#     if flat.size == 0:
#         print("  Warning: No critical points found in the specified redshift range.")
#         return None, None

#     positive = flat[flat > 0]
#     negative = flat[flat < 0]

#     smallest_positive = np.min(positive) if positive.size else None
#     largest_negative = np.max(negative) if negative.size else None

#     print("       Smallest positive / largest negative:",
#           smallest_positive, "/", largest_negative)
#     return smallest_positive, largest_negative




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import convolve, gaussian_filter
# from scipy.optimize import brentq

# # Optional: parallelization; code runs without joblib too
# try:
#     from joblib import Parallel, delayed
#     _HAVE_JOBLIB = True
# except Exception:
#     _HAVE_JOBLIB = False

# from .RateFunction import (
#     get_psi_2cell,
#     get_psi_derivative_delta1,
#     get_psi_derivative_delta2,
# )


# class CriticalPointsFinder:
#     r"""
#     Fast critical-points finder using:
#       1) Convolution stencils for Hessian determinant of ψ on a coarse grid.
#       2) Lightweight marching-squares per cell to get local det(Hψ)=0 segments.
#       3) 1-D root-finding along each segment for g(d1,d2)=∂ψ/∂δ1+∂ψ/∂δ2.

#     Works well with ngrid ~ 17–25.
#     """

#     def __init__(self, variables, lw, z, chis, ngrid=25, plot=False,
#                  smooth_sigma=0.0,  # Gaussian smoothing (in grid pixels) on det(Hψ) before contouring
#                  ):
#         self.variables = variables
#         self.plot = plot
#         self.smooth_sigma = float(smooth_sigma)

#         print(f"       Setting ngrid = {ngrid}. Runtime ~ O(ngrid^2); 17–25 is usually enough.")
#         self.delta1_vals = np.linspace(-0.99, 1.99, ngrid)
#         self.delta2_vals = np.linspace(-0.99, 1.99, ngrid)
#         self.D1, self.D2 = np.meshgrid(self.delta1_vals, self.delta2_vals, indexing="ij")

#         self.dx = float(self.delta1_vals[1] - self.delta1_vals[0])
#         self.dy = float(self.delta2_vals[1] - self.delta2_vals[0])

#         self.lw = np.asarray(lw)
#         self.z = np.asarray(z)
#         self.chis = np.asarray(chis)

#         if not (len(self.lw) == len(self.z) == len(self.chis)):
#             raise ValueError("lw, z, and chis must have the same length (aligned slices).")

#         # constants
#         self.recal_value = self.variables.recal_value
#         self.theta1 = self.variables.theta1_radian
#         self.theta2 = self.variables.theta2_radian
#         self.h = self.variables.cosmo.h

#         # 2nd-derivative central-difference stencils
#         self._Dxx = np.array([[0, 0, 0],
#                               [1,-2, 1],
#                               [0, 0, 0]], dtype=float) / (self.dx*self.dx)
#         self._Dyy = np.array([[0, 1, 0],
#                               [0,-2, 0],
#                               [0, 1, 0]], dtype=float) / (self.dy*self.dy)
#         self._Dxy = np.array([[ 1, 0,-1],
#                               [ 0, 0, 0],
#                               [-1, 0, 1]], dtype=float) / (4*self.dx*self.dy)

#     # ---------- core numeric helpers ----------

#     def _psi_grid(self, variance, chi_value, z):
#         """Evaluate ψ on the full grid with broadcasting if supported."""
#         try:
#             psi = get_psi_2cell(
#                 variance, chi_value, self.recal_value, z,
#                 self.D1, self.D2, self.theta1, self.theta2
#             )
#             psi = np.asarray(psi, dtype=float)
#             if psi.shape != self.D1.shape:
#                 raise ValueError("Broadcasted shape mismatch for get_psi_2cell.")
#             return psi
#         except Exception:
#             psi = np.empty_like(self.D1, dtype=float)
#             it = np.ndindex(self.D1.shape)
#             for i, j in it:
#                 psi[i, j] = float(get_psi_2cell(
#                     variance, chi_value, self.recal_value, z,
#                     self.D1[i, j], self.D2[i, j], self.theta1, self.theta2
#                 ))
#             return psi

#     def _det_hessian(self, psi):
#         """Compute det(Hψ) via convolution stencils."""
#         psi_xx = convolve(psi, self._Dxx, mode="nearest")
#         psi_yy = convolve(psi, self._Dyy, mode="nearest")
#         psi_xy = convolve(psi, self._Dxy, mode="nearest")
#         detH = psi_xx*psi_yy - psi_xy*psi_xy
#         if self.smooth_sigma > 0:
#             detH = gaussian_filter(detH, sigma=self.smooth_sigma, mode="nearest")
#         return detH

#     # ---- marching-squares inside each sign-change cell (no matplotlib) ----

#     def _edge_cross(self, x0, y0, x1, y1, v0, v1):
#         """Linear interpolate zero crossing on an edge (v0*v1<0 assumed)."""
#         t = v0 / (v0 - v1)
#         return (x0 + t*(x1 - x0), y0 + t*(y1 - y0))

#     def _zero_segments(self, detH):
#         """
#         Return list of line segments approximating detH=0 inside cells.
#         Each segment is ((x1,y1),(x2,y2)).
#         Handles ambiguous cases (5, 10) by splitting into two segments.
#         """
#         segs = []
#         X = self.delta1_vals
#         Y = self.delta2_vals
#         nx, ny = detH.shape  # (ngrid, ngrid)

#         for i in range(nx - 1):
#             xL, xR = X[i], X[i+1]
#             for j in range(ny - 1):
#                 yB, yT = Y[j], Y[j+1]
#                 v00 = detH[i, j]
#                 v10 = detH[i+1, j]
#                 v01 = detH[i, j+1]
#                 v11 = detH[i+1, j+1]

#                 # quick reject if same sign everywhere
#                 if np.min([v00, v10, v01, v11]) > 0 or np.max([v00, v10, v01, v11]) < 0:
#                     continue

#                 # Collect edge intersections (at most 4)
#                 pts = []  # (x,y) + edge id
#                 # bottom edge (v00->v10)
#                 if v00 * v10 < 0:
#                     pts.append((*self._edge_cross(xL, yB, xR, yB, v00, v10), 'bottom'))
#                 # right edge (v10->v11)
#                 if v10 * v11 < 0:
#                     pts.append((*self._edge_cross(xR, yB, xR, yT, v10, v11), 'right'))
#                 # top edge (v01->v11)
#                 if v01 * v11 < 0:
#                     pts.append((*self._edge_cross(xL, yT, xR, yT, v01, v11), 'top'))
#                 # left edge (v00->v01)
#                 if v00 * v01 < 0:
#                     pts.append((*self._edge_cross(xL, yB, xL, yT, v00, v01), 'left'))

#                 if len(pts) == 2:
#                     segs.append(((pts[0][0], pts[0][1]), (pts[1][0], pts[1][1])))
#                 elif len(pts) == 4:
#                     # ambiguous: cases 5 and 10; connect (bottom-left) & (right-top)
#                     # Build dict by edge
#                     ed = {p[2]: (p[0], p[1]) for p in pts}
#                     # pairings: (bottom,left) and (right,top)
#                     if all(k in ed for k in ('bottom', 'left', 'right', 'top')):
#                         segs.append((ed['bottom'], ed['left']))
#                         segs.append((ed['right'], ed['top']))
#                 # else: 0 or odd number (rare due to equal zeros) → ignore

#         return segs

#     # ---------- g and segment root ----------

#     def _g(self, d1, d2, variance, lw, z, chi_value, deld=1e-6):
#         dpsi_d1 = get_psi_derivative_delta1(
#             deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
#         )
#         dpsi_d2 = get_psi_derivative_delta2(
#             deld, variance, chi_value, self.recal_value, z, d1, d2, self.theta1, self.theta2
#         )
#         return (dpsi_d1 + dpsi_d2) * self.h / lw

#     def _root_on_segment(self, p0, p1, variance, lw, z, chi_value,
#                          samples=9, return_mode="derivative", deld=1e-6):
#         """
#         Bracket zero(s) of g along a segment p(t)=p0+t*(p1-p0), t in [0,1], then refine with brentq.
#         Returns a (possibly empty) list of critical values in the requested return_mode.
#         """
#         x0, y0 = p0
#         x1, y1 = p1
#         def xy(t):
#             return (x0 + t*(x1 - x0), y0 + t*(y1 - y0))

#         ts = np.linspace(0.0, 1.0, samples)
#         gvals = np.empty_like(ts)
#         for k, t in enumerate(ts):
#             xx, yy = xy(t)
#             gvals[k] = self._g(xx, yy, variance, lw, z, chi_value, deld=deld)

#         crits = []
#         for k in range(len(ts) - 1):
#             g1, g2 = gvals[k], gvals[k+1]
#             if not (np.isfinite(g1) and np.isfinite(g2)):
#                 continue
#             if g1 == 0.0:
#                 t0 = ts[k]
#                 xz, yz = xy(t0)
#                 if return_mode == "derivative":
#                     val = -get_psi_derivative_delta1(
#                         deld, variance, chi_value, self.recal_value, z, xz, yz, self.theta1, self.theta2
#                     ) * self.h / lw
#                 else:
#                     val = -xz
#                 crits.append(float(val))
#                 continue
#             if g1 * g2 < 0.0:
#                 try:
#                     root_t = brentq(lambda tt: self._g(*xy(tt), variance, lw, z, chi_value, deld=deld),
#                                     ts[k], ts[k+1], maxiter=200)
#                     xz, yz = xy(root_t)
#                     if return_mode == "derivative":
#                         val = -get_psi_derivative_delta1(
#                             deld, variance, chi_value, self.recal_value, z, xz, yz, self.theta1, self.theta2
#                         ) * self.h / lw
#                     else:
#                         val = -xz
#                     crits.append(float(val))
#                 except ValueError:
#                     pass
#         return crits

#     @staticmethod
#     def _unique_tol(vals, atol=1e-6, rtol=1e-6):
#         """Deduplicate nearly-equal values."""
#         if len(vals) == 0:
#             return []
#         v = np.sort(np.asarray(vals, dtype=float))
#         keep = [v[0]]
#         for x in v[1:]:
#             if not (np.isclose(x, keep[-1], atol=atol, rtol=rtol)):
#                 keep.append(x)
#         return keep

#     # ---------- public API ----------

#     def get_critical_points(self, variance, lw, z, chi_value,
#                             return_mode="derivative"):
#         """
#         Compute critical points for one (lw, z, chi_value) triple.
#         Returns a Python list of critical values (can be empty).
#         """
#         # 1) ψ grid and det(Hψ)
#         psi = self._psi_grid(variance, chi_value, z)
#         detH = self._det_hessian(psi)

#         # 2) local zeroline segments via marching-squares
#         segments = self._zero_segments(detH)
#         if len(segments) == 0:
#             if self.plot:
#                 print(f"   z={z:.3f}: no det(Hψ)=0 segment found.")
#             return []

#         # 3) root(s) of g along each segment
#         all_crits = []
#         for p0, p1 in segments:
#             all_crits.extend(self._root_on_segment(p0, p1, variance, lw, z, chi_value,
#                                                    samples= nine_if(len(segments)),  # see helper below
#                                                    return_mode=return_mode))
#         # 4) deduplicate near-equal roots
#         return self._unique_tol(all_crits, atol=1e-5, rtol=1e-5)


# def nine_if(n):
#     """Fewer segments → sample more densely; many segments → fewer samples."""
#     if n <= 10:
#         return 13
#     if n <= 50:
#         return 9
#     return 7


# def find_smallest_pair(critical_values):
#     """
#     From an array-like of critical points, finds the pair with the smallest absolute spacing.
#     """
#     cv = np.asarray(critical_values, dtype=float)
#     cv = cv[np.isfinite(cv)]
#     if cv.size < 2:
#         return None
#     cv_sorted = np.sort(cv)
#     diffs = np.diff(cv_sorted)
#     k = np.argmin(np.abs(diffs))
#     return (cv_sorted[k], cv_sorted[k+1])


# def _as_scalar(x):
#     x = np.asarray(x)
#     return float(np.mean(x)) if x.ndim > 0 else float(x)


# def _process_one_z(i, z_i, lw_i, chi_i, finder, variance, return_mode):
#     lw_s = _as_scalar(lw_i)  # ensure scalar per z
#     crits = finder.get_critical_points(variance, lw=lw_s, z=z_i, chi_value=chi_i,
#                                        return_mode=return_mode)
#     if len(crits) == 0:
#         return []
#     crits = np.asarray(crits, dtype=float)
#     return crits[np.isfinite(crits)].tolist()


# def find_critical_points_for_cosmo(
#     variables,
#     variance,
#     ngrid_critical=25,
#     plot=False,
#     min_z=1,
#     max_z=4,
#     smooth_sigma=0.0,
#     parallel=True,
#     n_jobs=-1,
#     return_mode="derivative",
# ):
#     """
#     Driver that returns (smallest_positive, largest_negative) aggregated across
#     z in [min_z:max_z]. Works well with ngrid_critical ~ 17–25.
#     """
#     print("   Finding critical points (optimized-msq)...")

#     # Align slices
#     lw_slice = variables.lensingweights[min_z:max_z]
#     z_slice = variables.redshifts[min_z:max_z]
#     chi_slice = variables.chis[min_z:max_z]

#     finder = CriticalPointsFinder(
#         variables,
#         lw=lw_slice,
#         z=z_slice,
#         chis=chi_slice,
#         ngrid=ngrid_critical,
#         plot=plot,
#         smooth_sigma=smooth_sigma,
#     )

#     if parallel and _HAVE_JOBLIB:
#         results = Parallel(n_jobs=n_jobs, prefer="threads")(
#             delayed(_process_one_z)(i, z_i, lw_i, chi_i, finder, variance, return_mode)
#             for i, (z_i, lw_i, chi_i) in enumerate(zip(finder.z, finder.lw, finder.chis))
#         )
#     else:
#         results = [
#             _process_one_z(i, z_i, lw_i, chi_i, finder, variance, return_mode)
#             for i, (z_i, lw_i, chi_i) in enumerate(zip(finder.z, finder.lw, finder.chis))
#         ]

#     flat = np.array([c for sub in results for c in sub], dtype=float)
#     flat = flat[np.isfinite(flat)]
#     if flat.size == 0:
#         print("  Warning: No critical points found in the specified redshift range.")
#         return None, None

#     positive = flat[flat > 0]
#     negative = flat[flat < 0]
#     smallest_positive = np.min(positive) if positive.size else None
#     largest_negative = np.max(negative) if negative.size else None

#     print("       Smallest positive / largest negative:",
#           smallest_positive, "/", largest_negative)
#     return smallest_positive, largest_negative


# #######################################
# ######################################

import numpy as np
from scipy.optimize import brentq

# Your provided primitives
from .RateFunction import (
    get_psi_2cell,
    get_psi_derivative_delta1,
    get_psi_derivative_delta2,
)


# =========================
# Low-level numeric helpers
# =========================

def _psi_fd_hessian(psi_func, d1, d2, h=1e-3):
    """
    3x3 central-difference Hessian of psi at (d1,d2).
    Returns (psi_xx, psi_yy, psi_xy).
    """
    # center
    p00 = psi_func(d1, d2)
    # axial
    p10 = psi_func(d1 + h, d2)
    pm10 = psi_func(d1 - h, d2)
    p01 = psi_func(d1, d2 + h)
    pm01 = psi_func(d1, d2 - h)
    # diagonals
    pp  = psi_func(d1 + h, d2 + h)
    pm  = psi_func(d1 + h, d2 - h)
    mp  = psi_func(d1 - h, d2 + h)
    mm  = psi_func(d1 - h, d2 - h)

    psi_xx = (p10 - 2.0*p00 + pm10) / (h*h)
    psi_yy = (p01 - 2.0*p00 + pm01) / (h*h)
    psi_xy = (pp - pm - mp + mm) / (4.0*h*h)
    return psi_xx, psi_yy, psi_xy


def _det_hessian_psi(psi_func, d1, d2, h_fd=1e-3):
    """det(H_psi) via finite differences at (d1,d2)."""
    xx, yy, xy = _psi_fd_hessian(psi_func, d1, d2, h=h_fd)
    return xx*yy - xy*xy, (xx, yy, xy)


def _g_and_grad(psi_d1_func, psi_d2_func, psi_func, d1, d2, h_fd=1e-3):
    """
    g = psi_1 + psi_2 (unscaled). Also returns grad g using Hessian entries:
    ∂g/∂d1 = psi_11 + psi_12, ∂g/∂d2 = psi_12 + psi_22.
    """
    g = psi_d1_func(d1, d2) + psi_d2_func(d1, d2)
    xx, yy, xy = _psi_fd_hessian(psi_func, d1, d2, h=h_fd)
    dg_d1 = xx + xy
    dg_d2 = xy + yy
    return g, dg_d1, dg_d2


def _newton_project_to_g0(psi_d1_func, psi_d2_func, psi_func, d1, d2, h_fd=1e-3, max_iter=3):
    """
    Project a point (d1,d2) back onto the curve g=psi_1+psi_2=0
    using 1–3 Newton iterations on one scalar equation.
    """
    x1, x2 = float(d1), float(d2)
    for _ in range(max_iter):
        g, gx, gy = _g_and_grad(psi_d1_func, psi_d2_func, psi_func, x1, x2, h_fd=h_fd)
        denom = gx*gx + gy*gy
        if not np.isfinite(g) or denom <= 0:
            break
        # step along gradient direction (steepest descent for the scalar eqn)
        step = g / denom
        x1 -= step * gx
        x2 -= step * gy
        if abs(g) < 1e-10:
            break
    return x1, x2


# ==============================
# g=0 contour via marching cells
# ==============================

def _edge_cross(x0, y0, x1, y1, v0, v1):
    """Linear interpolation of zero crossing on an edge where v0*v1<0."""
    t = v0 / (v0 - v1)
    return x0 + t*(x1 - x0), y0 + t*(y1 - y0)


def _g_zero_segments(g_grid, x_vals, y_vals):
    """
    Marching-squares per cell for g=0, using a coarse grid.
    Returns list of segments ((xA,yA),(xB,yB)).
    """
    segs = []
    nx, ny = g_grid.shape
    for i in range(nx-1):
        xL, xR = x_vals[i], x_vals[i+1]
        for j in range(ny-1):
            yB, yT = y_vals[j], y_vals[j+1]
            v00 = g_grid[i, j]
            v10 = g_grid[i+1, j]
            v01 = g_grid[i, j+1]
            v11 = g_grid[i+1, j+1]

            # If all same sign, skip
            smin, smax = np.min([v00,v10,v01,v11]), np.max([v00,v10,v01,v11])
            if smin > 0 or smax < 0:
                continue

            pts = []
            # bottom
            if v00 * v10 < 0:
                pts.append(_edge_cross(xL, yB, xR, yB, v00, v10))
            # right
            if v10 * v11 < 0:
                pts.append(_edge_cross(xR, yB, xR, yT, v10, v11))
            # top
            if v01 * v11 < 0:
                pts.append(_edge_cross(xL, yT, xR, yT, v01, v11))
            # left
            if v00 * v01 < 0:
                pts.append(_edge_cross(xL, yB, xL, yT, v00, v01))

            if len(pts) == 2:
                segs.append((pts[0], pts[1]))
            elif len(pts) == 4:
                # split ambiguous into two reasonable pairs
                segs.append((pts[0], pts[2]))
                segs.append((pts[1], pts[3]))
    return segs


# ===================================
# Fast critical points (coarse → 1-D)
# ===================================

class CriticalPointsFast:
    """
    Find critical points without a fine 2-D grid:
      1) Build a *coarse* grid of g=psi_1+psi_2, extract g=0 segments.
      2) Along each segment, project points back to g=0 (1–2 Newton steps).
      3) Search for sign-changes of D=det(H_psi) along the segment.
      4) Brent root of D=0 along the arclength parameter, evaluating D with
         a local 3x3 FD Hessian (only at a few points).
    """

    def __init__(self, variables, ngrid_coarse=21, dmin=-0.99, dmax=1.99, h_fd=1e-3):
        self.h = float(variables.cosmo.h)
        self.theta1 = float(variables.theta1_radian)
        self.theta2 = float(variables.theta2_radian)
        self.recal = float(variables.recal_value)

        self.ng = int(ngrid_coarse)
        self.dmin = float(dmin)
        self.dmax = float(dmax)
        self.h_fd = float(h_fd)

        self.xg = np.linspace(self.dmin, self.dmax, self.ng)
        self.yg = np.linspace(self.dmin, self.dmax, self.ng)

    # wrappers that close over slice-specific context
    def _make_psi_funcs(self, variance, chi, z):
        def psi(d1, d2):
            return float(get_psi_2cell(variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2))
        def psi1(d1, d2):
            return float(get_psi_derivative_delta1(self.h_fd, variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2))
        def psi2(d1, d2):
            return float(get_psi_derivative_delta2(self.h_fd, variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2))
        return psi, psi1, psi2

    def _g_grid(self, psi1, psi2):
        """Compute g=psi_1+psi_2 on the coarse grid."""
        gg = np.empty((self.ng, self.ng), dtype=float)
        for i, x in enumerate(self.xg):
            for j, y in enumerate(self.yg):
                gg[i, j] = psi1(x, y) + psi2(x, y)
        return gg

    def _segment_param(self, A, B, t):
        """Point on segment endpoints A=(xA,yA), B=(xB,yB) for t∈[0,1]."""
        return (A[0] + t*(B[0]-A[0]), A[1] + t*(B[1]-A[1]))

    def _D_on_segment(self, t, A, B, psi):
        """D(t) = det(H_psi) evaluated at the point projected to g=0 near segment AB."""
        x, y = self._segment_param(A, B, t)
        # snap to g=0 (one or two Newton steps)
        x, y = self._proj_g0(x, y)
        D, _ = _det_hessian_psi(psi, x, y, h_fd=self.h_fd)
        return D

    # These are set per slice
    def _bind_slice(self, variance, lw_scalar, z, chi):
        self._psi, self._psi1, self._psi2 = self._make_psi_funcs(variance, chi, z)
        self._scale = self.h / float(lw_scalar)
        self._proj_g0 = lambda x, y: _newton_project_to_g0(self._psi1, self._psi2, self._psi, x, y, h_fd=self.h_fd)

    def _lambda_out(self, x, y):
        """Return -psi_1 * h/lw at (x,y) on g=0 (your original scale)."""
        lam = - self._psi1(x, y) * self._scale
        return float(lam)

    def solve_slice(self, variance, lw_scalar, z, chi):
        """
        Return list of critical values (your λ_c) for one redshift slice.
        """
        self._bind_slice(variance, lw_scalar, z, chi)

        # 1) coarse g grid and g=0 segments
        gg = self._g_grid(self._psi1, self._psi2)
        segments = _g_zero_segments(gg, self.xg, self.yg)
        if not segments:
            return []

        crit_vals = []

        # 2) for each segment, look for D sign change and Brent root
        for A, B in segments:
            # sample D at a few t; project each sample back to g=0
            ts = np.linspace(0.0, 1.0, 9)
            Ds = []
            XY = []
            for t in ts:
                x, y = self._segment_param(A, B, t)
                x, y = self._proj_g0(x, y)
                Dval, _ = _det_hessian_psi(self._psi, x, y, h_fd=self.h_fd)
                Ds.append(Dval)
                XY.append((x, y))
            Ds = np.asarray(Ds)

            # scan consecutive pairs for sign change and refine with Brent on t
            for k in range(len(ts)-1):
                a, b = ts[k], ts[k+1]
                Da, Db = Ds[k], Ds[k+1]
                if not (np.isfinite(Da) and np.isfinite(Db)):
                    continue
                if Da == 0.0:
                    xr, yr = XY[k]
                    crit_vals.append(self._lambda_out(xr, yr))
                elif Da*Db < 0.0:
                    try:
                        t_root = brentq(lambda tt: self._D_on_segment(tt, A, B, self._psi), a, b, maxiter=100)
                        xr, yr = self._segment_param(A, B, t_root)
                        xr, yr = self._proj_g0(xr, yr)
                        crit_vals.append(self._lambda_out(xr, yr))
                    except ValueError:
                        pass

        # deduplicate nearly identical roots
        if not crit_vals:
            return []
        crit_vals = np.array(crit_vals, dtype=float)
        crit_vals = crit_vals[np.isfinite(crit_vals)]
        if crit_vals.size == 0:
            return []
        crit_vals.sort()
        dedup = [crit_vals[0]]
        for v in crit_vals[1:]:
            if not np.isclose(v, dedup[-1], rtol=1e-5, atol=1e-6):
                dedup.append(v)
        return dedup


# ==========================
# Multi-slice convenience API
# ==========================

def _as_scalar(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.ndim > 0 else float(x)

def find_critical_points_for_cosmo_fast(
    variables,
    variance,
    min_z=0,
    max_z=None,
    ngrid_coarse=21,
    dmin=-0.99,
    dmax=1.99,
    h_fd=1e-3,
):
    """
    Fast critical points across redshift slices using the g=0 → 1-D root strategy.
    Returns (smallest_positive, largest_negative) in your original output scale.

    Parameters
    ----------
    variables : object with attributes
        lensingweights, redshifts, chis, recal_value, theta{1,2}_radian, cosmo.h
    variance : any
        Passed to your RateFunction primitives.
    min_z, max_z : int
        Slice range. If max_z is None, goes to the end.
    ngrid_coarse : int
        Coarse grid to trace g=0 (21–25 is usually fine).
    dmin, dmax : float
        Domain for (delta1, delta2).
    h_fd : float
        Finite-difference step for Hessian and Newton projection.
    """
    zs = variables.redshifts
    if max_z is None:
        max_z = len(zs)
    lw_slice  = variables.lensingweights[min_z:max_z]
    z_slice   = variables.redshifts[min_z:max_z]
    chi_slice = variables.chis[min_z:max_z]

    solver = CriticalPointsFast(variables, ngrid_coarse=ngrid_coarse, dmin=dmin, dmax=dmax, h_fd=h_fd)

    all_vals = []
    for lw_i, z_i, chi_i in zip(lw_slice, z_slice, chi_slice):
        lw_s = _as_scalar(lw_i)  # ensure scalar per slice
        vals = solver.solve_slice(variance, lw_s, z_i, chi_i)
        all_vals.extend(vals)

    if not all_vals:
        return None, None

    arr = np.array(all_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None

    pos = arr[arr > 0]
    neg = arr[arr < 0]
    smallest_positive = float(np.min(pos)) if pos.size else None
    largest_negative  = float(np.max(neg)) if neg.size else None
    return smallest_positive, largest_negative


##################################################################
##################################################################
##################################################################

import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import convolve

from .RateFunction import get_psi_2cell  # keep ONLY ψ; we will FD for derivatives


# ---------- finite-diff Hessian & helpers (with ψ cache) ----------

class _PsiCache:
    def __init__(self, hq=1e-6):
        self.hq = float(hq)   # quantization to improve cache hits
        self._cache = {}
    def _key(self, x, y):
        # quantize to reduce float key churn (aligned with FD step magnitude)
        q = self.hq
        return (round(x / q) * q, round(y / q) * q)
    def get(self, func, x, y):
        k = self._key(x, y)
        v = self._cache.get(k)
        if v is None:
            v = float(func(x, y))
            self._cache[k] = v
        return v
    def clear(self):
        self._cache.clear()


def _psi_fd_hessian_cached(psi_eval, d1, d2, h=1e-3):
    """3x3 FD Hessian using cached ψ."""
    p00 = psi_eval(d1, d2)
    p10 = psi_eval(d1 + h, d2)
    pm10 = psi_eval(d1 - h, d2)
    p01 = psi_eval(d1, d2 + h)
    pm01 = psi_eval(d1, d2 - h)
    pp  = psi_eval(d1 + h, d2 + h)
    pm  = psi_eval(d1 + h, d2 - h)
    mp  = psi_eval(d1 - h, d2 + h)
    mm  = psi_eval(d1 - h, d2 - h)

    psi_xx = (p10 - 2.0*p00 + pm10) / (h*h)
    psi_yy = (p01 - 2.0*p00 + pm01) / (h*h)
    psi_xy = (pp - pm - mp + mm) / (4.0*h*h)
    return psi_xx, psi_yy, psi_xy


# ---------- marching-squares on coarse g=0 (unchanged API) ----------

def _edge_cross(x0, y0, x1, y1, v0, v1):
    t = v0 / (v0 - v1)
    return x0 + t*(x1 - x0), y0 + t*(y1 - y0)

def _g_zero_segments(g_grid, x_vals, y_vals):
    segs = []
    nx, ny = g_grid.shape
    for i in range(nx-1):
        xL, xR = x_vals[i], x_vals[i+1]
        for j in range(ny-1):
            yB, yT = y_vals[j], y_vals[j+1]
            v00 = g_grid[i, j];  v10 = g_grid[i+1, j]
            v01 = g_grid[i, j+1]; v11 = g_grid[i+1, j+1]
            smin, smax = np.min([v00,v10,v01,v11]), np.max([v00,v10,v01,v11])
            if smin > 0 or smax < 0:
                continue
            pts = []
            if v00 * v10 < 0: pts.append(_edge_cross(xL, yB, xR, yB, v00, v10))
            if v10 * v11 < 0: pts.append(_edge_cross(xR, yB, xR, yT, v10, v11))
            if v01 * v11 < 0: pts.append(_edge_cross(xL, yT, xR, yT, v01, v11))
            if v00 * v01 < 0: pts.append(_edge_cross(xL, yB, xL, yT, v00, v01))
            if len(pts) == 2:
                segs.append((pts[0], pts[1]))
            elif len(pts) == 4:
                segs.append((pts[0], pts[2])); segs.append((pts[1], pts[3]))
    return segs


# ---------- coarse-grid derivatives via convolution (fast) ----------

def _build_stencils(dx, dy):
    Kx = np.array([[0, 0, 0],
                   [-0.5, 0, 0.5],
                   [0, 0, 0]]) / dx
    Ky = np.array([[0, -0.5, 0],
                   [0,  0.0, 0],
                   [0,  0.5, 0]]) / dy
    Dxx = np.array([[0, 0, 0],
                    [1,-2, 1],
                    [0, 0, 0]]) / (dx*dx)
    Dyy = np.array([[0, 1, 0],
                    [0,-2, 0],
                    [0, 1, 0]]) / (dy*dy)
    Dxy = np.array([[ 1, 0,-1],
                    [ 0, 0, 0],
                    [-1, 0, 1]]) / (4*dx*dy)
    return Kx, Ky, Dxx, Dyy, Dxy

def _bilinear(Z, x, y, xs, ys):
    # bilinear using the regular grid xs, ys (monotonic)
    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    i = np.clip(i, 0, len(xs)-2); j = np.clip(j, 0, len(ys)-2)
    x0, x1 = xs[i], xs[i+1]; y0, y1 = ys[j], ys[j+1]
    tx = (x - x0)/(x1 - x0); ty = (y - y0)/(y1 - y0)
    f00 = Z[i, j]; f10 = Z[i+1, j]; f01 = Z[i, j+1]; f11 = Z[i+1, j+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11


# ===================================
# Fast critical points (improved)
# ===================================

class CriticalPointsFast:
    """
    Same algorithm as your fast option, but:
      - cache ψ evaluations everywhere,
      - build coarse ψ grid once, get g and det(Hψ) by convolution,
      - use coarse det(Hψ) only to bracket; do FD Hessian only inside Brent,
      - project to g=0 only once at the end.
    """

    def __init__(self, variables, ngrid_coarse=21, dmin=-0.99, dmax=1.99, h_fd=1e-3):
        self.h = float(variables.cosmo.h)
        self.theta1 = float(variables.theta1_radian)
        self.theta2 = float(variables.theta2_radian)
        self.recal = float(variables.recal_value)

        self.ng = int(ngrid_coarse)
        self.dmin = float(dmin); self.dmax = float(dmax)
        self.h_fd = float(h_fd)

        self.xg = np.linspace(self.dmin, self.dmax, self.ng)
        self.yg = np.linspace(self.dmin, self.dmax, self.ng)
        self.dx = self.xg[1]-self.xg[0]; self.dy = self.yg[1]-self.yg[0]
        self.Kx, self.Ky, self.Dxx, self.Dyy, self.Dxy = _build_stencils(self.dx, self.dy)

        self._psi_cache = _PsiCache(hq=min(1e-6, 0.1*self.h_fd))  # quantization tied to FD step

    # ψ with cache
    def _psi_eval(self, variance, chi, z):
        def base(d1, d2):
            return get_psi_2cell(variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2)
        def wrapped(d1, d2):
            return self._psi_cache.get(base, d1, d2)
        return wrapped

    # one slice
    def solve_slice(self, variance, lw_scalar, z, chi):
        if not np.isfinite(lw_scalar) or abs(lw_scalar) < 1e-30:
            return []

        psi = self._psi_eval(variance, chi, z)

        # --- (A) coarse ψ grid once ---
        PSI = np.empty((self.ng, self.ng), dtype=float)
        for i, x in enumerate(self.xg):
            for j, y in enumerate(self.yg):
                PSI[i, j] = psi(x, y)

        # derivatives on coarse grid via convolution (cheap)
        PSI_x  = convolve(PSI, self.Kx,  mode="nearest")
        PSI_y  = convolve(PSI, self.Ky,  mode="nearest")
        PSI_xx = convolve(PSI, self.Dxx, mode="nearest")
        PSI_yy = convolve(PSI, self.Dyy, mode="nearest")
        PSI_xy = convolve(PSI, self.Dxy, mode="nearest")

        g_grid   = PSI_x + PSI_y                      # zeros unaffected by scaling
        detH_coarse = PSI_xx*PSI_yy - PSI_xy*PSI_xy   # cheap approximate det(Hψ)

        # --- (B) extract g=0 segments on the coarse grid ---
        segments = _g_zero_segments(g_grid, self.xg, self.yg)
        if not segments:
            self._psi_cache.clear()
            return []

        crit_vals = []
        scale = self.h / float(lw_scalar)

        # --- (C) for each segment, bracket with cheap detH; refine with FD only inside Brent ---
        for A, B in segments:
            # sample coarse detH along segment to find brackets
            ts = np.linspace(0.0, 1.0, 9)
            Dc = np.array([_bilinear(detH_coarse,
                                     A[0] + t*(B[0]-A[0]),
                                     A[1] + t*(B[1]-A[1]),
                                     self.xg, self.yg) for t in ts])

            for k in range(len(ts)-1):
                Da, Db = Dc[k], Dc[k+1]
                if not (np.isfinite(Da) and np.isfinite(Db)):
                    continue
                if Da == 0.0:
                    t0 = ts[k]
                elif Da*Db < 0.0:
                    # Brent over t with TRUE (FD) detH — but no projection each call
                    def D_true(t):
                        x = A[0] + t*(B[0]-A[0])
                        y = A[1] + t*(B[1]-A[1])
                        xx, yy, xy = _psi_fd_hessian_cached(psi, x, y, h=self.h_fd)
                        return xx*yy - xy*xy
                    try:
                        t0 = brentq(D_true, ts[k], ts[k+1], maxiter=100, xtol=1e-8, rtol=1e-6)
                    except ValueError:
                        continue
                else:
                    continue

                # final point (optionally snap once to g=0 using 1 Newton-like step with coarse grad)
                xr = A[0] + t0*(B[0]-A[0])
                yr = A[1] + t0*(B[1]-A[1])

                # 1 cheap correction step towards g=0 using coarse grad (no extra ψ calls)
                gx = _bilinear(PSI_x, xr, yr, self.xg, self.yg)
                gy = _bilinear(PSI_y, xr, yr, self.xg, self.yg)
                g  = gx + gy
                dgd1 = _bilinear(PSI_xx + PSI_xy, xr, yr, self.xg, self.yg)
                dgd2 = _bilinear(PSI_xy + PSI_yy, xr, yr, self.xg, self.yg)
                denom = dgd1*dgd1 + dgd2*dgd2
                if np.isfinite(g) and denom > 0:
                    step = g/denom
                    xr -= step*dgd1; yr -= step*dgd2

                # output λ_c = -ψ_1*h/lw using FD of ψ_1 (2 extra cached ψ calls)
                psi1 = (psi(xr + self.h_fd, yr) - psi(xr - self.h_fd, yr)) / (2*self.h_fd)
                lam = - psi1 * scale
                if np.isfinite(lam):
                    crit_vals.append(lam)

        self._psi_cache.clear()

        if not crit_vals:
            return []
        crit_vals = np.array(crit_vals, dtype=float)
        crit_vals = crit_vals[np.isfinite(crit_vals)]
        if crit_vals.size == 0:
            return []
        crit_vals.sort()
        out = [crit_vals[0]]
        for v in crit_vals[1:]:
            if not np.isclose(v, out[-1], rtol=1e-5, atol=1e-6):
                out.append(v)
        return out


# --------------- driver ---------------

def _as_scalar(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.ndim > 0 else float(x)

def find_critical_points_for_cosmo_fast(
    variables,
    variance,
    min_z=0,
    max_z=None,
    ngrid_coarse=21,
    dmin=-0.99,
    dmax=1.99,
    h_fd=1e-3,
):
    if max_z is None:
        max_z = len(variables.redshifts)
    lw_slice  = variables.lensingweights[min_z:max_z]
    z_slice   = variables.redshifts[min_z:max_z]
    chi_slice = variables.chis[min_z:max_z]

    solver = CriticalPointsFast(variables, ngrid_coarse=ngrid_coarse, dmin=dmin, dmax=dmax, h_fd=h_fd)

    all_vals = []
    for lw_i, z_i, chi_i in zip(lw_slice, z_slice, chi_slice):
        lw_s = _as_scalar(lw_i)
        all_vals.extend(solver.solve_slice(variance, lw_s, z_i, chi_i))

    if not all_vals:
        return None, None
    arr = np.array(all_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    pos = arr[arr > 0]; neg = arr[arr < 0]
    # print(" the critical points are: ", np.min(pos), np.max(neg))

    return (float(np.min(pos)) if pos.size else None,
            float(np.max(neg)) if neg.size else None)


import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import convolve

# Your provided primitives
from .RateFunction import (
    get_psi_2cell,
    get_psi_derivative_delta1,
    get_psi_derivative_delta2,
)

# =========================================================
# Utilities: caching, finite-difference, marching-squares
# =========================================================

class _PsiCache:
    """Quantized cache for ψ(d1,d2) to cut repeated calls in FD Hessians."""
    def __init__(self, hq=1e-6):
        self.hq = float(hq)
        self._cache = {}
    def _key(self, x, y):
        q = self.hq
        return (round(x / q) * q, round(y / q) * q)
    def get(self, func, x, y):
        k = self._key(x, y)
        v = self._cache.get(k)
        if v is None:
            v = float(func(x, y))
            self._cache[k] = v
        return v
    def clear(self):
        self._cache.clear()

def _psi_fd_hessian_cached(psi_eval, d1, d2, h=1e-3):
    """3×3 central-diff Hessian using cached ψ."""
    p00 = psi_eval(d1, d2)
    p10 = psi_eval(d1 + h, d2)
    pm10 = psi_eval(d1 - h, d2)
    p01 = psi_eval(d1, d2 + h)
    pm01 = psi_eval(d1, d2 - h)
    pp  = psi_eval(d1 + h, d2 + h)
    pm  = psi_eval(d1 + h, d2 - h)
    mp  = psi_eval(d1 - h, d2 + h)
    mm  = psi_eval(d1 - h, d2 - h)

    psi_xx = (p10 - 2.0*p00 + pm10) / (h*h)
    psi_yy = (p01 - 2.0*p00 + pm01) / (h*h)
    psi_xy = (pp - pm - mp + mm) / (4.0*h*h)
    return psi_xx, psi_yy, psi_xy

def _edge_cross(x0, y0, x1, y1, v0, v1, eps=0.0):
    """Linear interp of zero crossing on an edge where v0*v1<0."""
    denom = (v0 - v1)
    if denom == 0:
        denom = np.copysign(eps if eps > 0 else 1e-16, denom if denom != 0 else 1.0)
    t = v0 / denom
    return x0 + t*(x1 - x0), y0 + t*(y1 - y0)

def _g_zero_segments(g_grid, x_vals, y_vals):
    """
    Marching-squares per cell for g=0 on a coarse grid.
    Returns list of segments: ((xA,yA),(xB,yB)).
    """
    segs = []
    nx, ny = g_grid.shape
    for i in range(nx-1):
        xL, xR = x_vals[i], x_vals[i+1]
        for j in range(ny-1):
            yB, yT = y_vals[j], y_vals[j+1]
            v00 = g_grid[i, j]
            v10 = g_grid[i+1, j]
            v01 = g_grid[i, j+1]
            v11 = g_grid[i+1, j+1]

            smin, smax = np.min([v00, v10, v01, v11]), np.max([v00, v10, v01, v11])
            if smin > 0 or smax < 0:
                continue

            pts = []
            if v00 * v10 < 0: pts.append(_edge_cross(xL, yB, xR, yB, v00, v10))
            if v10 * v11 < 0: pts.append(_edge_cross(xR, yB, xR, yT, v10, v11))
            if v01 * v11 < 0: pts.append(_edge_cross(xL, yT, xR, yT, v01, v11))
            if v00 * v01 < 0: pts.append(_edge_cross(xL, yB, xL, yT, v00, v01))

            if len(pts) == 2:
                segs.append((pts[0], pts[1]))
            elif len(pts) == 4:
                segs.append((pts[0], pts[2]))
                segs.append((pts[1], pts[3]))
    return segs

def _build_stencils(dx, dy):
    """Central-diff stencils for first/second derivatives on a regular grid."""
    Kx = np.array([[0, 0, 0],
                   [-0.5, 0, 0.5],
                   [0, 0, 0]]) / dx
    Ky = np.array([[0, -0.5, 0],
                   [0,  0.0, 0],
                   [0,  0.5, 0]]) / dy
    Dxx = np.array([[0, 0, 0],
                    [1,-2, 1],
                    [0, 0, 0]]) / (dx*dx)
    Dyy = np.array([[0, 1, 0],
                    [0,-2, 0],
                    [0, 1, 0]]) / (dy*dy)
    Dxy = np.array([[ 1, 0,-1],
                    [ 0, 0, 0],
                    [-1, 0, 1]]) / (4*dx*dy)
    return Kx, Ky, Dxx, Dyy, Dxy

# =========================================================
# Main solver: coarse g=0 → 1D root of det(Hψ)=0 ON g=0
# =========================================================

class CriticalPointsFast:
    """
    Fast, correct critical-points finder:

      1) Build coarse ψ grid once (M×M) and compute
         ψx, ψy, ψxx, ψyy, ψxy via cheap convolutions.
      2) Extract g=ψx+ψy=0 contour (marching squares).
      3) Along each segment, bracket sign-changes of det(Hψ)
         using the cheap coarse det(Hψ) for speed.
      4) Refine with Brent on t∈[0,1], but **evaluate det(Hψ) after
         projecting the point to g=0** at **every** function call.
      5) Report λc = -ψ₁*h/lw using your derivative primitive.
    """

    def __init__(self, variables, ngrid_coarse=21, dmin=-0.99, dmax=1.99, h_fd=1e-3,
                 n_samples=9, proj_iters=3):
        self.h       = float(variables.cosmo.h)
        self.theta1  = float(variables.theta1_radian)
        self.theta2  = float(variables.theta2_radian)
        self.recal   = float(variables.recal_value)

        self.ng      = int(ngrid_coarse)
        self.dmin    = float(dmin)
        self.dmax    = float(dmax)
        self.h_fd    = float(h_fd)
        self.n_samples = int(max(5, n_samples))
        self.proj_iters = int(max(1, proj_iters))

        self.xg = np.linspace(self.dmin, self.dmax, self.ng)
        self.yg = np.linspace(self.dmin, self.dmax, self.ng)
        self.dx = self.xg[1] - self.xg[0]
        self.dy = self.yg[1] - self.yg[0]
        self.Kx, self.Ky, self.Dxx, self.Dyy, self.Dxy = _build_stencils(self.dx, self.dy)

        # cache for ψ evaluations (used in FD Hessian and projection)
        self._psi_cache = _PsiCache(hq=min(1e-6, 0.1*self.h_fd))

    # --- support ---

    def _clip(self, x, y):
        """Keep FD stencils inside domain."""
        eps = 2.5*self.h_fd
        return (np.clip(x, self.dmin+eps, self.dmax-eps),
                np.clip(y, self.dmin+eps, self.dmax-eps))

    def _psi_eval(self, variance, chi, z):
        """Closure + cache wrapper for ψ(d1,d2)."""
        def base(d1, d2):
            return get_psi_2cell(variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2)
        def wrapped(d1, d2):
            x, y = self._clip(d1, d2)
            return self._psi_cache.get(base, x, y)
        return wrapped

    def _g_exact(self, d1, d2, variance, chi, z):
        """g = ψ₁ + ψ₂ using your derivative primitives (stable)."""
        d1v = get_psi_derivative_delta1(self.h_fd, variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2)
        d2v = get_psi_derivative_delta2(self.h_fd, variance, chi, self.recal, z, d1, d2, self.theta1, self.theta2)
        return float(d1v + d2v)

    def _proj_to_g0(self, x, y, variance, chi, z, psi_eval):
        """Small number of Newton-like steps to land on g=0."""
        x, y = float(x), float(y)
        for _ in range(self.proj_iters):
            x, y = self._clip(x, y)
            g  = self._g_exact(x, y, variance, chi, z)
            xx, yy, xy = _psi_fd_hessian_cached(psi_eval, x, y, h=self.h_fd)
            gx, gy = xx + xy, xy + yy
            den = gx*gx + gy*gy
            if not (np.isfinite(g) and den > 0):
                break
            step = g / den
            x -= step * gx
            y -= step * gy
            if abs(g) < 1e-10:
                break
        return self._clip(x, y)

    # --- one slice ---

    def solve_slice(self, variance, lw_scalar, z, chi):
        if not np.isfinite(lw_scalar) or abs(lw_scalar) < 1e-30:
            return []

        psi_eval = self._psi_eval(variance, chi, z)
        scale    = self.h / float(lw_scalar)

        # (A) coarse ψ grid once
        PSI = np.empty((self.ng, self.ng), dtype=float)
        for i, x in enumerate(self.xg):
            for j, y in enumerate(self.yg):
                PSI[i, j] = psi_eval(x, y)

        # cheap derivatives via convolution
        PSI_x  = convolve(PSI, self.Kx,  mode="nearest")
        PSI_y  = convolve(PSI, self.Ky,  mode="nearest")
        PSI_xx = convolve(PSI, self.Dxx, mode="nearest")
        PSI_yy = convolve(PSI, self.Dyy, mode="nearest")
        PSI_xy = convolve(PSI, self.Dxy, mode="nearest")

        g_grid = PSI_x + PSI_y
        detH_coarse = PSI_xx * PSI_yy - PSI_xy * PSI_xy

        # (B) g=0 segments
        segments = _g_zero_segments(g_grid, self.xg, self.yg)
        if not segments:
            self._psi_cache.clear()
            return []

        crit_vals = []
        ts = np.linspace(0.0, 1.0, self.n_samples)

        # (C) for each segment: bracket with cheap detH; refine with projected Brent
        for A, B in segments:
            # coarse bracket along the segment
            Dc = []
            for t in ts:
                x = A[0] + t*(B[0]-A[0])
                y = A[1] + t*(B[1]-A[1])
                # bilinear of detH_coarse along the segment (reuse grid indexing directly)
                # Use manual bilinear to avoid extra deps:
                i = np.searchsorted(self.xg, x) - 1
                j = np.searchsorted(self.yg, y) - 1
                i = np.clip(i, 0, self.ng-2)
                j = np.clip(j, 0, self.ng-2)
                x0, x1 = self.xg[i], self.xg[i+1]
                y0, y1 = self.yg[j], self.yg[j+1]
                tx = (x - x0) / (x1 - x0 + 1e-16)
                ty = (y - y0) / (y1 - y0 + 1e-16)
                f00 = detH_coarse[i, j]; f10 = detH_coarse[i+1, j]
                f01 = detH_coarse[i, j+1]; f11 = detH_coarse[i+1, j+1]
                Dc.append((1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11)
            Dc = np.asarray(Dc)

            for k in range(len(ts)-1):
                Da, Db = Dc[k], Dc[k+1]
                if not (np.isfinite(Da) and np.isfinite(Db)):
                    continue

                # quick true signs at projected endpoints to avoid false brackets
                xa = A[0] + ts[k]  *(B[0]-A[0]); ya = A[1] + ts[k]  *(B[1]-A[1])
                xb = A[0] + ts[k+1]*(B[0]-A[0]); yb = A[1] + ts[k+1]*(B[1]-A[1])
                xa, ya = self._proj_to_g0(xa, ya, variance, chi, z, psi_eval)
                xb, yb = self._proj_to_g0(xb, yb, variance, chi, z, psi_eval)
                xxa, yya, xya = _psi_fd_hessian_cached(psi_eval, xa, ya, h=self.h_fd)
                xxb, yyb, xyb = _psi_fd_hessian_cached(psi_eval, xb, yb, h=self.h_fd)
                Da_true = xxa*yya - xya*xya
                Db_true = xxb*yyb - xyb*xyb

                if Da_true == 0.0:
                    # Already on root
                    psi1 = get_psi_derivative_delta1(self.h_fd, variance, chi, self.recal, z, xa, ya, self.theta1, self.theta2)
                    lam = - float(psi1) * scale
                    if np.isfinite(lam):
                        crit_vals.append(lam)
                    continue

                if not (np.isfinite(Da_true) and np.isfinite(Db_true) and Da_true*Db_true < 0.0):
                    continue

                # Brent on t with projected det(Hψ)
                def D_true(t):
                    x = A[0] + t*(B[0]-A[0])
                    y = A[1] + t*(B[1]-A[1])
                    x, y = self._proj_to_g0(x, y, variance, chi, z, psi_eval)
                    xx, yy, xy = _psi_fd_hessian_cached(psi_eval, x, y, h=self.h_fd)
                    return xx*yy - xy*xy

                try:
                    t0 = brentq(D_true, ts[k], ts[k+1], maxiter=100, xtol=1e-8, rtol=1e-6)
                except ValueError:
                    continue

                xr = A[0] + t0*(B[0]-A[0])
                yr = A[1] + t0*(B[1]-A[1])
                xr, yr = self._proj_to_g0(xr, yr, variance, chi, z, psi_eval)

                # λc = -ψ₁*h/lw via your derivative primitive (accurate, single call)
                psi1 = get_psi_derivative_delta1(self.h_fd, variance, chi, self.recal, z, xr, yr, self.theta1, self.theta2)
                lam = - float(psi1) * scale
                if np.isfinite(lam):
                    crit_vals.append(lam)

        self._psi_cache.clear()

        if not crit_vals:
            return []
        crit_vals = np.array(crit_vals, dtype=float)
        crit_vals = crit_vals[np.isfinite(crit_vals)]
        if crit_vals.size == 0:
            return []
        crit_vals.sort()
        out = [crit_vals[0]]
        for v in crit_vals[1:]:
            if not np.isclose(v, out[-1], rtol=1e-5, atol=1e-6):
                out.append(v)
        return out

# =========================================================
# Multi-slice driver
# =========================================================

def _as_scalar(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.ndim > 0 else float(x)

def find_critical_points_for_cosmo_fast(
    variables,
    variance,
    min_z=0,
    max_z=None,
    ngrid_coarse=21,
    dmin=-0.99,
    dmax=1.99,
    h_fd=1e-3,
    n_samples=9,
    proj_iters=3,
):
    """
    Fast critical points across redshift slices using g=0 → 1-D root on g=0.
    Returns (smallest_positive, largest_negative) in your original scale.
    """
    zs = variables.redshifts
    if max_z is None:
        max_z = len(zs)
    lw_slice  = variables.lensingweights[min_z:max_z]
    z_slice   = variables.redshifts[min_z:max_z]
    chi_slice = variables.chis[min_z:max_z]

    solver = CriticalPointsFast(
        variables,
        ngrid_coarse=ngrid_coarse,
        dmin=dmin,
        dmax=dmax,
        h_fd=h_fd,
        n_samples=n_samples,
        proj_iters=proj_iters,
    )

    all_vals = []
    for lw_i, z_i, chi_i in zip(lw_slice, z_slice, chi_slice):
        lw_s = _as_scalar(lw_i)
        all_vals.extend(solver.solve_slice(variance, lw_s, z_i, chi_i))

    if not all_vals:
        return None, None

    arr = np.array(all_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None

    pos = arr[arr > 0]
    neg = arr[arr < 0]
    smallest_positive = float(np.min(pos)) if pos.size else None
    largest_negative  = float(np.max(neg)) if neg.size else None
    return smallest_positive, largest_negative
