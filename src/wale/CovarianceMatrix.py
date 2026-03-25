import numpy as np
import pyccl as ccl
from pyccl.halos.pk_4pt import halomod_Tk3D_4h
from pyccl.halos.pk_4pt import halomod_Tk3D_cNG
from scipy.integrate import quad
import matplotlib.pyplot as plt

# def get_covariance(cosmo, z, variability, numberofrealisations):
#     """
#     Compute the nonlinear matter power spectrum P(k) and optionally its covariance
#     using halo model trispectrum contributions.

#     If `variability` is enabled, the function uses the halo model (via pyccl) to
#     compute the connected non-Gaussian trispectrum and generate power spectrum
#     realizations by sampling from a multivariate Gaussian distribution.

#     Parameters
#     ----------
#     cosmo : Cosmology_function
#         A Cosmology_function object that wraps pyccl and includes nonlinear P(k) access,
#         k-grid, and other cosmological parameters.
#     z : array_like
#         Array of redshift values at which the power spectrum is evaluated.
#     variability : bool
#         Whether to compute and include non-Gaussian covariance (trispectrum) and
#         draw realizations of P(k) using a halo model.
#     numberofrealisations : int
#         Number of mock realizations to draw for each redshift (if `variability=True`).

#     Returns
#     -------
#     if variability is True:
#         cov_dict : dict
#             Dictionary mapping redshift z to full covariance matrix C(k, k').
#         pk_samples_dict : dict
#             Dictionary mapping redshift z to an array of shape (N, nk) containing N sampled
#             realizations of P(k).
#         pk_dict : dict
#             Dictionary mapping redshift z to the mean nonlinear P(k) at that redshift.
#     else:
#         pk_dict : dict
#             Dictionary mapping redshift z to the mean nonlinear P(k) at that redshift.
#     """
#     Lbox = 505  # Mpc/h
#     vol = Lbox**3

#     Nmodes = (
#         vol
#         / 3
#         / (2 * np.pi**2)
#         * ((cosmo.k + cosmo.dk / 2) ** 3 - (cosmo.k - cosmo.dk / 2) ** 3)
#     )  # Number of k-modes in shells

#     sf = 1.0 / (1.0 + z)
#     # 2) get the sorting indices for ascending order
#     idx = np.argsort(sf)
#     # 3) reorder
#     scale_factor = sf[idx]

#     Pnl = cosmo.get_nonlinear_pk(z, cosmo.k)
#     # print(Pnl.shape, "shape of Pnl")
#     # print(scale_factor.shape, "shape of scale_factor")
#     if variability:
#         # We will use a mass definition with Delta = 200 times the matter density
#         hmd_200m = "200m"

#         # The Duffy 2008 concentration-mass relation
#         cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

#         # The Tinker 2008 mass function
#         nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)

#         # The Tinker 2010 halo bias
#         bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

#         # The NFW profile to characterize the matter density around halos
#         prof = ccl.halos.HaloProfileNFW(
#             mass_def=hmd_200m, concentration=cM, fourier_analytic=True
#         )
#         print("          Using NFW profile with mass definition:", hmd_200m)

#         hmc = ccl.halos.halo_model.HMCalculator(
#             mass_function=nM,  # must be a keyword
#             halo_bias=bM,  # must be a keyword
#             mass_def=hmd_200m,  # optional (default is 200m anyway)
#         )
#         # print("step 2 done")

#         # 4) build trispectrum splines ONCE
#         Tk = halomod_Tk3D_cNG(
#             cosmo=cosmo.cosmoccl,
#             hmc=hmc,
#             prof=prof,
#             lk_arr=np.log(cosmo.k),  # interpolate in ln k exactly where you want
#             a_arr=np.atleast_1d(
#                 scale_factor
#             ),  # only one scale factor → 2D interpolation
#             use_log=True,  # builds spline in log‐space for accuracy
#             separable_growth=False,
#         )
#         # print("step 3 done")
#         Tmat = Tk(cosmo.k, scale_factor)
#         # print("step 4 done")
#         Cgauss = np.array([np.diag(2.0 * Pnl[i] ** 2 / Nmodes) for i in range(len(z))])
        
#         # print(Cgauss.shape, "shape of Cgauss")
#         # print(Tmat.shape, "shape of Tmat")
#         Cfull = Cgauss + Tmat / vol
        
#         # jitter = 1e-8 * np.diag(np.diag(Cfull))
#         cov = Cfull  # + jitter
#         N = numberofrealisations  # number of realizations per redshift
#         na, nk = Pnl.shape

#         # container: shape (na, N, nk)
#         pnl_samples = np.empty((na, N, nk))
#         pk = []
#         for i in range(na):
#             mean_i = Pnl[i]  # length-nk mean vector at a_vals[i]
#             cov_i = cov[i]  # same covariance used for all, or recompute per-z if needed
#             cov_i = cov_i + np.eye(nk) * 1e-12 * np.trace(cov_i) / (nk)
#             pnl_samples[i] = np.random.multivariate_normal(mean_i, cov_i, size=N)
#         pk_dict = {z_: Pnl[i, :] for i, z_ in enumerate(z)}
#         # pk_samples_dict = {z_: pnl_samples[i, :] for i, z_ in enumerate(z)}
#         for i in range(N):
#             pk_i = {z[j]: pnl_samples[j, i, :] for j in range(na)}  # dict: redshift → P(k)
#             pk.append(pk_i) 
#         cov_dict = {z_: cov[i, :] for i, z_ in enumerate(z)}
#         return cov_dict, pk, pk_dict
#     else:
#         pk_dict = {z_: Pnl[i, :] for i, z_ in enumerate(z)}
#         return pk_dict


# def get_covariance(cosmo, z, Lbox=505.0, k_survey=None):
#     # ---- 1) setup ----
#     ks = cosmo.k
#     nk = len(ks)
#     vol = Lbox**3
#     a   = 1.0/(1.0+z)

#     # Gaussian diagonal
#     dk     = cosmo.dk
#     # exact mode count if you prefer:
#     Nmodes = vol/(2*np.pi**2)/3 * ((ks+dk/2)**3 - (ks-dk/2)**3)
#     Pnl    = ccl.nonlin_matter_power(cosmo.cosmoccl, ks, a)
#     Cgauss = np.diag(2.0 * Pnl**2 / Nmodes)

#     # ---- 2) tree‐level trispectrum pieces ----
#     # 2.1) 1-halo:
#     Tk1 = halomod_Tk3D_1h(cosmo = cosmo.cosmoccl, hmc   = hmc, prof  = prof, use_log = True, separable_growth = False)
#     T1 = Tk1(ks, a)   # shape (nk,nk)

#     # 2.2) 3-halo:
#     Tk3 = halomod_Tk3D_3h(cosmo = cosmo.cosmoccl, hmc   = hmc, prof  = prof, use_log = True, separable_growth = False)
#     T3 = Tk3(ks, a)

#     # 2.3) 4-halo (tree‐level):
#     Tk4 = Tk3D_pt(
#         cosmo = cosmo.cosmoccl,
#         lk_arr = None,    # let CCL pick its internal grid
#         a_arr  = None
#     )
#     T4 = Tk4(ks, a)

#     # assemble tree‐level covariance (skipping the two slow 2‐halo terms)
#     C_tree = (T1 + T3 + T4) / vol

#     # ---- 3) super‐sample covariance (SSC) via Eq. (D.3–D.5) ----
#     # 3.1) response ∂P/∂δb from Eq. (D.3)
#     #    here I use the “separate‐universe” trick in CCL:
#     dP_deltab = ccl.covariances.pk_s_sigma(cosmo.cosmoccl, ks, a)
#     # 3.2) σ²_b from Eq. (D.4) for a square mask of area A_survey
#     if k_survey is None:
#         raise ValueError("Please pass the survey side length in Mpc/h via k_survey")
#     A_survey = k_survey**2
#     def Mtil(lx,ly):
#         # Eq. D.5: sinc mask Fourier transform for a square
#         L = np.sqrt(A_survey)
#         return np.sinc(lx*L/2/np.pi) * np.sinc(ly*L/2/np.pi)

#     def integrand(l):
#         # integrate over |ℓ|
#         return l * special.j0(0)  # dummy: replace with actual ∫dφ |M̃|² P(l/χ)
#     # for brevity, you can approximate σ²_b analytically for a square:
#     chi = ccl.comoving_radial_distance(cosmo.cosmoccl, a)
#     sigma_b2 = (1/A_survey) * np.trapz(
#         Mtil(chi*ks, chi*ks)**2 * ccl.linear_matter_power(cosmo.cosmoccl, ks/chi, a),
#         ks
#     )

#     Css = np.outer(dP_deltab, dP_deltab) * sigma_b2 / vol

#     # ---- 4) final sum ----
#     C_full = Cgauss + C_tree + Css

#     return ks, C_full


def get_sigma_covariance(
    cosmo,
    z,
    theta1,
    theta2,
    filter_type="tophat",
    f_sky=0.35,
    A_survey_deg2=None,
    use_ssc=True,
    use_ng=True,
    Lbox=505.0,
):
    """
    Compute the (3*nz) × (3*nz) covariance matrix of the LDT variance projections
    (σ₁₁, σ₂₂, σ₁₂) across redshift slices.

    The output vector is ordered as:
        s = [σ₁₁(z₀), σ₂₂(z₀), σ₁₂(z₀), σ₁₁(z₁), σ₂₂(z₁), σ₁₂(z₁), ...]

    Three covariance components are included:

    **Gaussian**: diagonal in k, block-diagonal in z (finite-volume cosmic variance)::

        Cov_G[σ_ab(z), σ_cd(z')] = δ_{zz'} Σ_k 2 f_ab(k,z) f_cd(k,z) / N_modes(k)

    **Non-Gaussian i-trispectrum** (Gualdi et al. 2021), block-diagonal in z::

        Cov_NG[σ_ab(z), σ_cd(z')] = δ_{zz'} [Σ_k f_ab A_eff] [Σ_k f_cd A_eff] / V

    **Super-sample covariance (SSC)**, rank-1 across *all* z-slices::

        Cov_SSC[σ_ab(z), σ_cd(z')] = [Σ_k f_ab R₁(k,z)] [Σ_k f_cd R₁(k,z')] σ²_b

    where ``f_ab(k,z) = (k/2π) W_a(kR_a) W_b(kR_b) dk`` with fiducial radii
    R_a = χ(z) θ_a.

    Parameters
    ----------
    cosmo : Cosmology_function
        Provides ``cosmo.k`` (1/Mpc), ``cosmo.h``, ``cosmo.cosmoccl`` (pyccl),
        and ``cosmo.get_nonlinear_pk(z, k)``.
    z : array_like, shape (nz,)
        Redshifts at which P(k) is evaluated (must be sorted ascending).
    theta1, theta2 : float
        Angular scales (radians) of the two LDT cells.
    filter_type : {'tophat', 'starlet'}
        Fourier-space window function type.
    f_sky : float, optional
        Survey sky fraction (default 0.35, roughly Euclid-like). Used for SSC
        only when *A_survey_deg2* is None.
    A_survey_deg2 : float or None, optional
        Survey area in deg². Overrides *f_sky* when provided.
    use_ssc : bool, optional
        Include SSC component (default True).
    use_ng : bool, optional
        Include non-Gaussian i-trispectrum component (default True).
    Lbox : float, optional
        Simulation box side in Mpc/h for the Gaussian (shot-noise) term
        (default 505.0, the SLICS box).

    Returns
    -------
    s_mean : ndarray, shape (3*nz,)
        Mean LDT variance projections at fiducial cosmology,
        ``[σ₁₁(z₀), σ₂₂(z₀), σ₁₂(z₀), σ₁₁(z₁), ...]``.
    C_sigma : ndarray, shape (3*nz, 3*nz)
        Full covariance matrix of *s_mean*.
    """
    from .FilterFunctions import top_hat_filter_numpy, starlet_filter_numpy

    z = np.asarray(z, dtype=float)
    nz = len(z)
    k = np.asarray(cosmo.k, dtype=float)

    # Trapezoid weights on the k-grid
    k_edges = np.concatenate([[k[0]], 0.5 * (k[:-1] + k[1:]), [k[-1]]])
    dk = np.diff(k_edges)

    # Gaussian term: simulation volume determines N_modes
    vol = (Lbox / cosmo.h) ** 3  # Mpc³
    Nmodes = vol / 3.0 / (2.0 * np.pi ** 2) * (
        (k + dk / 2.0) ** 3 - (k - dk / 2.0) ** 3
    )

    # Comoving distances at each z
    a_arr = 1.0 / (1.0 + z)
    chi_arr = np.asarray(ccl.comoving_radial_distance(cosmo.cosmoccl, a_arr))

    # SSC: σ²_b = variance of the matter density background over the survey volume
    sigma2_b = 0.0
    if use_ssc:
        if A_survey_deg2 is not None:
            A_survey_sr = float(A_survey_deg2) * (np.pi / 180.0) ** 2
        else:
            A_survey_sr = float(f_sky) * 4.0 * np.pi

        chi_eff = float(np.mean(chi_arr))
        Dchi = float(chi_arr[-1] - chi_arr[0]) if nz > 1 else 0.1 * chi_eff
        V_survey = A_survey_sr * chi_eff ** 2 * Dchi  # Mpc³
        R_survey = (3.0 * V_survey / (4.0 * np.pi)) ** (1.0 / 3.0)

        a_eff = 1.0 / (1.0 + float(np.mean(z)))
        P_lin_eff = np.asarray(ccl.linear_matter_power(cosmo.cosmoccl, k, a_eff))

        # 3-D top-hat window W_3D(kR)
        kR = k * R_survey
        with np.errstate(divide="ignore", invalid="ignore"):
            W_3D = 3.0 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)
        W_3D = np.where(kR < 1e-6, np.ones_like(kR), W_3D)
        sigma2_b = float(np.trapezoid(k ** 2 * P_lin_eff * W_3D ** 2 / (2.0 * np.pi ** 2), k))

    # Nonlinear P(k) at each z: shape (nz, nk)
    Pnl = np.vstack([cosmo.get_nonlinear_pk(z_i, k) for z_i in z])

    # Nonlinear growth response R₁(k,z) ≈ 26/21 + (1/3) d(ln P_nl)/d(ln k)
    ln_k = np.log(k)
    ln_Pnl = np.log(np.maximum(Pnl, 1e-300))
    d_lnP_d_lnk = np.gradient(ln_Pnl, ln_k, axis=1)   # (nz, nk)
    R1_response = 26.0 / 21.0 + (1.0 / 3.0) * d_lnP_d_lnk  # (nz, nk)

    # Filter projections f_ab[i, m] = (k_m / 2π) W_a(k_m χ_i θ_a) W_b(k_m χ_i θ_b) dk_m
    # Evaluated at fiducial δ = 0 → R_a = χ(z_i) θ_a.
    nk = len(k)
    f11 = np.empty((nz, nk))
    f22 = np.empty((nz, nk))
    f12 = np.empty((nz, nk))

    for i in range(nz):
        chi_i = float(chi_arr[i])
        Ra = chi_i * theta1
        Rb = chi_i * theta2
        if filter_type == "tophat":
            Wa = np.asarray(top_hat_filter_numpy(k, Ra))
            Wb = np.asarray(top_hat_filter_numpy(k, Rb))
        else:
            Wa = np.asarray(starlet_filter_numpy(k, Ra))
            Wb = np.asarray(starlet_filter_numpy(k, Rb))
        bw = k * dk / (2.0 * np.pi)
        f11[i] = bw * Wa ** 2
        f22[i] = bw * Wb ** 2
        f12[i] = bw * Wa * Wb

    # Mean σ values: σ_ab(z_i) = Σ_k f_ab(k, z_i) P_nl(k, z_i)
    sig11 = np.einsum("ik,ik->i", f11, Pnl)
    sig22 = np.einsum("ik,ik->i", f22, Pnl)
    sig12 = np.einsum("ik,ik->i", f12, Pnl)

    s_mean = np.empty(3 * nz)
    s_mean[0::3] = sig11
    s_mean[1::3] = sig22
    s_mean[2::3] = sig12

    # F[i, c, m]: c=0 → f11, c=1 → f22, c=2 → f12
    F = np.stack([f11, f22, f12], axis=1)  # (nz, 3, nk)

    C_sigma = np.zeros((3 * nz, 3 * nz))

    # --- Gaussian (block-diagonal in z) ---
    for i in range(nz):
        for ca in range(3):
            for cb in range(ca, 3):
                val = 2.0 * float(np.dot(F[i, ca] * F[i, cb], 1.0 / Nmodes))
                C_sigma[3 * i + ca, 3 * i + cb] += val
                if ca != cb:
                    C_sigma[3 * i + cb, 3 * i + ca] += val

    # --- Non-Gaussian i-trispectrum (block-diagonal in z) ---
    if use_ng:
        def _A_eff(k_):
            return 35.0 * (k_ / 0.1) ** 0.87 * (1.0 + (k_ / 1.0) ** 1.94) ** (-2.11)

        A = _A_eff(k)
        for i in range(nz):
            v = F[i] @ A  # (3,): v[c] = Σ_m F[i,c,m] A_eff(k_m)
            C_sigma[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += np.outer(v, v) / vol

    # --- SSC (rank-1 across all z-slices) ---
    if use_ssc:
        # g[i, c] = Σ_k f_c(k, z_i) R₁(k, z_i)
        g = np.einsum("icm,im->ic", F, R1_response)  # (nz, 3)
        g_flat = g.reshape(-1)                         # (3*nz,)
        C_sigma += sigma2_b * np.outer(g_flat, g_flat)

    return s_mean, C_sigma


def sample_sigma_as_pk(cosmo, z, s_mean, C_sigma, n_samples=100, seed=None):
    """
    Draw realizations from the σ-space covariance and return them as P(k) dicts.

    Each σ-space sample is mapped to a P(k) realization via a per-z amplitude
    rescaling::

        P_sample(k, z_i) = P_fid(k, z_i) × (σ₁₁_sample(z_i) / σ₁₁_fid(z_i))

    This is exact for SSC-type fluctuations (δP ∝ P_fid) and captures the
    dominant z-correlated variability mode efficiently. The output format is
    identical to ``get_covariance(..., variability=True)``.

    Parameters
    ----------
    cosmo : Cosmology_function
    z : array_like, shape (nz,)
    s_mean : ndarray, shape (3*nz,)
        From :func:`get_sigma_covariance`.
    C_sigma : ndarray, shape (3*nz, 3*nz)
        From :func:`get_sigma_covariance`.
    n_samples : int, optional
        Number of P(k) realizations (default 100).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    pk_samples_list : list of dict
        Length *n_samples*. Each dict maps ``z_i`` (float) → P(k) ndarray,
        same format as ``get_covariance`` with ``variability=True``.
    pk_dict : dict
        Fiducial mean P(k) at each z.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z, dtype=float)
    nz = len(z)
    k = np.asarray(cosmo.k, dtype=float)

    Pnl = np.vstack([cosmo.get_nonlinear_pk(z_i, k) for z_i in z])  # (nz, nk)
    pk_dict = {float(z_i): Pnl[i] for i, z_i in enumerate(z)}

    # Regularize for numerical stability
    jitter = 1e-10 * np.trace(C_sigma) / max(3 * nz, 1) * np.eye(3 * nz)
    s_samples = rng.multivariate_normal(s_mean, C_sigma + jitter, size=n_samples)

    sig11_fid = s_mean[0::3]  # (nz,) reference amplitude per z-slice
    safe_fid = np.where(sig11_fid > 0, sig11_fid, 1e-300)

    pk_samples_list = []
    for n in range(n_samples):
        sig11_n = s_samples[n, 0::3]  # (nz,)
        alpha = np.clip(sig11_n / safe_fid, 0.01, 100.0)
        pk_n = {float(z[i]): Pnl[i] * alpha[i] for i in range(nz)}
        pk_samples_list.append(pk_n)

    return pk_samples_list, pk_dict


def get_covariance(cosmo, z, variability, numberofrealisations):
    """
    Compute the nonlinear matter power spectrum P(k) and optionally its covariance
    using the i-trispectrum model from Gualdi et al. (2021) instead of a full halo model.

    Parameters
    ----------
    cosmo : Cosmology_function
        A Cosmology_function object that wraps pyccl and includes nonlinear P(k) access,
        k-grid, and other cosmological parameters.
    z : array_like
        Array of redshift values at which the power spectrum is evaluated.
    variability : bool
        Whether to compute and include non-Gaussian covariance and
        draw realizations of P(k) using a multivariate Gaussian distribution.
    numberofrealisations : int
        Number of mock realizations to draw for each redshift (if `variability=True`).

    Returns
    -------
    if variability is True:
        cov_dict : dict
            Dictionary mapping redshift z to full covariance matrix C(k, k').
        pk_samples_list : list
            List of length N. Each element is a dict mapping z to P(k) realization.
        pk_dict : dict
            Dictionary mapping redshift z to the mean nonlinear P(k).
    else:
        pk_dict : dict
            Dictionary mapping redshift z to the mean nonlinear P(k).
    """
    # SLICS box is 505 Mpc/h; cosmo.k is in Mpc⁻¹, so convert to Mpc.
    Lbox = 505.0 / cosmo.h  # Mpc  (505 Mpc/h → Mpc for consistent mode counting)
    vol = Lbox**3
    k = cosmo.k
    Nk = len(k)

    # Shell widths for the (now log-spaced) k grid: use centre-to-centre differences.
    k_edges = np.concatenate([[k[0]], 0.5 * (k[:-1] + k[1:]), [k[-1]]])
    dk_shells = np.diff(k_edges)
    Nmodes = vol / 3 / (2 * np.pi**2) * ((k + dk_shells / 2) ** 3 - (k - dk_shells / 2) ** 3)

    scale_factors = 1.0 / (1.0 + z)
    idx = np.argsort(scale_factors)
    scale_factors = scale_factors[idx]
    z = np.array(z)[idx]

    Pnl = cosmo.get_nonlinear_pk(z, k)
    na = len(z)

    # Gualdi et al. 2021 i-trispectrum model (z ~ 0.5)
    def A_eff(k):
        alpha = .35e2
        beta = .87
        gamma = 1.94
        delta = 2.11
        k0 = 0.1
        k1 = 1.
        return alpha * (k / k0) ** beta * (1 + (k / k1) ** gamma) ** (-delta)

    if variability:
        Cgauss = np.array([np.diag(2.0 * Pnl[i] ** 2 / Nmodes) for i in range(na)])
        cov = np.empty((na, Nk, Nk))
        A = A_eff(k)
        for i in range(na):
            # i-trispectrum approximation
            T_eff = np.outer(A * Pnl[i], A * Pnl[i])
            cov[i] = Cgauss[i] + T_eff / vol

        N = numberofrealisations
        pnl_samples = np.empty((na, N, Nk))
        pk_samples_list = []

        for i in range(na):
            mean_i = Pnl[i]
            cov_i = cov[i]
            cov_i += np.eye(Nk) * 1e-12 * np.trace(cov_i) / Nk
            pnl_samples[i] = np.random.multivariate_normal(mean_i, cov_i, size=N)
            
        # for i in range(na):
        #     plt.loglog(k, np.array(pnl_samples[i]).T,alpha=0.5)
        #     plt.loglog(k, Pnl[i], color='black', lw=2, label='Pnl')

        pk_dict = {z_: Pnl[i] for i, z_ in enumerate(z)}
        for n in range(N):
            pk_i = {z[j]: pnl_samples[j, n, :] for j in range(na)}
            pk_samples_list.append(pk_i)

    
        cov_dict = {z_: cov[i] for i, z_ in enumerate(z)}
        return cov_dict, pk_samples_list, pk_dict

    else:
        
        pk_dict = {z_: Pnl[i] for i, z_ in enumerate(z)}
        return pk_dict
