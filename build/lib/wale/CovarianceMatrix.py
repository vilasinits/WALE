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
    Lbox = 505  # Mpc/h
    vol = Lbox**3
    k = cosmo.k
    dk = cosmo.dk
    Nk = len(k)

    Nmodes = (
        vol
        / 3
        / (2 * np.pi**2)
        * ((k + dk / 2) ** 3 - (k - dk / 2) ** 3)
    )

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
