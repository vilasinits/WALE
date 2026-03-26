import numpy as np
from scipy.integrate import simpson
from astropy import units as u

from .FilterFunctions import *


def apply_pixel_window(ells, theta_deg=10.0, npix=1200):
    """
    Compute the pixel window function for a square map and apply it to multipoles.

    Parameters
    ----------
    ells : array_like
        Multipole values (ℓ) at which the window function is evaluated.
    theta_deg : float, optional
        Total angular size of the map in degrees (default is 10.0).
    npix : int, optional
        Number of pixels per side of the square map (default is 1200).

    Returns
    -------
    W_ell : ndarray
        The pixel window function evaluated at each ℓ.
    """
    theta_pix_rad = np.deg2rad(theta_deg / npix)
    arg = ells * theta_pix_rad / 2
    W_ell = np.sinc(arg / np.pi) ** 2

    return W_ell


def fourier_coordinate(x, y, map_size):
    """
    Return the 1D Fourier coordinate index corresponding to 2D (x, y) on a square map.

    Parameters
    ----------
    x : int
        X-coordinate (horizontal index).
    y : int
        Y-coordinate (vertical index).
    map_size : int
        Size of one side of the square map.

    Returns
    -------
    idx : int
        Flattened Fourier-space index.
    """
    return (((map_size // 2) + 1) * x) + y


def get_moments(kappa_values, pdf_values):
    """
    Compute the first four moments of a given 1D probability distribution.

    Parameters
    ----------
    kappa_values : ndarray
        Bin centers or sample points along the kappa (x-axis).
    pdf_values : ndarray
        Corresponding PDF values at each kappa.

    Returns
    -------
    mean_kappa : float
        Mean of the distribution.
    variance : float
        Variance of the distribution.
    S_3 : float
        Skewness (third standardized moment).
    K : float
        Kurtosis minus 3 (excess kurtosis).
    norm : float
        Normalization constant of the input PDF.
    """
    norm = np.trapezoid(pdf_values, kappa_values)
    normalized_pdf_values = pdf_values / norm
    mean_kappa = np.trapezoid(kappa_values * normalized_pdf_values, kappa_values)
    variance = np.trapezoid(
        (kappa_values - mean_kappa) ** 2 * normalized_pdf_values, kappa_values
    )
    third_moment = np.trapezoid(
        (kappa_values - mean_kappa) ** 3 * normalized_pdf_values, kappa_values
    )
    fourth_moment = np.trapezoid(
        (kappa_values - mean_kappa) ** 4 * normalized_pdf_values, kappa_values
    )
    S_3 = third_moment / (variance**2.0)
    K = fourth_moment / variance**2 - 3
    return mean_kappa, variance, S_3, K, norm


def get_l1_from_pdf(counts, bins):
    """
    Compute the L1 norm (∫|x|P(x)dx) from a histogram representation of a PDF.

    Parameters
    ----------
    counts : ndarray
        Histogram bin counts or PDF values (P(x)).
    bins : ndarray
        Bin centers or values corresponding to the counts.

    Returns
    -------
    l1_norm : ndarray
        L1 norm approximation (P(x) * |x| per bin).
    """
    return counts * np.abs(bins)


def get_l1_estimator_variance(
    pdf_values,
    kappa_values,
    A_survey_deg2,
    theta_arcmin,
):
    """
    Analytical variance of the L1-norm estimator from a finite survey.

    For a survey containing N_pix effectively independent pixels (resolution
    elements of size θ), the sample-mean L1 estimator has variance:

        Var(L1) = [⟨κ²⟩ − ⟨|κ|⟩²] / N_pix_eff

    where ⟨κ²⟩ and ⟨|κ|⟩ are computed from the supplied PDF.  This is the
    central-limit-theorem (Poisson) contribution and is the dominant source of
    L1 variability for weak-lensing surveys.

    Parameters
    ----------
    pdf_values : ndarray
        PDF values P(κ) (need not be normalised).
    kappa_values : ndarray
        Corresponding κ values (same units used in the LDT output).
    A_survey_deg2 : float
        Survey area in deg².
    theta_arcmin : float
        Angular scale of the smoothing filter in arcmin.  The number of
        independent pixels is estimated as A_survey / (π θ²).

    Returns
    -------
    l1_mean : float
        ⟨|κ|⟩ = ∫ |κ| P(κ) dκ  (the fiducial L1 prediction).
    l1_std : float
        √Var(L1) — the 1-σ scatter of the L1 estimator.
    n_pix_eff : float
        Effective number of independent pixels used.
    """
    norm = np.trapezoid(pdf_values, kappa_values)
    p = pdf_values / norm

    l1_mean  = float(np.trapezoid(np.abs(kappa_values) * p, kappa_values))
    kappa2   = float(np.trapezoid(kappa_values ** 2 * p, kappa_values))

    A_survey_arcmin2 = A_survey_deg2 * 3600.0          # deg² → arcmin²
    n_pix_eff = A_survey_arcmin2 / (np.pi * theta_arcmin ** 2)

    l1_var = max(kappa2 - l1_mean ** 2, 0.0) / n_pix_eff
    l1_std = float(np.sqrt(l1_var))

    return l1_mean, l1_std, n_pix_eff


def sample_l1_from_pdf(
    pdf_values,
    kappa_values,
    A_survey_deg2,
    theta_arcmin,
    n_maps=500,
    seed=None,
):
    """
    Monte-Carlo distribution of the L1-norm estimator over mock survey realisations.

    For each mock map, ``n_pix_eff`` κ values are drawn independently from the
    LDT PDF and the L1 estimator is computed.  The resulting array of L1 values
    gives the full sampling distribution (including non-Gaussian tails).

    Parameters
    ----------
    pdf_values : ndarray
    kappa_values : ndarray
    A_survey_deg2 : float
        Survey area in deg².
    theta_arcmin : float
        Filter scale in arcmin.
    n_maps : int, optional
        Number of mock maps to draw (default 500).
    seed : int or None, optional

    Returns
    -------
    l1_samples : ndarray, shape (n_maps,)
        L1 estimate for each mock map.
    n_pix_eff : int
        Number of pixels drawn per map.
    """
    rng = np.random.default_rng(seed)

    norm = np.trapezoid(pdf_values, kappa_values)
    p = np.maximum(pdf_values / norm, 0.0)
    p /= p.sum()   # discrete weights for np.random.choice

    A_survey_arcmin2 = A_survey_deg2 * 3600.0
    n_pix_eff = max(int(A_survey_arcmin2 / (np.pi * theta_arcmin ** 2)), 1)

    l1_samples = np.empty(n_maps)
    for i in range(n_maps):
        kappa_drawn = rng.choice(kappa_values, size=n_pix_eff, p=p)
        l1_samples[i] = float(np.mean(np.abs(kappa_drawn)))

    return l1_samples, n_pix_eff


def compute_sigma_kappa_squared(
    theta_arcmin, chis, lensingweights, redshifts, k, pnl, filter_type, h
):
    """
    σ^2_κ(θ) from Limber: P_κ(ℓ) = ∫ dχ [W(χ)/χ]^2 P(k=(ℓ+1/2)/χ, z(χ)).
    Inputs:
      - chis: comoving distances in Mpc
      - k: in 1/Mpc (PyCCL convention; P(k) in Mpc³)
      - pnl: dict mapping redshift -> P(k) array (len == len(k))
    """

    def _p_of_k_for_z(z):
        # exact match
        if z in pnl:
            return np.asarray(pnl[z], dtype=float)
        # float-key within tiny tol
        for kk in pnl.keys():
            try:
                if isinstance(kk, float) and abs(kk - z) < 1e-9:
                    return np.asarray(pnl[kk], dtype=float)
            except Exception:
                pass
        # try common string formats
        for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}", "{:.5f}"):
            key = fmt.format(z)
            if key in pnl:
                return np.asarray(pnl[key], dtype=float)
        # nearest-key fallback (robust if z-grid differs slightly)
        keys_float, key_map = [], []
        for kk in pnl.keys():
            try:
                keys_float.append(float(kk)); key_map.append(kk)
            except Exception:
                continue
        if keys_float:
            keys_float = np.asarray(keys_float)
            ksel = key_map[int(np.argmin(np.abs(keys_float - z)))]
            return np.asarray(pnl[ksel], dtype=float)
        raise KeyError(f"No pnl entry for z={z}")

    theta_rad = (theta_arcmin * u.arcmin).to(u.rad).value

    # ℓ grid focused where the filter has support
    ell_min = 2.0
    ell_max = 2e4 #min(5e6, 200.0 / max(theta_rad, 1e-6))
    # print(f"  Computing σ²_κ at θ={theta_arcmin:.2f} arcmin using ℓ in [{ell_min:.1f}, {ell_max:.1f}]")

    ell = np.logspace(np.log10(ell_min), np.log10(ell_max), 500)

    chis = np.asarray(chis, dtype=float)                # (n_chi,)
    lensingweights = np.asarray(lensingweights, float)  # (n_chi,)
    redshifts = np.asarray(redshifts, dtype=float)      # (n_chi,)
    k = np.asarray(k, dtype=float)                      # (n_k,)

    # Build P(k,z) array *from dict*, aligned to redshifts (your style)
    pnl_array = np.vstack([_p_of_k_for_z(z) for z in redshifts])  # (n_chi, n_k)
    if pnl_array.shape[1] != k.size:
        raise ValueError("Each pnl[z] must be 1D with length len(k).")

    # Limber k(ℓ,χ) in 1/Mpc. Improved Limber approximation (ℓ+1/2).
    k_l = (ell[:, None] + 0.5) / chis[None, :]  # (n_ell, n_chi)

    # Interpolate P(k,z) at each χ onto k_l
    pk_vals = np.empty_like(k_l)
    for j in range(chis.size):
        pk_vals[:, j] = np.interp(k_l[:, j], k, pnl_array[j], left=0.0, right=0.0)

    # Project to P_kappa(ℓ)
    W_over_chi_sq = (lensingweights / chis) ** 2
    integrand_chi = pk_vals * W_over_chi_sq[None, :]
    P_kappa = simpson(integrand_chi, x=chis, axis=1)  # (n_ell,)

    # Filter in ℓ-space (your choice)
    if filter_type.lower() == "tophat":
        Wl = top_hat_filter(ell, 2.0 * theta_rad) - top_hat_filter(ell, theta_rad)
    elif filter_type.lower() == "starlet":
        Wl = starlet_filter(ell, 2.0 * theta_rad) - starlet_filter(ell, theta_rad)
    else:
        raise ValueError("filter_type must be 'tophat' or 'starlet'.")

    # Apply pixel window if you have pixelization
    pixel_window = apply_pixel_window(ell, theta_deg=theta_rad * u.rad.to(u.deg))
    Wtot = Wl #* pixel_window

    # σ^2_κ(θ) = ∫ dℓ ℓ/(2π) P_κ(ℓ) |W(ℓθ)|^2
    sigma2 = simpson(ell * P_kappa * (Wtot ** 2), x=ell) / (2.0 * np.pi)
    return float(sigma2)



import pyccl as ccl
import pyccl.nl_pt as pt

# ---------- build PT P(k,z) with pyccl.nl_pt ----------
def build_pk2d_pt(cosmo, scheme="eulerian", with_IA=False,
                  log10k_min=-4, log10k_max=2, nk_per_decade=20):
    """
    Returns a ccl.Pk2D for matter×matter from CCL PT:
      - scheme='eulerian' -> FAST-PT (1-loop SPT/EFT kernels available)
      - scheme='lagrangian' -> velocileptors
    """
    if scheme == "eulerian":
        ptc = pt.EulerianPTCalculator(with_NC=True, with_IA=with_IA,
                                      log10k_min=log10k_min,
                                      log10k_max=log10k_max,
                                      nk_per_decade=nk_per_decade)
    elif scheme == "lagrangian":
        ptc = pt.LagrangianPTCalculator(log10k_min=log10k_min,
                                        log10k_max=log10k_max,
                                        nk_per_decade=nk_per_decade)
    else:
        raise ValueError("scheme must be 'eulerian' or 'lagrangian'")
    ptc.update_ingredients(cosmo)
    ptt_m = pt.PTMatterTracer()
    pk_mm = ptc.get_biased_pk2d(ptt_m, tracer2=ptt_m)  # ccl.Pk2D
    return pk_mm

# ---------- σ_κ^2(θ) using YOUR W_l and n(z) ----------
def sigma_kappa_var_from_ccl(theta_arcmin,
                             z, n_z,
                             cosmo,
                             pk2d_override=None,   
                             ell=None, ell_min=40, ell_max=None, n_ell=400,
                             normalize_nz=True,
                             extra_window=None,     # optional callable: A(ell) to multiply (pixel window, etc.)
                             filter_type="tophat"):
    """
    σ^2_κ(θ) = ∫ dℓ ℓ/(2π) C_ℓ^{κκ} |W_l(ℓ,θ)|^2, with C_ℓ computed by CCL.
    Uses your z, n_z and your Fourier-space window W_l.
    """
    z = np.asarray(z, float)
    n_z = np.asarray(n_z, float)
    if normalize_nz:
        nz_norm = simpson(n_z, x=z)
        if nz_norm <= 0:
            raise ValueError("n_z normalization is non-positive.")
        n_z = n_z / nz_norm

    # Tracer for weak lensing with your n(z)
    t_l = ccl.WeakLensingTracer(cosmo, dndz=(z, n_z))

    theta_rad = (theta_arcmin / 60.0) * np.pi / 180.0

    # ℓ sampling: use yours if provided; else build a sensible grid from θ
    if ell is None:
        if ell_max is None:
            ell_max = int(min(1e3, 50.0 / max(theta_rad, 1e-6)))  # heuristic; adjust if your W_l has longer tails
        ell = np.logspace(np.log10(max(ell_min, 1.0)), np.log10(ell_max), n_ell)
    else:
        ell = np.asarray(ell, float)

    # Angular power with PT override if provided
    C_ell = ccl.angular_cl(cosmo, t_l, t_l, ell,
                           p_of_k_a=pk2d_override,  # None -> uses cosmo's matter_power_spectrum setting
                           l_limber=-1)             # Limber for all ℓ (good for lensing)

    # Filter in ℓ-space (your choice)
    if filter_type.lower() == "tophat":
        Wl = top_hat_filter(ell, 2.0 * theta_rad) - top_hat_filter(ell, theta_rad)
    elif filter_type.lower() == "starlet":
        Wl = starlet_filter(ell, 2.0 * theta_rad) - starlet_filter(ell, theta_rad)
    else:
        raise ValueError("filter_type must be 'tophat' or 'starlet'.")
    
    if extra_window is not None:
        Wl = Wl # * extra_window(ell)

    # σ^2_κ
    sigma2 = simpson(ell * C_ell * (Wl**2), x=ell) / (2.0 * np.pi)
    return float(sigma2)


def get_l1_ssc_variance(
    pdf_values,
    kappa_values,
    sigmasq_kappa,
    A_survey_deg2,
    chis,
    lensingweights,
    redshifts,
    k,
    pnl,
    filter_type="tophat",
    h=0.681,
    bias="gaussian",
):
    """
    Super-sample covariance (SSC) contribution to the L1-norm variance.

    From Uhlemann et al. 2022 (arXiv:2210.07819), Eq. (13a) applied to L1:

        Var_SSC(L1) = ξ̄ × [∫ |κ| b₁(κ) P(κ) dκ]²

    where:
      - ξ̄ = σ²_κ(θ_survey)  — variance of the mean convergence over the survey
        (variance of the background mode δ_b averaged over the survey patch)
        computed at the effective survey angular scale θ_survey = sqrt(A/π).
      - b₁(κ) = κ / σ²_κ    — Gaussian linear bias (response of the one-point
        PDF to a uniform background shift δ_b; Uhlemann+2022 Eq. 22).
      - P(κ) is the fiducial one-point PDF.

    The physical picture: different survey realizations probe different background
    modes. A positive δ_b shifts the whole κ-field upward → the PDF shifts right
    → L1 changes coherently. ξ̄ measures how much the background mode varies
    between realizations of a survey of area A.

    Parameters
    ----------
    pdf_values : ndarray
        Fiducial PDF P(κ), need not be normalised.
    kappa_values : ndarray
        κ grid matching pdf_values.
    sigmasq_kappa : float
        σ²_κ at the filter scale θ (the LDT/PT variance, not the survey scale).
        Used to normalise b₁(κ) = κ / σ²_κ.
    A_survey_deg2 : float
        Survey area in deg².
    chis : ndarray
        Comoving distances of lens planes (Mpc).
    lensingweights : ndarray
        Lensing weight W(χ) at each chi.
    redshifts : ndarray
        Redshifts at each chi plane.
    k : ndarray
        Wavenumber grid in 1/Mpc.
    pnl : dict
        {z: P(k)} nonlinear power spectrum dict.
    filter_type : str
        'tophat' or 'starlet'.
    h : float
        Dimensionless Hubble parameter (used only to label units).
    bias : str
        'gaussian'  → b₁(κ) = κ / σ²_κ  (Uhlemann+2022 Eq. 22)
        (lognormal support can be added later)

    Returns
    -------
    l1_mean : float
        ⟨|κ|⟩ — fiducial L1 value.
    l1_std_ssc : float
        √Var_SSC(L1) — SSC contribution to L1 scatter.
    xi_bar : float
        ξ̄ = σ²_κ(θ_survey) — the mean convergence variance over the survey.
    A_ssc : float
        The SSC amplitude ∫|κ| b₁(κ) P(κ) dκ = ⟨κ|κ|⟩ / σ²_κ.
    """
    # Normalise PDF
    norm = np.trapezoid(pdf_values, kappa_values)
    p = pdf_values / norm

    # L1 mean
    l1_mean = float(np.trapezoid(np.abs(kappa_values) * p, kappa_values))

    # Gaussian bias: b₁(κ) = κ / σ²_κ
    b1 = kappa_values / sigmasq_kappa

    # SSC amplitude A = ∫ |κ| b₁(κ) P(κ) dκ = ⟨κ|κ|⟩ / σ²_κ
    A_ssc = float(np.trapezoid(np.abs(kappa_values) * b1 * p, kappa_values))

    # Survey angular scale: θ_survey = sqrt(A/π) in arcmin
    A_survey_arcmin2 = A_survey_deg2 * 3600.0
    theta_survey_arcmin = float(np.sqrt(A_survey_arcmin2 / np.pi))

    # ξ̄ = σ²_κ(θ_survey) — convergence variance at the survey scale
    xi_bar = float(compute_sigma_kappa_squared(
        theta_survey_arcmin,
        chis,
        lensingweights,
        redshifts,
        k,
        pnl,
        filter_type=filter_type,
        h=h,
    ))

    # Var_SSC(L1) = ξ̄ × A²
    l1_var_ssc = xi_bar * A_ssc ** 2
    l1_std_ssc = float(np.sqrt(max(l1_var_ssc, 0.0)))

    return l1_mean, l1_std_ssc, xi_bar, A_ssc


def sample_l1_ssc(
    pdf_values,
    kappa_values,
    xi_bar,
    n_samples=500,
    seed=None,
):
    """
    Generate L1-norm samples driven by SSC (background convergence mode).

    Physical picture (Uhlemann+2022 §2.2):
    A background convergence mode δ_b coherent over the entire survey shifts
    every pixel's κ by the same amount.  This is the dominant source of
    realisation-to-realisation scatter in small-to-medium surveys.

    Sampling procedure:
        δ_b  ~  N(0, ξ̄)                    (background amplitude)
        P_n(κ) = P_fid(κ − δ_b)             (pure shift of the PDF)
        L1_n   = ∫ |κ| P_n(κ) dκ
               = ∫ |κ + δ_b| P_fid(κ) dκ

    By construction this gives:
        Var(L1_samples) → ξ̄ × [∫ sign(κ) P(κ) dκ]²   (linear in δ_b)

    which is exactly the SSC prediction from ``get_l1_ssc_variance`` at
    linear order.  The PDF-shift approach is exact for the SSC mechanism
    without requiring a full LDT re-run per sample.

    Parameters
    ----------
    pdf_values : ndarray
        Fiducial PDF P(κ), need not be normalised.
    kappa_values : ndarray
        κ grid matching pdf_values.
    xi_bar : float
        ξ̄ = σ²_κ(θ_survey) — background mode variance, from
        ``get_l1_ssc_variance`` or ``compute_sigma_kappa_squared`` at the
        survey angular scale.
    n_samples : int
        Number of L1 realisations to draw.
    seed : int or None
        Random seed.

    Returns
    -------
    l1_samples : ndarray, shape (n_samples,)
        L1-norm values for each background-mode realisation.
    delta_b_samples : ndarray, shape (n_samples,)
        The drawn background convergence amplitudes δ_b.
    """
    rng = np.random.default_rng(seed)
    norm = np.trapezoid(pdf_values, kappa_values)
    p = pdf_values / norm

    delta_b_samples = rng.normal(0.0, np.sqrt(float(xi_bar)), size=n_samples)

    l1_samples = np.array([
        float(np.trapezoid(np.abs(kappa_values + db) * p, kappa_values))
        for db in delta_b_samples
    ])
    return l1_samples, delta_b_samples


def sample_l1_combined(
    pdf_values,
    kappa_values,
    sigmasq_kappa,
    xi_bar,
    A_survey_deg2,
    theta_arcmin,
    n_samples=500,
    seed=None,
):
    """
    Generate L1-norm samples including BOTH shot noise and SSC.

    Shot noise:  sample N_pix κ values from P(κ) per mock → mean |κ|
    SSC:         add a coherent background shift δ_b ~ N(0, ξ̄)

    The two are independent so the total variance is additive:
        Var_total(L1) = Var_shot + Var_SSC

    Parameters
    ----------
    pdf_values, kappa_values : ndarray
        Fiducial PDF (need not be normalised).
    sigmasq_kappa : float
        σ²_κ at the filter scale (used only for reference; not needed for sampling).
    xi_bar : float
        ξ̄ — background mode variance from ``get_l1_ssc_variance``.
    A_survey_deg2 : float
        Survey area in deg².
    theta_arcmin : float
        Filter scale in arcmin.  N_pix = A / (π θ²).
    n_samples : int
        Number of mock survey realisations.
    seed : int or None

    Returns
    -------
    l1_samples : ndarray, shape (n_samples,)
        Combined (shot + SSC) L1 samples.
    l1_shot_only : ndarray, shape (n_samples,)
        Shot-noise-only L1 samples (δ_b = 0).
    l1_ssc_only : ndarray, shape (n_samples,)
        SSC-only L1 samples (N_pix → ∞, just the δ_b shift).
    """
    rng = np.random.default_rng(seed)
    norm = np.trapezoid(pdf_values, kappa_values)
    p = pdf_values / norm

    A_arcmin2 = A_survey_deg2 * 3600.0
    n_pix = int(A_arcmin2 / (np.pi * theta_arcmin ** 2))

    # SSC background shifts
    delta_b = rng.normal(0.0, np.sqrt(float(xi_bar)), size=n_samples)

    l1_combined  = np.empty(n_samples)
    l1_shot_only = np.empty(n_samples)
    l1_ssc_only  = np.empty(n_samples)

    kappa_grid = np.asarray(kappa_values)
    dk = kappa_grid[1] - kappa_grid[0]   # assume uniform spacing
    cdf = np.cumsum(p) * dk
    cdf /= cdf[-1]

    from scipy.interpolate import interp1d
    inv_cdf = interp1d(cdf, kappa_grid, bounds_error=False,
                       fill_value=(kappa_grid[0], kappa_grid[-1]))

    for n in range(n_samples):
        # Shot noise: sample N_pix kappas from P(kappa)
        u = rng.uniform(size=n_pix)
        kappa_drawn = inv_cdf(u)
        l1_shot = float(np.mean(np.abs(kappa_drawn)))

        db = delta_b[n]
        # SSC: uniform shift of drawn kappas
        l1_combined[n]  = float(np.mean(np.abs(kappa_drawn + db)))
        l1_shot_only[n] = l1_shot
        l1_ssc_only[n]  = float(np.trapezoid(np.abs(kappa_grid + db) * p, kappa_grid))

    return l1_combined, l1_shot_only, l1_ssc_only
