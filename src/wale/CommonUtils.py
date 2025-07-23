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
    norm = np.trapz(pdf_values, kappa_values)
    normalized_pdf_values = pdf_values / norm
    mean_kappa = np.trapz(kappa_values * normalized_pdf_values, kappa_values)
    variance = np.trapz(
        (kappa_values - mean_kappa) ** 2 * normalized_pdf_values, kappa_values
    )
    third_moment = np.trapz(
        (kappa_values - mean_kappa) ** 3 * normalized_pdf_values, kappa_values
    )
    fourth_moment = np.trapz(
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


def compute_sigma_kappa_squared(
    theta_arcmin, chis, lensingweights, redshifts, k, pnl, filter_type, h
):
    """
    Compute the smoothed convergence variance σ²_κ(θ) at a given angular scale using a filter.

    This function computes the convergence power spectrum P_κ(ℓ) from a 3D P(k, z)
    and integrates over ℓ using a top-hat or starlet filter.

    Parameters
    ----------
    theta_arcmin : float
        Angular smoothing scale θ in arcminutes.
    chis : ndarray
        Comoving distances χ (in Mpc) corresponding to redshifts.
    lensingweights : ndarray
        Lensing kernel W(χ) evaluated at each χ.
    redshifts : ndarray
        Redshifts corresponding to chis.
    k : ndarray
        Wavenumber grid (in h/Mpc).
    pnl : 2D ndarray
        Nonlinear power spectrum P(k, z), shape (n_z, len(k)).
    filter_type : str
        Type of filter to apply ("tophat" or "starlet").
    h : float
        Reduced Hubble constant (H0 / 100).

    Returns
    -------
    sigma2 : float
        Smoothed convergence variance σ²_κ(θ).
    """
    theta_rad = (theta_arcmin * u.arcmin).to(u.rad).value
    ell = np.logspace(1, 5, 200)

    P_kappa = np.zeros_like(ell)
    pnl_array = np.array([pnl[z_] for z_ in redshifts])  # shape: (n_chi, nk)

    # plt.loglog(k, pnl_array.T)
    # plt.show()

    for i, l in enumerate(ell):

        chis_h_inv = chis * h  # now in h⁻¹ Mpc
        k_l = l / chis_h_inv
        # Interpolate P(k) at each redshift slice
        pk_vals = np.array(
            [
                np.interp(k_l[j], k, pnl_array[j], left=0, right=0)
                for j in range(len(chis))
            ]
        )
        integrand = (lensingweights / chis) ** 2 * pk_vals
        P_kappa[i] = simpson(integrand, chis)

    # Apply top-hat filter window in Fourier space
    if filter_type == "tophat":
        W = top_hat_filter(ell, 2.0 * theta_rad) - top_hat_filter(ell, theta_rad)
    elif filter_type == "starlet":
        W = starlet_filter(ell, 2.0 * theta_rad) - starlet_filter(ell, theta_rad)

    pixel_window = apply_pixel_window(ell, theta_deg=theta_rad * u.rad.to(u.deg))
    integrand = ell * P_kappa * (W**2)
    sigma2 = simpson(integrand, ell) / (2.0 * np.pi)

    return sigma2
