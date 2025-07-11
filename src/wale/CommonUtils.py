import numpy as np
from scipy.integrate import simpson
from astropy import units as u

from .FilterFunctions import *


def apply_pixel_window(ells, theta_deg=10.0, npix=1200):
    """
    Apply pixel window function to theoretical Cls.

    Parameters:
    - cls: array of C_ell values (same length as ells)
    - ells: array of multipoles (ell values)
    - theta_deg: total angular size of the map (in degrees)
    - npix: number of pixels on one side of the square map

    Returns:
    - cls_smoothed: Cls multiplied by the pixel window function
    """
    theta_pix_rad = np.deg2rad(theta_deg / npix)
    arg = ells * theta_pix_rad / 2
    W_ell = np.sinc(arg / np.pi) ** 2

    return W_ell


def fourier_coordinate(x, y, map_size):
    return (((map_size // 2) + 1) * x) + y


def get_moments(kappa_values, pdf_values):
    """
    Calculates the moments (mean, variance, skewness, kurtosis) of a probability distribution function.

    Parameters:
        kappa_values (numpy.ndarray): A 1D array of kappa values.
        pdf_values (numpy.ndarray): A 1D array of PDF values corresponding to `kappa_values`.

    Returns:
        tuple: Contains mean, variance, skewness, kurtosis, and normalization of the PDF.
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
    Calculates the L1 norm from a probability distribution function represented as a histogram.

    Parameters:
        counts (numpy.ndarray): The counts or heights of the histogram bins.
        bins (numpy.ndarray): The values of the bins.

    Returns:
        numpy.ndarray: L1 norm of the PDF represented by the histogram.
    """
    return counts * np.abs(bins)

def compute_sigma_kappa_squared(theta_arcmin, chis, lensingweights, redshifts, k, pnl, filter_type,h):
    """
    Compute smoothed variance σ²_κ(θ) for a top-hat filter at angular scale θ (arcmin).
    
    Parameters:
    - theta_arcmin : float
    - chis : array of comoving distances (shape n)
    - lensingweights : array of W(chi) (shape n)
    - redshifts : array corresponding to chis (shape n)
    - k : array of wavenumbers (shape nk)
    - pnl : 2D array of shape (n, nk), i.e., P(k, z)
    
    Returns:
    - sigma²_κ(θ)
    """
    theta_rad = (theta_arcmin * u.arcmin).to(u.rad).value
    ell = np.logspace(1,5,200)
    
    P_kappa = np.zeros_like(ell)
    pnl_array = np.array([pnl[z_] for z_ in redshifts])  # shape: (n_chi, nk)
    
    # plt.loglog(k, pnl_array.T)
    # plt.show()
    
    for i, l in enumerate(ell):
        
        chis_h_inv = chis * h          # now in h⁻¹ Mpc
        k_l     =l / chis_h_inv  
        # Interpolate P(k) at each redshift slice
        pk_vals = np.array([
            np.interp(k_l[j], k, pnl_array[j], left=0, right=0)
            for j in range(len(chis))
        ])
        integrand = (lensingweights / chis)**2 * pk_vals
        P_kappa[i] = simpson(integrand, chis)

    # Apply top-hat filter window in Fourier space
    if filter_type == 'tophat':
        W = top_hat_filter(ell , 2. * theta_rad) - top_hat_filter(ell , theta_rad)
    elif filter_type == 'starlet':
        W = starlet_filter(ell , 2. * theta_rad) - starlet_filter(ell , theta_rad)
    
    pixel_window = apply_pixel_window(ell, theta_deg=theta_rad * u.rad.to(u.deg)) 
    integrand = ell * P_kappa * (W**2)
    sigma2 = simpson(integrand, ell) / (2. * np.pi)
    
    return sigma2
