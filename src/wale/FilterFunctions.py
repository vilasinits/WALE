# from imports import *
import numpy as np

# import scipy.special
from scipy import special as sp
from functools import lru_cache
import mpmath as mp


def top_hat_filter(k, R):
    """
    Calculates the top-hat window function for a given radius.

    Parameters:
        R (float or numpy.ndarray): The scale (or array of scales) at which to calculate the window function.

    Returns:
        numpy.ndarray: The top-hat window function values at the given scale(s).
    """
    return 2.0 * sp.j1(k * R) / (k * R)


# def top_hat_window(R):
#     """
#     Calculates the top-hat window function for a given radius.

#     Parameters:
#         R (float or numpy.ndarray): The scale (or array of scales) at which to calculate the window function.

#     Returns:
#         numpy.ndarray: The top-hat window function values at the given scale(s).
#     """
#     return 2.0 * sp.j1(R) / R


def get_W2D_FL(window_radius, map_shape, filter_type, L=505):
    """
    Constructs a 2D Fourier-space window function for a top-hat filter.

    Parameters:
        window_radius : float
            The top-hat window radius in physical units (must be consistent with L).
        map_shape     : tuple
            Shape of the map (assumed square, e.g. (600,600)).
        L             : float, optional
            Physical size of the map (default is 505, as used for SLICS).

    Returns:
        2D numpy array representing the Fourier-space window.
    """
    N = map_shape[0]
    dx = N / N
    # Generate Fourier frequencies.
    kx = np.fft.fftshift(np.fft.fftfreq(N, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(N, dx))
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx**2 + ky**2
    # Convert to radial wavenumber (with 2pi factor).
    k = 2 * np.pi * np.sqrt(k2)
    # Avoid division by zero at the center.
    ind = int(N / 2)
    k[ind, ind] = 1e-7
    if filter_type == "tophat":
        return top_hat_filter(k, window_radius)
    elif filter_type == "starlet":
        # print("Getting starlet W2D_FL")
        return starlet_filter(k, window_radius)
        # return uHat_starlet_analytical(k, window_radius)


def b3_1D_ft(x):
    return (np.sin(x/2)/(x/2))**4. 
def b3_2D_ft(x,y):
    return b3_1D_ft(x)*b3_1D_ft(y)

def starlet_filter(k, R):
    """
    Computes the Fourier-space starlet filter.

    Args:
        k (np.ndarray): 2D array of Fourier frequencies.
        R (float): The scale at which to compute the filter.

    Returns:
        np.ndarray: The computed starlet filter in Fourier space.
    """
    # Calculate the radial frequency
    # k_radial = np.sqrt(k**2)
    # Compute the starlet filter
    return b3_2D_ft(k * R, k * R)

# Fast memoized scalar S function
@lru_cache(maxsize=None)
def S_scalar(n: int, b: float) -> float:
    if n < -1:
        raise ValueError("n cannot be smaller than -1.")

    J0 = sp.j0(b)
    J1 = sp.j1(b)

    if n == 0:
        return b * J1
    elif n == -1:
        return b * float(mp.hyp1f2(0.5, 1, 1.5, -(b**2) / 4))
    else:
        return b ** (n + 1) * J1 + n * b**n * J0 - n**2 * S_scalar(n - 2, b)


# Wrapper to handle arrays
def S(n: int, b):
    b = np.asarray(b)
    if b.ndim == 0:
        return S_scalar(n, float(b))
    else:
        vec_func = np.vectorize(lambda x: S_scalar(n, float(x)))
        return vec_func(b)


# Fast uHat_starlet_analytical
def uHat_starlet_analytical(eta, R):
    """
    Computes the analytical Hankel transform of the starlet U-filter.

    Args:
        eta (np.ndarray or float): Dimensionless argument \( \hat{u} \).

    Returns:
        float or np.ndarray: Computed \( \hat{u} \).
    """
    # print("Calculating uHat_starlet_analytical (optimized version)")

    eta = np.asarray(eta) * R
    eta_safe = np.clip(eta, 2e-2, 100)  # Stability for small eta

    # Precompute all needed S values
    b_half = 0.5 * eta_safe
    b_one = eta_safe
    b_two = 2.0 * eta_safe

    S0_half = S(0, b_half)
    S1_half = S(1, b_half)
    S2_half = S(2, b_half)
    S3_half = S(3, b_half)

    S0_one = S(0, b_one)
    S1_one = S(1, b_one)
    S2_one = S(2, b_one)
    S3_one = S(3, b_one)

    S0_two = S(0, b_two)
    S1_two = S(1, b_two)
    S2_two = S(2, b_two)
    S3_two = S(3, b_two)

    # Compute factors
    factor1 = (
        0.125 * eta_safe**3 * S0_half
        - 0.75 * eta_safe**2 * S1_half
        + 1.5 * eta_safe * S2_half
        - S3_half
    )
    # print("done factor1")
    factor2 = (
        eta_safe**3 * S0_one - 3 * eta_safe**2 * S1_one + 3 * eta_safe * S2_one - S3_one
    )
    # print("done factor2")
    factor3 = (
        8 * eta_safe**3 * S0_two
        - 12 * eta_safe**2 * S1_two
        + 6 * eta_safe * S2_two
        - S3_two
    )
    # print("done factor3")
    # Final result
    result = (
        (2 * np.pi) * (-128 / 9 * factor1 + 4 * factor2 - 1 / 9 * factor3) / eta_safe**5
    )

    return result
