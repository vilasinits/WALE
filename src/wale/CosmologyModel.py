import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import pyccl as ccl
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from pyccl.halos.pk_4pt import halomod_Tk3D_4h


class Cosmology_function:
    """
    A class for initializing and handling cosmological computations using PyCCL.

    This class encapsulates key cosmological parameters and provides methods to compute
    quantities such as the Hubble parameter, comoving distances, redshift inversions,
    non-linear matter power spectrum, and lensing weights. Internally, it constructs
    a `pyccl.Cosmology` object for use with the Core Cosmology Library (CCL).
    """

    def __init__(self, h, Oc, Ob, w=-1.0, wa=0.0, **kwargs):
        """
        Initialize a cosmological model with the given parameters.

        Parameters
        ----------
        h : float
            Dimensionless Hubble parameter (i.e., H0 = 100 * h km/s/Mpc).
        Oc : float
            Cold dark matter density fraction, Ω_c.
        Ob : float
            Baryon density fraction, Ω_b.
        w : float, optional
            Dark energy equation-of-state parameter at present (default is -1.0).
        wa : float, optional
            Evolution parameter of dark energy equation-of-state (default is 0.0).
        **kwargs : dict, optional
            Additional optional parameters:
            - 'sigma8' : float, RMS density fluctuations at 8 Mpc/h.
            - 'As' : float, amplitude of primordial scalar fluctuations.
            - 'ns' : float, scalar spectral index (default is 0.973).
            - 'kmin' : float, minimum wavenumber for calculations (default is 1e-4).
            - 'kmax' : float, maximum wavenumber (default is 10.0).
            - 'dk'   : float, step size for k-array (default is 0.1).

        Raises
        ------
        ValueError
            If neither 'sigma8' nor 'As' is specified.
        """
        self.h = h
        self.Oc = Oc
        self.Ob = Ob
        self.w = w
        self.wa = wa
        self.H0 = 100.0 * h  # Hubble constant in km/s/Mpc
        self.Om = Oc + Ob
        self.ns = kwargs.get("ns", 0.973)  # scalar spectral index

        # power normalization (either sigma8 or As must be set)
        self.sig8 = kwargs.get("sigma8", None)
        self.As = kwargs.get("As", None)
        if self.sig8 is None and self.As is None:
            raise ValueError("Please specify either sigma8 or As.")

        self.speed_light = 299792.458

        self.kmin = kwargs.get("kmin", 1e-4)
        self.kmax = kwargs.get("kmax", 10.0)
        self.dk = kwargs.get("dk", 0.1)
        self.k = np.arange(self.kmin, self.kmax, self.dk)
        self.nk = len(self.k)

        def _set_params(self):
            if self.sig8 is not None:
                return ccl.Cosmology(
                    h=self.h,
                    Omega_c=self.Oc,
                    Omega_b=self.Ob,
                    sigma8=self.sig8,
                    n_s=self.ns,
                    w0=self.w,
                    wa=self.wa,
                    transfer_function="boltzmann_camb",
                )
            elif self.As is not None:
                return ccl.Cosmology(
                    h=self.h,
                    Omega_c=self.Oc,
                    Omega_b=self.Ob,
                    A_s=self.As,
                    n_s=self.ns,
                    w0=self.w,
                    wa=self.wa,
                    transfer_function="boltzmann_camb",
                )

        self.cosmoccl = _set_params(self)

    def _E(self, z):
        """
        Compute the dimensionless Hubble expansion rate E(z) = H(z)/H0.

        This method uses the CPL (Chevallier-Polarski-Linder) parameterization for
        the dark energy equation of state:
        w(a) = w0 + wa * (1 - a),
        where a = 1 / (1 + z) is the scale factor.

        The expansion rate is given by:
            E(z)^2 = Ω_m (1 + z)^3 + Ω_k (1 + z)^2 + Ω_de (1 + z)^{3(1 + w0 + wa)} exp[-3wa z / (1 + z)]

        Parameters
        ----------
        z : float or array_like
            Redshift value(s) at which to compute the dimensionless expansion rate.

        Returns
        -------
        E : float or ndarray
            The dimensionless Hubble parameter E(z) = H(z)/H0.

        Notes
        -----
        - Assumes a flat or curved universe with Ω_k = 0 by default.
        - This function does not use the pyccl Cosmology object and computes E(z) analytically.
        """
        self.Ok = 0.0
        self.Ode = 1.0 - self.Om - self.Ok
        # CPL parameterization for w(a)=w0 + wa(1−a)
        return np.sqrt(
            self.Om * (1 + z) ** 3
            + self.Ok * (1 + z) ** 2
            + self.Ode
            * (1 + z) ** (3 * (1 + self.w + self.wa))
            * np.exp(-3 * self.wa * z / (1 + z))
        )

    def get_H(self, z):
        """
        Compute the Hubble parameter H(z) in km/s/Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate H(z).

        Returns
        -------
        H : float or ndarray
            Hubble parameter at redshift z, in units of km/s/Mpc.
        """
        return self.H0 * self._E(z)

    def get_chi(self, z):
        """
        Compute the comoving radial distance χ(z) in Mpc.

        This function numerically integrates the inverse of the dimensionless
        Hubble parameter E(z) = H(z)/H0 to compute:

            χ(z) = ∫₀ᶻ (c / H(z')) dz'

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the comoving distance.

        Returns
        -------
        chi : float or ndarray
            Comoving radial distance(s) to redshift z, in units of Mpc.

        Notes
        -----
        - Uses `scipy.integrate.quad` for numerical integration.
        - Assumes a flat universe with constant c = 299792.458 km/s.
        - The integration is done individually for each redshift if an array is given.
        """
        H0 = self.H0  # km/s/Mpc
        c = self.speed_light  # speed of light in km/s

        def integrand(zp):
            return 1.0 / self._E(zp)

        # scalar case
        if np.isscalar(z):
            integral, _ = quad(integrand, 0.0, z)
            return (c / H0) * integral

        # array case
        z = np.asanyarray(z)
        chi = np.empty_like(z, dtype=float)
        for i, zi in enumerate(z):
            integral, _ = quad(integrand, 0.0, zi)
            chi[i] = (c / H0) * integral
        return chi

    def get_z_from_chi(self, chi_target, z_max=10.0, tol=1e-8):
        """
        Invert χ(z) to find z such that χ(z)=chi_target.

        Parameters
        ----------
        chi_target : float or array_like
            Comoving distance(s) in Mpc.
        z_max : float
            Maximum redshift to search within (must be large enough that χ(z_max) > chi_target).
        tol : float
            Desired precision in z.

        Returns
        -------
        z : float or ndarray
            Redshift(s) corresponding to chi_target.
        """

        # root function for a given target χ
        def f(z, chi_t):
            return self.get_chi(z) - chi_t

        def find_root(chi_t):
            # ensure bracket: χ(0)=0, χ(z_max)>chi_t
            chi_max = self.get_chi(z_max)
            if chi_t < 0 or chi_t > chi_max:
                raise ValueError(
                    f"chi_target={chi_t:.3f} outside [0, {chi_max:.3f}] Mpc (z_max={z_max})"
                )
            return brentq(f, 0.0, z_max, args=(chi_t,), xtol=tol)

        if np.isscalar(chi_target):
            return find_root(chi_target)

        chi_arr = np.asanyarray(chi_target)
        z_arr = np.empty_like(chi_arr, dtype=float)
        for i, chi_t in enumerate(chi_arr):
            z_arr[i] = find_root(chi_t)
        return z_arr

    def get_nonlinear_pk(self, z, ks=None):
        """
        Compute the non-linear matter power spectrum P(k, z) using HALOFIT.

        Parameters
        ----------
        z : float
            Redshift at which to evaluate the power spectrum.
        ks : array_like, optional
            Wavenumber values in h/Mpc. If None, uses the default self.k array.

        Returns
        -------
        Pnl : ndarray
            Non-linear matter power spectrum at redshift z for the given ks, in (Mpc/h)^3.
        """
        # default k-grid
        if ks is None:
            ks = self.k  # np.logspace(np.log10(self.kmin),
            #  np.log10(self.kmax),
            #  self.nk)
        else:
            ks = np.atleast_1d(ks)
        a = 1.0 / (1.0 + z)
        # compute HALOFIT non-linear power
        Pnl = ccl.nonlin_matter_power(self.cosmoccl, self.k, a)
        return Pnl

    def get_lensing_weight_array(self, chis, chi_source):
        """
        Compute lensing weight array W(chi) for a single source plane at chi_source.

        Parameters
        ----------
        chis : array_like
            Comoving radial distances (chi) at which to compute lensing weights.
        chi_source : float
            Comoving distance to the source plane (single redshift).

        Returns
        -------
        z_values : ndarray
            Redshift values corresponding to chis, computed via inversion.
        lensing_weight : ndarray
            Lensing weight W(chi) evaluated at each input chi.
        """
        z_values = self.get_z_from_chi(chis)
        lensing_weight = np.zeros_like(chis)
        for i in range(len(chis)):
            z = z_values[i]
            lensing_weight[i] = (
                1.5
                * self.Om
                * (self.speed_light**-2.0)
                * ((self.H0) ** 2.0)
                * chis[i]
                * (1 - (chis[i] / chi_source))
                * (1 + z)
            )
        return z_values, lensing_weight

    def get_lensing_weight_array_nz(self, chis, z_nz, n_z):
        """
        Compute lensing weight W(chi) using a redshift distribution n(z),
        without relying on pyccl.

        Parameters
        ----------
        chis : array_like
            Comoving radial distances (chi) at which to compute lensing weights.
        z_nz : array_like
            Redshift values at which the distribution n(z) is defined.
        n_z : array_like
            Source redshift distribution n(z), not necessarily normalized.

        Returns
        -------
        z_values : ndarray
            Redshift values corresponding to the input chis, estimated via interpolation.
        lensing_weight : ndarray
            Lensing weight values W(chi) computed using the n(z) distribution.
        """

        n_norm = n_z / trapezoid(n_z, z_nz)  # normalized n(z)

        # Get chi(z) using self.get_chi
        chi_nz = self.get_chi(z_nz)  # in Mpc
        z_of_chi_interp = interp1d(chi_nz, z_nz, bounds_error=False, fill_value=0.0)

        # Convert n(z) → n(chi)
        dz_dchi = np.gradient(z_nz, chi_nz)
        n_chi = n_norm * dz_dchi
        n_chi_interp = interp1d(chi_nz, n_chi, bounds_error=False, fill_value=0.0)

        # Prepare lensing weights
        lensing_weight = np.zeros_like(chis)
        z_values = z_of_chi_interp(chis)

        for i, chi in enumerate(chis):
            # Integrate over chi' > chi
            chi_prime = chi_nz[chi_nz > chi]
            if chi_prime.size == 0:
                lensing_weight[i] = 0.0
                continue

            integrand = n_chi_interp(chi_prime) * (chi_prime - chi) / chi_prime
            integral = trapezoid(integrand, chi_prime)

            lensing_weight[i] = (
                (1.5 * self.Om * (self.H0) ** 2 / self.speed_light**2)
                * chi
                * (1 + z_values[i])
                * integral
            )

        return z_values, lensing_weight

    def get_lensing_weight(self, chis, chisource, **kwargs):
        """
        Dispatch method to compute lensing weights depending on source distribution.

        Parameters
        ----------
        chis : array_like
            Comoving radial distances where lensing weights are evaluated.
        chisource : float
            Source-plane comoving distance (used only if no n(z) is provided).
        **kwargs :
            Optional arguments:
            - nz_file : str, optional
                If provided, uses redshift distribution from file (not yet implemented).

        Returns
        -------
        z_values : ndarray
            Redshifts corresponding to chis.
        lensing_weight : ndarray
            Lensing weights W(chi) using either a single plane or n(z).
        """
        if kwargs.get("nz_file") is not None:
            return self.get_lensing_weight_array_nz(chis)
        else:
            return self.get_lensing_weight_array(chis, chisource)
