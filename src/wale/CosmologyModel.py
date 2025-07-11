import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import pyccl as ccl
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from pyccl.halos.pk_4pt import halomod_Tk3D_4h


class Cosmology_function:
    def __init__(self, h, Oc, Ob, w=-1.0, wa=0.0, **kwargs):
        """
        h  : dimensionless Hubble parameter (H0 = 100 h km/s/Mpc)
        Oc : cold dark matter density Ω_c
        Ob : baryon density Ω_b
        w0 : dark-energy equation of state today
        wa : dark-energy evolution parameter
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
        """Dimensionless expansion rate E(z)=H(z)/H0."""
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
        Hubble parameter H(z) in km/s/Mpc.
        """
        return self.H0 * self._E(z)

    def get_chi(self, z):
        """
        Comoving radial distance χ(z) in Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate χ.

        Returns
        -------
        chi : float or ndarray
            Comoving distance in Mpc.
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
        Compute lensing weight W(chi) using internal chi(z), z(chi), and a given n(z) file.
        No pyccl used.
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
        if kwargs.get("nz_file") is not None:
            return self.get_lensing_weight_array_nz(chis)
        else:
            return self.get_lensing_weight_array(chis, chisource)
