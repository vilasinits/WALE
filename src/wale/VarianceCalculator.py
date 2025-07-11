import numpy as np
from scipy.integrate import simps
import pyccl as ccl
from wale.FilterFunctions import top_hat_filter, starlet_filter



class Variance:
    """
    A class to compute linear and nonlinear variance using power spectrum interpolators and a specific cosmological model.

    Attributes:
        cosmo (Cosmology): An instance of a cosmology class providing necessary cosmological functions and parameters.
        PK_interpolator_linear (Interpolator): An interpolator instance for linear power spectrum calculations.
        PK_interpolator_nonlinear (Interpolator): An interpolator instance for nonlinear power spectrum calculations.
        model (str): The name of the cosmological model to be used for variance calculations.

    """

    def __init__(
        self,
        cosmo,
        filter_type,
        pk
    ):
        """
        Initializes the Variance class with cosmology and parameters for P(k) calculation.
        Calculates the non-linear power spectrum, including cosmic variance noise if volume is specified.

        Parameters:
            cosmo (Cosmology_function): An instance of the cosmology class.
            z_values (array-like): Redshifts for lensing planes/primary calculations.
            volume (float, optional): Volume for cosmic variance calculation in (Mpc/h)^3. Defaults to None (no CV noise).
            delta_A0 (float, optional): Parameter for additional non-Gaussian noise. Defaults to 1.9.
        """
        self.cosmo = cosmo
        self.filter_type = filter_type
        print("Variance module initialized...")
        self.pk = pk

    def linear_sigma2(self, redshift, R1, R2=None, **kwargs):
        """
        Calculates the linear variance σ² for given scales and redshift, considering the specified model adjustments.

        Parameters:
            redshift (float): The redshift at which to evaluate the variance.
            R1 (float): The first scale radius.
            R2 (float, optional): The second scale radius. Defaults to R1 if not specified.

        Returns:
            float: The linear variance σ² at the given scales and redshift.
        """
        if R2 is None:
            R2 = R1
        else:
            R2 = R2

        # pk = self.PK_interpolator_linear.P(redshift, self.cosmo.k_values)
        pk = (
            ccl.linear_matter_power(
                self.cosmo.cosmoccl, self.cosmo.k, 1 / (1 + redshift)
            )
            * self.cosmo.h**3
        )
        if self.filter_type == "tophat":
            w1_2D = top_hat_filter(self.cosmo.k , R1)
            w2_2D = top_hat_filter(self.cosmo.k , R2)
            w2 = w1_2D * w2_2D
        elif self.filter_type == "starlet":
            w1_2D = starlet_filter(self.cosmo.k, R1)
            w2_2D = starlet_filter(self.cosmo.k, R2)
            w2 = w1_2D * w2_2D
        constant = 1.0 / 2.0 / np.pi
        integrand = self.cosmo.k_values * pk * w2 * constant
        return simps(integrand, x=self.cosmo.k_values)

    def nonlinear_sigma2(self, redshift, R1, R2=None, **kwargs):
        """
        Calculates the nonlinear variance σ² for given scales and redshift, considering the specified model adjustments.

        Parameters:
            redshift (float): The redshift at which to evaluate the variance.
            R1 (float): The first scale radius.
            R2 (float, optional): The second scale radius. Defaults to R1 if not specified.

        Returns:
            float: The nonlinear variance σ² at the given scales and redshift.
        """
        if R2 is None:
            R2 = R1
        else:
            R2 = R2
        # pk = kwargs.get("pk", self.pk[redshift])
        pk = kwargs["pk"] if "pk" in kwargs else self.pk[redshift]

        # pk = self.pk[redshift]
        k = self.cosmo.k * self.cosmo.h
        if self.filter_type == "tophat":
            w1_2D = top_hat_filter(self.cosmo.k , R1)
            w2_2D = top_hat_filter(self.cosmo.k , R2)

            w2 = w1_2D * w2_2D
        elif self.filter_type == "starlet":
            w1_2D = starlet_filter(self.cosmo.k, R1)
            w2_2D = starlet_filter(self.cosmo.k, R2)

            w2 = w1_2D * w2_2D
        constant = 1.0 / 2.0 / np.pi
        integrand = k * pk * w2 * constant
        return simps(integrand, x=k)/self.cosmo.h

    def get_sig_slice(self, z, R1, R2):
        """
        Calculates the slice variance σ² for the given scales and redshift in the nonlinear regime.

        Parameters:
            z (float): The redshift at which to evaluate the slice variance.
            R1 (float): The first scale radius.
            R2 (float): The second scale radius.

        Returns:
            float: The slice variance σ² at the given scales and redshift.
        """
        if self.filter_type == "tophat":
            sigslice = (
                self.nonlinear_sigma2(z, R1)
                + self.nonlinear_sigma2(z, R2)
                - 2.0 * self.nonlinear_sigma2(z, R1, R2)
            )
            return sigslice
        elif self.filter_type == "starlet":
            sigslice = (
                self.nonlinear_sigma2(z, R1)
                + self.nonlinear_sigma2(z, R2)
                - 2.0 * self.nonlinear_sigma2(z, R1, R2)
            )
            return sigslice
