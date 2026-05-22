import numpy as np
from astropy import units as u
from .CosmologyModel import Cosmology_function
from .CovarianceMatrix import *

import matplotlib.pyplot as plt

class InitialiseVariables:
    def __init__(
        self, h, Oc, Ob, w, wa, sigma8, dk, kmin, kmax, nplanes, theta1, **kwargs
    ):
        """
        Initialise variables for WALE.
        Parameters:
        - h: Hubble constant (dimensionless)
        - Oc: Omega matter (dimensionless)
        - Ob: Omega baryon (dimensionless)
        - w: Dark energy equation of state parameter (dimensionless)
        - wa: Dark energy equation of state parameter (dimensionless)
        - sigma8: Amplitude of matter fluctuations (dimensionless)
        - dk: Step size in k (1/Mpc)
        - kmin: Minimum k value (1/Mpc)
        - kmax: Maximum k value (1/Mpc)
        - nplanes: Number of lens planes (integer)
        - theta1: Angular scale in arcminutes (float)
        - **kwargs: Additional keyword arguments (optional)

        Calculates:
        - Angular scales in radians
        - Comoving distances for lens planes
        - Lensing weights based on source redshift or redshift distribution file
        - Covariance matrix or single P_nl based on variability flag
        - Initializes Cosmology_function with provided parameters
        """
        # Initialise cosmology
        self.cosmo = Cosmology_function(
            h=h,
            Oc=Oc,
            Ob=Ob,
            w=w,
            wa=wa,
            sigma8=sigma8,
            dk=dk,
            kmin=kmin,
            kmax=kmax,
            **kwargs,
        )
        print("\nInitialised Cosmology:")
        print(f"  h = {h}, Oc = {Oc}, Ob = {Ob}, w = {w}, wa = {wa}, sigma8 = {sigma8}")
        print(f"  k-range: [{kmin}, {kmax}] with step dk = {dk}")
        print()

        # Angular scales in radians
        self.theta1_radian = theta1 * u.arcmin.to(u.radian)
        self.theta2_radian = 2. * self.theta1_radian
        print("the angles are: ", self.theta1_radian, " and ", self.theta2_radian)
        # Number of lens planes
        self.nplanes = nplanes

        # Source redshift info
        self.zsource = kwargs.get("zs", None)
        self.nz_file = kwargs.get("nz_file", None)
        # BNT mode: list of 4 n(z) files + 4 coefficients. When both are supplied,
        # the lensing kernel is built as Σ_i bnt_coeffs[i] * q_i(χ), where each q_i
        # comes from get_lensing_weight_array_nz on the i-th normalised n(z).
        self.bnt_nz_files = kwargs.get("bnt_nz_files", None)
        self.bnt_coeffs = kwargs.get("bnt_coeffs", None)
        self.bnt_mode = (
            self.bnt_nz_files is not None and self.bnt_coeffs is not None
        )
        # print("Initialising WALE with the following parameters:")
        # print("   Initialised Variables:")
        # print(
        #     f"      Source redshift: {self.zsource if self.zsource is not None else 'from nz file'}"
        # )
        # print(f"      Number of planes: {self.nplanes}")
        # print(f"      Angular scale theta1 (radians): {self.theta1_radian}, arcmin: {theta1}")


        if self.zsource is None and self.nz_file is None and not self.bnt_mode:
            raise ValueError("Please specify either 'zs', 'nz_file', or BNT files+coeffs.")

        if self.bnt_mode:
            if len(self.bnt_nz_files) != len(self.bnt_coeffs):
                raise ValueError(
                    f"bnt_nz_files (len {len(self.bnt_nz_files)}) and bnt_coeffs "
                    f"(len {len(self.bnt_coeffs)}) must have the same length."
                )
            bnt_nzs = [np.loadtxt(f) for f in self.bnt_nz_files]
            # Sanity: every n(z) file must share the same redshift grid (cosmoGRID does).
            z_ref = bnt_nzs[0][:, 0]
            for k, nz in enumerate(bnt_nzs[1:], start=1):
                if not np.array_equal(nz[:, 0], z_ref):
                    raise ValueError(
                        f"BNT n(z) file {self.bnt_nz_files[k]} has a different z-grid "
                        f"than {self.bnt_nz_files[0]}."
                    )
            # Use the largest z as the source for χ_source — all 4 bins have the same grid.
            self.z_nz = z_ref
            self.n_z = bnt_nzs[-1][:, 1]  # representative; not used downstream
            self.chisource = self.cosmo.get_chi(self.z_nz[-1])
        elif self.zsource is not None:
            self.chisource = self.cosmo.get_chi(self.zsource)
        else:
            nz = np.loadtxt(self.nz_file)
            self.z_nz = nz[:, 0]
            self.n_z = nz[:, 1]
            self.chisource = self.cosmo.get_chi(self.z_nz[-1])

        # Comoving distances for planes
        self.dchi = (self.chisource - 100) / self.nplanes
        self.chis = np.arange(100, self.chisource, self.dchi)

        # Lensing weights
        if self.bnt_mode:
            weights_per_bin = []
            redshifts_ref = None
            for nz_arr in bnt_nzs:
                z_i, n_i = nz_arr[:, 0], nz_arr[:, 1]
                z_vals, w_i = self.cosmo.get_lensing_weight_array_nz(self.chis, z_i, n_i)
                if redshifts_ref is None:
                    redshifts_ref = z_vals
                weights_per_bin.append(w_i)
            coeffs = np.asarray(self.bnt_coeffs, dtype=float)
            combined = np.zeros_like(weights_per_bin[0])
            for c, w in zip(coeffs, weights_per_bin):
                combined = combined + c * w
            self.redshifts = redshifts_ref
            self.lensingweights = combined
        elif self.nz_file is not None:
            self.redshifts, self.lensingweights = (
                self.cosmo.get_lensing_weight_array_nz(self.chis, self.z_nz, self.n_z)
            )
        else:
            # print("      Lensing weights calculated from source redshift")

            self.redshifts, self.lensingweights = self.cosmo.get_lensing_weight_array(
                self.chis, self.chisource
            )

        # Covariance or single P_nl
        self.variability = kwargs.get("variability", None)
        print(f"      Variability: {'enabled' if self.variability else 'disabled'}")
        if self.variability:
            self.numberofrealisations = kwargs.get("numberofrealisations", 5)
            print("          Using halo model for covariance and realizations")
            print(f"          Number of realizations: {self.numberofrealisations}")

            self.cosmo.cov, self.cosmo.pnlsamples, self.cosmo.pnl = get_covariance(
                self.cosmo,
                z=self.redshifts,
                variability=True,
                numberofrealisations=self.numberofrealisations,
            )
        else:
            self.cosmo.pnl = get_covariance(
                self.cosmo, z=self.redshifts, variability=False, numberofrealisations=1
            )

        # Linear P(k) — required for the 1-cell rate function ψ_l(δ) = τ²/(2σ²_l)
        # and for the per-slice rescaling ratio r = σ²_nl(R₀)/σ²_l(R₀).
        self.cosmo.plin = {
            float(z): self.cosmo.get_linear_pk(float(z))
            for z in self.redshifts
        }

        # Default recalibration factor (2-cell only).  Override per-filter with
        # variables.recal_value = σ²_LDT / σ²_sim before calling computePDF.
        self.recal_value = 1.0
