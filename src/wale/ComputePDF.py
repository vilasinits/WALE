import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
import matplotlib.pyplot as plt

from wale.RateFunction import get_scaled_cgf


class computePDF:
    """
    Compute the Probability Distribution Function (PDF) of the convergence field (kappa)
    using Large Deviation Theory (LDT) and the scaled cumulant generating function (SCGF).

    This class takes a `VariablesGenerator` instance that encapsulates cosmological inputs,
    angular scales, redshift distributions, and lensing weights, along with a smoothed variance value.

    The PDF is computed using the Legendre transform of the SCGF and the Bromwich integral.

    Attributes
    ----------
    variables : VariablesGenerator
        Contains all required parameters like angles, redshifts, chi, weights, and lambda grid.
    variance : float
        Smoothed variance σ²_κ(θ) for the angular scale and filter used.
    plot_scgf : bool
        Whether to plot the SCGF during PDF computation.
    pdf_values : list of float
        Computed PDF values over the kappa grid.
    kappa_values : ndarray
        Grid of kappa values over which the PDF is evaluated.
    """

    def __init__(self, variables, variance, plot_scgf=False):
        """
        Initialize the PDF computation class.

        Parameters
        ----------
        variables : VariablesGenerator
            Instance containing all necessary cosmological parameters and variable arrays.
        variance : float
            Smoothed convergence variance σ²_κ.
        plot_scgf : bool, optional
            Whether to display a plot of the scaled cumulant generating function (default is False).
        """
        self.variables = variables
        self.plot_scgf = plot_scgf
        self.variance = variance
        self.pdf_values, self.kappa_values = self.compute_pdf_values()

    def get_scgf(self):
        """
        Compute the scaled cumulant generating function (SCGF) for the given inputs.

        Returns
        -------
        scgf : ndarray
            Array of SCGF values evaluated on the lambda grid.
        """
        # Utilizing variables from the VariablesGenerator instance
        scgf = get_scaled_cgf(
            self.variables.theta1_radian,
            self.variables.theta2_radian,
            self.variables.redshifts,
            self.variables.chis,
            self.variables.dchi,
            self.variables.lensingweights,
            self.variables.lambdas,
            self.variables.recal_value,
            self.variance,
        )
        return scgf

    def compute_phi_values(self):
        """
        Compute the Legendre-transformed function φ(λ) from the SCGF.

        Returns
        -------
        lambda_new : ndarray
            Imaginary grid of lambda values used for the inverse Laplace transform (Bromwich integral).
        phi_values : ndarray
            Corresponding φ(λ) values used to compute the PDF.
        """
        scgf = self.get_scgf()
        scgf_spline = CubicSpline(self.variables.lambdas, scgf[:, 0], axis=0)
        dscgf = scgf_spline(self.variables.lambdas, 1)
        if self.plot_scgf:
            plt.figure(figsize=(4, 4))
            plt.plot(self.variables.lambdas, scgf)
            plt.show()

        tau_effective = np.sqrt(2.0 * (self.variables.lambdas * dscgf - scgf[:, 0]))
        x_data = np.sign(self.variables.lambdas) * tau_effective
        y_data = dscgf

        coeffs = np.polyfit(x_data, y_data, 7)
        p = np.poly1d(coeffs)
        dp = p.deriv()
        # print("the coeffs are", p.coeffs)
        print("The variance from PDF is: ", p.coeffs[-2] ** 2)
        lambda_new = 1j * np.arange(0, 100000)

        taus = np.zeros_like(lambda_new, dtype=np.complex128)

        def vectorized_equation(tau, lambda_):
            return tau - dp(tau) * lambda_

        for n, lambda_ in enumerate(lambda_new):
            initial_guess = np.sqrt(1j * (10 ** (-16))) if n == 0 else taus[n - 1]
            taus[n] = newton(vectorized_equation, x0=initial_guess, args=(lambda_,))

        phi_values = lambda_new * p(taus) - ((taus**2) / 2.0)
        return lambda_new, phi_values

    def compute_pdf_for_kappa(self, kappa, lambda_new, phi_values):
        """
        Evaluate the PDF at a given kappa using the inverse Laplace (Bromwich) integral.

        Parameters
        ----------
        kappa : float
            Value of the convergence (κ) for which the PDF is evaluated.
        lambda_new : ndarray
            Array of complex λ values.
        phi_values : ndarray
            Corresponding φ(λ) values.

        Returns
        -------
        pdf_kappa : float
            Estimated value of the PDF at the given kappa.
        """
        delta_lambda = np.abs(lambda_new[1] - lambda_new[0]) * 1j
        lambda_weight = np.full(len(lambda_new), delta_lambda)
        lambda_weight[0] = lambda_weight[-1] = delta_lambda / 2.0

        integral_sum = np.sum(np.exp(-lambda_new * kappa + phi_values) * lambda_weight)
        pdf_kappa = np.imag(integral_sum / (1.0 * np.pi))

        return pdf_kappa.real

    def compute_pdf_values(self):
        """
        Compute the PDF across a grid of kappa values using the Bromwich integral.

        Returns
        -------
        pdf_values : list of float
            Evaluated PDF values.
        kappa_values : ndarray
            Grid of kappa values over which the PDF is computed.
        """
        kappa_values = np.linspace(-0.06, 0.06, 501)
        lambda_new, phi_values = self.compute_phi_values()
        pdf_values = [
            self.compute_pdf_for_kappa(kappa, lambda_new, phi_values)
            for kappa in kappa_values
        ]
        return pdf_values, kappa_values
