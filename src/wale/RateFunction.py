import numpy as np
from scipy.optimize import root
from numpy import newaxis


def get_tau(rho):
    nu = 1.4
    return nu * (1.0 - rho ** (-1.0 / nu))


def get_psi_2cell(variance, chi, recal, z, delta1, delta2, theta1, theta2):
    delta1 = np.array([delta1])
    delta2 = np.array([delta2])
    tau1sq = get_tau(1.0 + delta1) * get_tau(1.0 + delta1)
    tau2sq = get_tau(1.0 + delta2) * get_tau(1.0 + delta2)
    tau12sq = get_tau(1.0 + delta1) * get_tau(1.0 + delta2)
    # sig2lR11 = 1.0
    sig2lr12 = variance.nonlinear_sigma2(
        redshift=z,
        R1=chi * ((1 + delta1[:, newaxis]) ** 0.5) * theta1,
        R2=chi * ((1 + delta2[:, newaxis]) ** 0.5) * theta2,
    )

    sig2lr11 = variance.nonlinear_sigma2(
        redshift=z,
        R1=chi * ((1 + delta1[:, newaxis]) ** 0.5) * theta1,
        R2=chi * ((1 + delta1[:, newaxis]) ** 0.5) * theta1,
    )

    sig2lr22 = variance.nonlinear_sigma2(
        redshift=z,
        R1=chi * ((1 + delta2[:, newaxis]) ** 0.5) * theta2,
        R2=chi * ((1 + delta2[:, newaxis]) ** 0.5) * theta2,
    )

    det = (sig2lr11 * sig2lr22) - (sig2lr12 * sig2lr12)
    psi = (
        (sig2lr11 * tau2sq - 2.0 * sig2lr12 * tau12sq + sig2lr22 * tau1sq)
        * recal
        / (det * 2.0)
    )
    return psi


def get_phi_projec_2cell(
    theta1, theta2, zarr, chis, dchis, w, y, recal, variance, **kwargs
):
    deld = kwargs.get("deld", 1e-8)
    nchi = len(chis)
    ny = len(y)

    def to_solve2(delta, A, recal, chi, z):
        delta_1, delta_2 = delta
        psi_at_delta = get_psi_2cell(
            variance,
            chi,
            recal=recal,
            z=z,
            delta1=delta_1,
            delta2=delta_2,
            theta1=theta1,
            theta2=theta2,
        )
        psi_at_delta1_plus_epsilon = get_psi_2cell(
            variance,
            chi=chi,
            recal=recal,
            z=z,
            delta1=delta_1 + deld,
            delta2=delta_2,
            theta1=theta1,
            theta2=theta2,
        )
        psi_at_delta2_plus_epsilon = get_psi_2cell(
            variance,
            chi=chi,
            recal=recal,
            z=z,
            delta1=delta_1,
            delta2=delta_2 + deld,
            theta1=theta1,
            theta2=theta2,
        )
        psi_derivative_delta1 = psi_at_delta1_plus_epsilon - psi_at_delta
        psi_derivative_delta2 = psi_at_delta2_plus_epsilon - psi_at_delta
        equation1 = A + psi_derivative_delta1
        equation2 = -A + psi_derivative_delta2
        return [equation1[0], equation2[0]]

    delta_zeros = np.zeros((nchi, ny, 2))

    for i in range(nchi):
        print(f"Iteration {i*100/nchi} %", end="\r")
        for j in range(ny):
            A = y[j] * w[i] * deld
            if j == 0:
                x0 = 0.0
                y0 = 0.0
            else:
                x0 = delta_zeros[i, j - 1][0]
                y0 = delta_zeros[i, j - 1][1]
            delta_zeros[i, j] = root(
                to_solve2,
                [x0, y0],
                args=(A, recal, np.array([chis[i]]), zarr[i]),
                method="hybr",
            ).x

    phi_proj = []
    for i in range(ny):
        phi_ = 0.0
        for j in range(nchi):
            phi_ += (
                y[i] * w[j] * (-delta_zeros[j, i, 0] + delta_zeros[j, i, 1])
                - get_psi_2cell(
                    variance,
                    chi=np.array([chis[j]]),
                    recal=recal,
                    z=zarr[j],
                    delta1=delta_zeros[j, i, 0],
                    delta2=delta_zeros[j, i, 1],
                    theta1=theta1,
                    theta2=theta2,
                )
            ) * dchis#[j]
        phi_proj.append(phi_)
    phi_proj = np.array(phi_proj)

    return phi_proj


def get_scaled_cgf(
    theta1, theta2, zarr, chis, dchis, lensing_weight, y, recal, variance
):
    scgf = get_phi_projec_2cell(
        theta1, theta2, zarr, chis, dchis, lensing_weight, y, recal, variance
    )
    return scgf


def get_psi_derivative_delta1(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    delh = deld  # Small step size for numerical differentiation
    delta1_plus_h = delta1 + delh
    delta1_minus_h = delta1 - delh

    psi_plus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_plus_h, delta2, theta1, theta2
    )
    psi_minus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_minus_h, delta2, theta1, theta2
    )

    derivative = (psi_plus_h - psi_minus_h) / (2 * delh)
    return derivative


def get_psi_derivative_delta2(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    delh = deld  # Small step size for numerical differentiation
    delta2_plus_h = delta2 + delh
    delta2_minus_h = delta2 - delh

    psi_plus_h = get_psi_2cell(
        variance, chi, recal, z, delta1, delta2_plus_h, theta1, theta2
    )
    psi_minus_h = get_psi_2cell(
        variance, chi, recal, z, delta1, delta2_minus_h, theta1, theta2
    )

    derivative = (psi_plus_h - psi_minus_h) / (2 * delh)
    return derivative


def get_psi_2nd_derivative_delta1(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    delh = deld
    delta1_plus_h = delta1 + delh
    delta1_minus_h = delta1 - delh

    psi_plus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_plus_h, delta2, theta1, theta2
    )
    psi_minus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_minus_h, delta2, theta1, theta2
    )
    psi_at_delta1 = get_psi_2cell(
        variance, chi, recal, z, delta1, delta2, theta1, theta2
    )

    second_derivative = (psi_plus_h - (2.0 * psi_at_delta1) + psi_minus_h) / (delh**2.0)
    return second_derivative


def get_psi_2nd_derivative_delta2(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    delh = deld
    delta2_plus_h = delta2 + delh
    delta2_minus_h = delta2 - delh

    psi_plus_h = get_psi_2cell(
        variance, chi, recal, z, delta1, delta2_plus_h, theta1, theta2
    )
    psi_minus_h = get_psi_2cell(
        variance, chi, recal, z, delta1, delta2_minus_h, theta1, theta2
    )
    psi_at_delta2 = get_psi_2cell(
        variance, chi, recal, z, delta1, delta2, theta1, theta2
    )

    second_derivative = (psi_plus_h - (2.0 * psi_at_delta2) + psi_minus_h) / (delh**2.0)
    return second_derivative


def get_psi_mixed_derivative_delta1_delta2(
    deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
):
    delh = deld
    delta1_plus_h = delta1 + delh
    delta1_minus_h = delta1 - delh
    delta2_plus_h = delta2 + delh
    delta2_minus_h = delta2 - delh

    psi_delta1_plus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_plus_h, delta2_plus_h, theta1, theta2
    )
    psi_delta1_minus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_minus_h, delta2_plus_h, theta1, theta2
    )
    psi_delta2_plus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_plus_h, delta2_minus_h, theta1, theta2
    )
    psi_delta2_minus_h = get_psi_2cell(
        variance, chi, recal, z, delta1_minus_h, delta2_minus_h, theta1, theta2
    )

    mixed_derivative = (
        psi_delta1_plus_h - psi_delta1_minus_h - psi_delta2_plus_h + psi_delta2_minus_h
    ) / (4.0 * delh * delh)
    return mixed_derivative


def psi_derivative_determinant(
    deld, delta1, delta2, z, variance, chi, recal, theta1, theta2
):
    psi_11 = get_psi_2nd_derivative_delta1(
        deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
    )
    psi_22 = get_psi_2nd_derivative_delta2(
        deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
    )
    psi_12 = get_psi_mixed_derivative_delta1_delta2(
        deld, variance, chi, recal, z, delta1, delta2, theta1, theta2
    )

    result = (psi_11 * psi_22) - (psi_12**2.0)
    return result
