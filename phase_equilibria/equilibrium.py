import numpy as np
from scipy.optimize import fsolve

from flash import ss_flash, flash_residual_function
from stability import calculate_stability_test
from utils import calculate_K_values_wilson



def calculate_vapor_liquid_equilibrium(
        eos,
        P,
        T,
        z,
        K_values_estimates,
        print_statistics=False
):
    size = z.shape[0]

    is_stable, K_values_est = calculate_stability_test(
        eos,
        P,
        T,
        z,
        K_values_estimates
    )

    if not is_stable:
        K_values_from_ss_flash, F_V, f_L = ss_flash(eos, P, T,
                                                    z, K_values_est,
                                                    tolerance=1.0e-5,
                                                    print_statistics=print_statistics)

        x0 = np.append(K_values_from_ss_flash, F_V)
        result, infodict, ier, mesg = fsolve(
            func=flash_residual_function,
            x0=x0,
            args=(T, P, eos, z),
            full_output=True,
        )

        if print_statistics:
            print('Newton flash converged? %d, %s' % (ier, mesg))

        K_values = result[0:size]
        F_V = result[size]
    else:
        if P < 50.0e5:
            F_V = 1.0
        else:
            F_V = 0.0

        K_values = np.ones(size)

    return F_V, K_values


def calculate_molar_fraction_curve(
        eos, pressure_points, T, z,
        print_statistics=False
):
    size = z.shape[0]
    res = []
    K_values = np.ones(size)

    for iteration, P in enumerate(pressure_points):
        if np.linalg.norm(K_values - 1.0) < 1.0e-3:
            # Estimate initial K-values
            K_values_estimates = calculate_K_values_wilson(P, T, eos.Pc, eos.Tc, eos.ω)
        else:
            K_values_estimates = np.copy(K_values)

        if print_statistics:
            print('Pressure: %g bar' % (P / 1.0e5))
        F_V, K_values = calculate_vapor_liquid_equilibrium(
            eos, P, T, z,
            K_values_estimates, print_statistics)
        # print P/1.0e5, F_V

        res.append(F_V)

    return np.array(res)