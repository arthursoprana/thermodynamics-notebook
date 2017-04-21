import numpy as np


def stability_test_residual_function(x, T, P, eos, z, test_type):
    # Getting unknowns
    u = x

    # ref
    f_ref = eos.calculate_fugacities_with_minimum_gibbs_energy(P, T, z)

    if test_type is 'vapor':
        other_type = 'liquid'
        K_values = u / z
        x_u = z * K_values

    else:
        assert test_type is 'liquid', 'Non existing test_type! ' + test_type
        other_type = 'vapor'
        K_values = z / u
        x_u = z / K_values

    x_u_normalized = x_u / np.sum(x_u)

    # Liquid
    f_u = eos.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_u_normalized)

    residual = f_ref - f_u * np.sum(x_u)

    return residual

def ss_stability_test(
        eos,
        P,
        T,
        z,
        test_type,
        K_values_initial,
        max_iter=100,
        tolerance=1.0e-5
):
    K = np.copy(K_values_initial)

    error = 100.0

    f_ref = eos.calculate_fugacities_with_minimum_gibbs_energy(P, T, z)

    counter = 0
    while error > tolerance and counter < max_iter:
        if test_type is 'vapor':
            other_type = 'liquid'
            x_u = z * K
        else:
            assert test_type is 'liquid', 'Non existing test_type! ' + test_type
            other_type = 'vapor'
            x_u = z / K

        sum_x_u = np.sum(x_u)
        x_u_normalized = x_u / sum_x_u

        f_u = eos.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_u_normalized)

        if test_type is 'vapor':
            correction = f_ref / (f_u * sum_x_u)
        else:
            assert test_type is 'liquid', 'Non existing test_type! ' + test_type
            correction = (f_u * sum_x_u) / f_ref

        K *= correction
        error = np.linalg.norm(correction - 1.0)
        counter += 1

    return sum_x_u, K

def calculate_stability_test(
        eos,
        P,
        T,
        z,
        K_values_initial
):
    sum_vapor, K_values_vapor = ss_stability_test(
        eos,
        P,
        T,
        z,
        'vapor',
        K_values_initial
    )

    sum_liquid, K_values_liquid = ss_stability_test(
        eos,
        P,
        T,
        z,
        'liquid',
        K_values_initial
    )

    sum_ln_K_vapor = np.linalg.norm(np.log(K_values_vapor)) ** 2
    sum_ln_K_liquid = np.linalg.norm(np.log(K_values_liquid)) ** 2
    sum_tol = 1.0e-8

    # Table 4.6 from Phase Equilibria
    if sum_ln_K_vapor < 1.0e-4 and sum_ln_K_liquid < 1.0e-4:
        is_stable = True
    elif (sum_vapor - 1.0) <= sum_tol and sum_ln_K_liquid < 1.0e-4:
        is_stable = True
    elif (sum_liquid - 1.0) <= sum_tol and sum_ln_K_vapor < 1.0e-4:
        is_stable = True
    elif (sum_vapor - 1.0) <= sum_tol and (sum_liquid - 1.0) <= sum_tol:
        is_stable = True
    elif (sum_vapor - 1.0) > sum_tol and sum_ln_K_liquid < 1.0e-4:
        is_stable = False
    elif (sum_liquid - 1.0) > sum_tol and sum_ln_K_vapor < 1.0e-4:
        is_stable = False
    elif (sum_vapor - 1.0) > sum_tol and (sum_liquid - 1.0) > sum_tol:
        is_stable = False
    elif (sum_vapor - 1.0) > sum_tol and (sum_liquid - 1.0) <= sum_tol:
        is_stable = False
    elif (sum_vapor - 1.0) <= sum_tol and (sum_liquid - 1.0) > sum_tol:
        is_stable = False
    else:
        assert False, 'ERROR: No stability condition found...'

    if not is_stable:
        K_values_estimates = K_values_vapor * K_values_liquid
    else:
        K_values_estimates = np.copy(K_values_initial)

    return is_stable, K_values_estimates