import numpy as np
from scipy.optimize import brentq, newton, bisect


def func_rachford_rice(x, z, K_values):
    '''
    x = n_moles_gas / n_moles_total
    '''
    c = 1.0 / (K_values - 1.0)
    return np.sum(z / (c + x))


def deriv_rachford_rice(x, z, K_values):
    c = 1.0 / (K_values - 1.0)
    return - np.sum(z / ((c + x) ** 2))


def calculate_rachford_rice(z, K_values):
    min_K = np.min(K_values)
    max_K = np.max(K_values)

    min_val = 0.999 / (1.0 - max_K)
    max_val = 0.999 / (1.0 - min_K)

    F_V = brentq(func_rachford_rice, min_val, max_val, args=(z, K_values))
    # F_V = newton(func=func_rachford_rice, x0=0.5, fprime=deriv_rachford_rice, args=(z, K_values))
    # F_V = bisect(func_rachford_rice, min_val, max_val, args=(z, K_values))

    return F_V