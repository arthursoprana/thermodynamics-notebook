import numpy as np
from math import sqrt, acos, cos


def calculate_K_values_wilson(P, T, Pc, Tc, ω):
    return (Pc / P) * np.exp(5.37 * (1 + ω) * (1 - (Tc / T)))


def calculate_roots_of_cubic_equation(p0, p1, p2, p3):
    coef_a = (3.0 * p2 - (p1 ** 2)) / 3.0
    coef_b = (2.0 * (p1 ** 3) - 9.0 * p1 * p2 + 27.0 * p3) / 27.0
    delta = 0.25 * (coef_b ** 2) + (coef_a ** 3) / 27.0

    roots = []
    if delta > 0.0:
        # 1 real root, 2 imaginary
        const_A = np.cbrt(-0.5 * coef_b + sqrt(delta))
        const_B = np.cbrt(-0.5 * coef_b - sqrt(delta))

        single_root = const_A + const_B - p1 / 3.0

        roots.append(single_root)
    else:
        # 3 real roots
        phi = acos(-0.5 * coef_b / sqrt(-(coef_a ** 3) / 27.0))
        root_1 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0) - p1 / 3.0
        root_2 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 2.0 * np.pi / 3.0) - p1 / 3.0
        root_3 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 4.0 * np.pi / 3.0) - p1 / 3.0

        roots.append(root_1)
        roots.append(root_2)
        roots.append(root_3)

    return roots