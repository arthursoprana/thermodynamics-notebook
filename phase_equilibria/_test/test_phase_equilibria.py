import numpy as np
from scipy.optimize import fsolve

from eos import VanDerWaalsEos, PengRobinsonEos, SoaveRedlichKwongEos
from flash import ss_flash, flash_residual_function
from stability import calculate_stability_test
from utils import calculate_K_values_wilson


def input_properties_case_whitson_problem_18_PR():
    '''
    TEST PROBLEM PHASE BEHAVIOUR WHITSON PROBLEM 18 APPENDIX

    Methane, Butane and Decane (C1, C4 and C10).

    Properties for the Van der Waals Equation of State.

    '''
    temperature = (280.0 + 459.67) * 5.0 / 9.0
    pressure = 500.0 * 6894.75729

    critical_pressure = 6894.75729 * np.array([667.8, 550.7, 304.0])  # [atm]
    critical_temperature = (5.0 / 9.0) * np.array([343.0, 765.3, 1111.8])  # [K]
    acentric_factor = np.array([0.011500, 0.192800, 0.490200])  # [-]
    molar_mass = 0.001 * np.array([16.04, 58.12, 142.29])  # [g/mol]
    omega_a = 0.45724 * np.array([1.0, 1.0, 1.0])  # [-]
    omega_b = 0.07780 * np.array([1.0, 1.0, 1.0])  # [-]

    binary_interaction = np.array(
        [[0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000]]
    )

    global_molar_fractions = np.array([0.5, 0.42, 0.08])

    return (pressure, temperature, global_molar_fractions,
            critical_pressure, critical_temperature, acentric_factor,
            molar_mass, omega_a, omega_b, binary_interaction)




def test_phase_equilibria():
    # Get input properties
    #props = input_properties_case_7_psudocomponents()
    props = input_properties_case_whitson_problem_18_PR()
    #props = input_properties_case_whitson_problem_18_SRK()
    #props = input_properties_case_whitson_problem_18_VDW()

    (pressure, temperature, global_molar_fractions,
    critical_pressure, critical_temperature, acentric_factor,
    molar_mass, omega_a, omega_b, binary_interaction) = props

    #temperature = 350.0 # [K]
    #pressure = 50.0 * 1e5 # [Pa]

    # Estimate initial K-values
    initial_K_values = calculate_K_values_wilson(
        pressure,
        temperature,
        critical_pressure,
        critical_temperature,
        acentric_factor
    )

    # Create EoS object and set properties
    #eos = VanDerWaalsEos, PengRobinsonEos, SoaveRedlichKwongEos
    eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                          omega_a, omega_b, binary_interaction)

    is_stable, K_values_est = calculate_stability_test(
        eos,
        pressure,
        temperature,
        global_molar_fractions,
        initial_K_values
    )

    print ('System is stable?', is_stable)
    print ('K_values estimates:', K_values_est)

    K_values_from_ss_flash, F_V, f_L = ss_flash(eos, pressure, temperature, global_molar_fractions, K_values_est, tolerance = 1.0e-1)

    fugacity_expected = np.array([294.397, 148.342, 3.02385]) * 6894.75729
    K_values_expected = np.array([6.65071, 0.890061, 0.03624])
    x_expected = np.array([0.08588, 0.46349, 0.45064])
    y_expected = np.array([0.57114, 0.41253, 0.01633])
    
    print ('K_values Successive Subst:', K_values_from_ss_flash)
    print ('Vapor molar fraction:', F_V)
    print ('\n-----\nFugacities obtained:', f_L)
    print ('Fugacities expected:', fugacity_expected)

    # Use estimates from Wilson's Equation!!!
    #x0 = np.append(initial_K_values, F_V) # It does not work!

    # Use estimates from stability test!!!
    #x0 = np.append(K_values_est, F_V) # It does not work!

    # Use estimates from successive substitutions!!!
    x0 = np.append(K_values_from_ss_flash, F_V) # Good estimate!

    result = fsolve(
        func=flash_residual_function,
        x0=x0,
        args=(temperature, pressure, eos, global_molar_fractions),
    )

    size = result.shape[0]
    K_values_newton = result[0:size-1]
    F_V = result[size-1]
    print ('K_values newton:', K_values_newton)
    print ('K_values expected:', K_values_expected)
    print ('Norm difference:', np.linalg.norm(K_values_expected - K_values_newton))
    print ('Vapor molar fraction:', F_V)

    assert np.allclose(K_values_newton, K_values_expected, rtol=0.01)
    assert np.allclose(fugacity_expected, f_L, rtol=0.1)

