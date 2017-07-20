import numpy as np
from scipy.optimize import fsolve

from eos import VanDerWaalsEos, PengRobinsonEos, SoaveRedlichKwongEos, R
from flash import ss_flash, flash_residual_function,\
    multiphase_flash_residual_function
from stability import calculate_stability_test
from utils import calculate_K_values_wilson
from cases import input_properties_case_Ghafri



def calculate_density(x, P, T, Mi, eos):
    M = (Mi * x).sum()
    eos.update_eos_coefficients(P, T, x)
    Z = eos.calculate_eos_roots()
    
    f0 = eos.calculate_fugacities(P, T, Z[0], x)

    if len(Z) == 1:
        return P * M / (Z[0] * R * T)

    if Z[1] < 0.0:
        return P * M / (Z[0] * R * T)

    f1 = eos.calculate_fugacities(P, T, Z[1], x)

    g0 = eos.calculate_normalized_gibbs_energy(f0, x)
    g1 = eos.calculate_normalized_gibbs_energy(f1, x)

    if g0 < g1:
        return P * M / (Z[0] * R * T)
    else:
        return P * M / (Z[1] * R * T)
    
    return P * M / (Z * R * T)
    
    
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
    

def test_phase_equilibria_CO2():
    # Get input properties
    temperature = 290.0 # [K]   
    pressure = 1.0 * 1e5 # [Pa]
    
    props = input_properties_case_Ghafri(temperature)

    (_, _, global_molar_fractions,
    critical_pressure, critical_temperature, acentric_factor,
    molar_mass, omega_a, omega_b, binary_interaction) = props


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
    #print ('K_values estimates:', K_values_est)

    K_values_from_ss_flash, F_V, f_L = ss_flash(
        eos, 
        pressure, 
        temperature, 
        global_molar_fractions, 
        K_values_est, 
        tolerance=1.0e-1,
        print_statistics=True
    )

    
    print ('K_values Successive Subst:', K_values_from_ss_flash)
    print ('Vapor molar fraction:', F_V)
    print ('\n-----\nFugacities obtained:', f_L)

    # Use estimates from Wilson's Equation!!!
    #x0 = np.append(initial_K_values, F_V) # It does not work!

    K_values_est = np.array([2.71350413e+01,   9.23075054e+01,   1.17478417e+01,   2.84833217e+00,
                             8.98083183e-02,   1.48315222e-02,   2.65989146e-03,   7.30566453e-04,
                             7.76675634e-04,   1.00536144e-04,   4.54130173e-05,   3.11385702e-05,
                             1.71481448e-06,   2.66939726e-06,   1.64767536e-07,   2.36654995e-08,
                             1.10913280e-08,   1.60469889e-09,   8.82593963e-11,   1.10828717e-13,
                             5.96510411e-03])
    
    # Use estimates from stability test!!!
    x0 = np.append(K_values_est, 0.749313901696) # It does not work!

    # Use estimates from successive substitutions!!!
    #x0 = np.append(K_values_from_ss_flash, F_V) # Good estimate!

    result = fsolve(
        func=flash_residual_function,
        x0=x0,
        args=(temperature, pressure, eos, global_molar_fractions),
        full_output=1
    )
    
    converged = result[2]
    msg = result[3]
    print (msg)
    
    X = result[0]
    size = X.shape[0]
    K_values_newton = X[0:size-1]
    F_V = X[size-1]
    
    
    print ('K_values newton:', K_values_newton)
    print ('Vapor molar fraction:', F_V)
    assert converged == 1
    
    Mi = molar_mass
    P, T = pressure, temperature
    z = global_molar_fractions
    K = K_values_newton
    x_L = z / (F_V * (K - 1) + 1)
    x_V = K * x_L

    ρ_V = calculate_density(x_V, P, T, Mi, eos)
    ρ_L = calculate_density(x_L, P, T, Mi, eos)
    
    
    print ('Vapor density:', ρ_V)
    print ('Liquid density:', ρ_L)



def test_multiphase_flash_residual_function():
    # Get input properties
    temperature = 278.0 # [K]   
    pressure = 0.5 * 1e5 # [Pa]
    
    props = input_properties_case_Ghafri(temperature)

    (_, _, global_molar_fractions,
    critical_pressure, critical_temperature, acentric_factor,
    molar_mass, omega_a, omega_b, binary_interaction) = props


    # Estimate initial K-values
    initial_K_values = calculate_K_values_wilson(
        pressure,
        temperature,
        critical_pressure,
        critical_temperature,
        acentric_factor
    )

    # Create EoS object and set properties
    eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                          omega_a, omega_b, binary_interaction)
    
    
    P, T = pressure, temperature
    z = global_molar_fractions
    
    K_values_est = initial_K_values
    x0 = np.r_[K_values_est, K_values_est, 0.1, 0.2]
    n_extra_phases = 2

    result, infodict, ier, mesg = fsolve(
        func=multiphase_flash_residual_function,
        x0=x0,
        args=(T, P, eos, z, n_extra_phases),
        full_output=1
    )
    
    print (mesg)
    
    shape = (n_extra_phases, z.size)

    # Get values from unknown vector
    K_values = result[               :-n_extra_phases].reshape(shape)
    β        = result[-n_extra_phases:               ]
   
    
    print ('K_values:\n', K_values)
    print ('Molar phase fractions:', β)
    assert ier == 1
    
    Mi = molar_mass
    K = K_values
    
    denominator = 1 + (β[:, np.newaxis] * (K - 1)).sum(axis=0)
    y_iF = z / denominator
    y_i  = K * y_iF 
    
    print ('Molar component phase fractions:\n', y_i)
    
    # Reference phase density
    ρ_F = calculate_density(y_iF, P, T, Mi, eos)

    ρ = []
    for j in range(n_extra_phases):
        ρ_j = calculate_density(y_i[j], P, T, Mi, eos)
        ρ = np.r_[ρ, ρ_j]        
    ρ = np.r_[ρ, ρ_F]    
    
    print ('Obtained Densities:', ρ)

    
    