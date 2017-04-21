import numpy as np
import matplotlib.pyplot as plt

from equilibrium import calculate_molar_fraction_curve
from cases import (input_properties_case_whitson_problem_18_PR, input_properties_case_whitson_problem_18_SRK,
                   input_properties_case_whitson_problem_18_VDW, input_properties_case_7_psudocomponents,
                   input_properties_case_dissertation_PR)
from eos import PengRobinsonEos, SoaveRedlichKwongEos, VanDerWaalsEos



def test_vapor_liquid_curves():
    temperature = 350.0 # [K]
    pressure = np.linspace(2.5, 150.0, num=200) * 1.0e5 # [Pa]

    # Create EoS object and set properties
    props = input_properties_case_whitson_problem_18_PR()
    (_, _, global_molar_fractions,
    critical_pressure, critical_temperature, acentric_factor,
    molar_mass, omega_a, omega_b, binary_interaction) = props
    eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                          omega_a, omega_b, binary_interaction)

    result = calculate_molar_fraction_curve(eos, pressure, temperature, global_molar_fractions)


    pressure_bar = pressure / 1.0e5
    plt.plot(pressure_bar, result, label='Vapor molar fraction')
    plt.plot(pressure_bar, 1.0-result, label='Liquid molar fraction')
    plt.xlabel('Pressure [bar]')
    plt.ylabel('Phase molar fraction [mol/mol]')
    plt.legend(loc='upper center')
    plt.axis([np.min(pressure_bar), np.max(pressure_bar), 0.0, 1.0])
    plt.show()

def test_different_eos():
    # PENG-ROBINSON
    props = input_properties_case_whitson_problem_18_PR()
    (_, _, global_molar_fractions,
     critical_pressure, critical_temperature, acentric_factor,
     molar_mass, omega_a, omega_b, binary_interaction) = props
    eos_pr = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                             omega_a, omega_b, binary_interaction)
    # SOAVE-REDLICH-KWONG
    props = input_properties_case_whitson_problem_18_SRK()
    (_, _, global_molar_fractions,
     critical_pressure, critical_temperature, acentric_factor,
     molar_mass, omega_a, omega_b, binary_interaction) = props
    eos_srk = SoaveRedlichKwongEos(critical_pressure, critical_temperature, acentric_factor,
                                   omega_a, omega_b, binary_interaction)
    # VAN DER WAALS
    props = input_properties_case_whitson_problem_18_VDW()
    (_, _, global_molar_fractions,
     critical_pressure, critical_temperature, acentric_factor,
     molar_mass, omega_a, omega_b, binary_interaction) = props
    eos_vdw = VanDerWaalsEos(critical_pressure, critical_temperature, acentric_factor,
                             omega_a, omega_b, binary_interaction)

    temperature = 350.0  # [K]
    pressure = np.linspace(1, 135.0, num=200) * 1.0e5  # [Pa]
    result_pr = calculate_molar_fraction_curve(eos_pr, pressure, temperature, global_molar_fractions)
    result_srk = calculate_molar_fraction_curve(eos_srk, pressure, temperature, global_molar_fractions)
    result_vdw = calculate_molar_fraction_curve(eos_vdw, pressure, temperature, global_molar_fractions)

    pressure_bar = pressure / 1.0e5
    plt.plot(pressure_bar, result_pr, label='Peng-Robinson')
    plt.plot(pressure_bar, result_srk, label='Soave-Redlich-Kwong')
    plt.plot(pressure_bar, result_vdw, label='Van der Waals')
    plt.xlabel('Pressure [bar]')
    plt.ylabel('Vapor molar fraction [mol/mol]')
    plt.legend(loc='upper center')
    plt.axis([np.min(pressure_bar), np.max(pressure_bar), 0.0, 1.0])
    plt.show()


def test_retrograde_condensation():
    # PENG-ROBINSON
    props = input_properties_case_7_psudocomponents()
    (_, _, global_molar_fractions,
     critical_pressure, critical_temperature, acentric_factor,
     molar_mass, omega_a, omega_b, binary_interaction) = props
    eos_pr = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                             omega_a, omega_b, binary_interaction)

    temperature = 350.0  # [K]
    pressure = np.linspace(1.0, 250.0, num=200) * 1.0e5  # [Pa]
    result_pr = calculate_molar_fraction_curve(eos_pr, pressure, temperature, global_molar_fractions)

    pressure_bar = pressure / 1.0e5
    plt.plot(pressure_bar, result_pr, label='Peng-Robinson')

    plt.xlabel('Pressure [bar]')
    plt.ylabel('Vapor molar fraction [mol/mol]')
    plt.legend(loc='upper center')
    plt.axis([np.min(pressure_bar), np.max(pressure_bar), 0.0, 1.0])
    plt.show()

def test_near_critical_point():
    # PENG-ROBINSON
    props = input_properties_case_dissertation_PR()
    (_, _, global_molar_fractions,
     critical_pressure, critical_temperature, acentric_factor,
     molar_mass, omega_a, omega_b, binary_interaction) = props
    eos_pr = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                             omega_a, omega_b, binary_interaction)

    # temperature = 366.5 # [K] Easy ss flash converges
    temperature = 408.0  # [K] Hard, number of iter reached!
    pressure = np.linspace(30.0, 80.0, num=200) * 1.0e5  # [Pa]
    result_pr = calculate_molar_fraction_curve(
        eos_pr, pressure,
        temperature, global_molar_fractions, print_statistics=False
    )

    pressure_bar = pressure / 1.0e5
    plt.plot(pressure_bar, result_pr, label='Peng-Robinson')

    plt.xlabel('Pressure [bar]')
    plt.ylabel('Vapor molar fraction [mol/mol]')
    plt.legend(loc='upper center')
    plt.axis([np.min(pressure_bar), np.max(pressure_bar), 0.0, 1.0])
    plt.show()