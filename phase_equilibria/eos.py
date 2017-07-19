import numpy as np

from utils import calculate_roots_of_cubic_equation

R = 8.314  # [kg/(J.K)]
SQRT_2 = np.sqrt(2)


def find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type):
    roots = calculate_roots_of_cubic_equation(p0, p1, p2, p3)
    if len(roots) > 1:
        assert len(roots) == 3, 'Size of roots has to be 1 or 3!'

        roots.sort()
        if fluid_type is 'liquid':
            correct_root = roots[0]  # smallest
        else:
            assert fluid_type is 'vapor', 'Wrong fluid type! ' + fluid_type
            correct_root = roots[2]  # largest
    else:
        correct_root = roots[0]

    assert correct_root > 0.0, fluid_type + ' Z-factor < 0.0! %f' % correct_root

    return correct_root


class EquationOfState:
    def __init__(
        self,
        Pc, 
        Tc,
        ω,
        Ω_a,
        Ω_b,
        κ_ij
    ):
        self.Pc   = Pc
        self.Tc   = Tc
        self.ω    = ω
        self.Ω_a  = Ω_a
        self.Ω_b  = Ω_b
        self.κ_ij = κ_ij

    def α_function(self, T, Tc, ω):
        raise NotImplementedError("This function has not been implemented for this class!")

    def update_eos_coefficients(self, P, T, x):
        κ_ij = self.κ_ij

        α = self.α_function(T, self.Tc, self.ω)

        self.a = (self.Ω_a * α * (R * self.Tc) ** 2) / self.Pc
        self.b = (self.Ω_b * R * self.Tc) / self.Pc
        self.a *= P / (R * T) ** 2
        self.b *= P / (R * T)
        
        A_ij = (1.0 - κ_ij) * np.sqrt(np.einsum('i,j', self.a, self.a))

        # This variables will be used in the fugacity expression
        self.A_ij_x_j = np.einsum('ij,j', A_ij, x)
        
        self.a_mix = np.dot(np.dot(x, A_ij), x)
        self.b_mix = np.sum(x * self.b)
        
    def calculate_normalized_gibbs_energy(self, f, x):
        g = (x * np.log(f)).sum()
        return g
    
    def calculate_fugacities_with_minimum_gibbs_energy(self, P, T, x):
        # TODO: Work in progress, calculate fugacities by 
        # calculating all roots and if it has two possible roots
        # calculate both minimim gibbs energy and choose the 
        # group of fugacities with minimum gibbs energy

        self.update_eos_coefficients(P, T, x)

        Z = self.calculate_eos_roots()

        f0 = self.calculate_fugacities(P, T, Z[0], x)

        if len(Z) == 1:
            return f0

        if Z[1] < 0.0:
            return f0

        f1 = self.calculate_fugacities(P, T, Z[1], x)

        g0 = self.calculate_normalized_gibbs_energy(f0, x)
        g1 = self.calculate_normalized_gibbs_energy(f1, x)

        if g0 < g1:
            return f0
        else:
            return f1


class VanDerWaalsEos(EquationOfState):
    def α_function(self, T, Tc, ω):
        return 1.0

    def calculate_eos_roots(self, fluid_type=None):
        A_mix = self.a_mix
        B_mix = self.b_mix

        p0 = 1.0
        p1 = - (B_mix + 1.0)
        p2 = A_mix
        p3 = - A_mix * B_mix

        if fluid_type is None:
            return calculate_roots_of_cubic_equation(p0, p1, p2, p3)
        else:
            return find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type)

    def calculate_fugacities(self, P, T, Z, x):
        ln_f = self.b / (Z - self.b_mix) - np.log(Z - self.b_mix) - 2.0 * self.A_ij_x_j / Z

        return (x * P) * np.exp(ln_f)  # [Pa]


class PengRobinsonEos(EquationOfState):
    def α_function(self, T, Tc, ω):
        m = np.where(
            ω < 0.49,
            0.374640 + 1.54226 * ω - 0.269920 * (ω ** 2),
            0.379642 + 1.48503 * ω - 0.164423 * (ω ** 2) + 0.016667 * (ω ** 3)
        )
        return (1.0 + m * (1.0 - np.sqrt(T / Tc))) ** 2

    def calculate_eos_roots(self, fluid_type=None):
        A_mix = self.a_mix
        B_mix = self.b_mix

        p0 = 1.0
        p1 = - (1.0 - B_mix)
        p2 = A_mix - 3.0 * (B_mix ** 2) - 2.0 * B_mix
        p3 = -(A_mix * B_mix - B_mix ** 2 - B_mix ** 3)

        if fluid_type is None:
            return calculate_roots_of_cubic_equation(p0, p1, p2, p3)
        else:
            return find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type)

    def calculate_fugacities(self, P, T, Z, x):

        ln_f = (self.b / self.b_mix) * (Z - 1.0) - np.log(Z - self.b_mix) \
               + (self.a_mix / (2.0 * SQRT_2 * self.b_mix)) \
                 * ((self.b / self.b_mix) - 2.0 * self.A_ij_x_j / self.a_mix) \
                 * np.log((Z + (1.0 + SQRT_2) * self.b_mix) / (Z + (1.0 - SQRT_2) * self.b_mix))

        return (x * P) * np.exp(ln_f)  # [Pa]


class SoaveRedlichKwongEos(EquationOfState):
    def α_function(self, T, Tc, ω):
        m = 0.480 + 1.574 * ω - 0.176 * (ω ** 2)
        return (1.0 + m * (1.0 - np.sqrt(T / Tc))) ** 2

    def calculate_eos_roots(self, fluid_type=None):
        A_mix = self.a_mix
        B_mix = self.b_mix

        p0 = 1.0
        p1 = -1.0
        p2 = A_mix - B_mix - (B_mix ** 2)
        p3 = -(A_mix * B_mix)

        if fluid_type is None:
            return calculate_roots_of_cubic_equation(p0, p1, p2, p3)
        else:
            return find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type)

    def calculate_fugacities(self, P, T, Z, x):
        ln_f = (self.b / self.b_mix) * (Z - 1.0) - np.log(Z - self.b_mix) \
               + (self.a_mix / self.b_mix) \
                 * ((self.b / self.b_mix) - 2.0 * self.A_ij_x_j / self.a_mix) \
                 * np.log(1.0 + (self.b_mix / Z))

        return (x * P) * np.exp(ln_f)  # [Pa]
