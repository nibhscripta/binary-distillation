import typing

from dataclasses import dataclass, field

R = 8.31446261815324

@dataclass
class XYLine():
    x: list = field(repr=False)
    y: list = field(repr=False)

    def __init_check__(self):
        from numpy import asarray
        self.x = asarray(self.x)
        self.y = asarray(self.y)

        if (self.x.min() < 0) or (self.y.min() < 0) or (self.x.max() > 1) or (self.y.max() > 1):
            raise TypeError("Vapor/Liquid composition must be between 0 and 1.")
        
    def __post_init__(self):
        self.__init_check__()


def _one_d_intersections(x1, y1, x2, y2, iterations=10, tol=1e-8):
    from numpy import interp
    from scipy.optimize import fsolve
    f = lambda x: interp(x, x1, y1) - interp(x, x2, y2)
    sols = []
    for i in range(iterations):
        x = fsolve(f, x1[int(x1.shape[0] * (i/iterations))])[0]
        y = interp(x, x1, y1)
        if [x, y] not in sols:
            sols.append([x, y])
    tolerant_sols = []
    tolerant_val = None
    for i, val in enumerate(sols):
        y_1i = interp(val[0], x1, y1)
        y_2i = interp(val[0], x2, y2)
        meets_tol = abs(y_2i - y_1i) <= tol
        within_x1 = (val[0] <= x1.max()) and (val[0] >= x1.min())
        within_x2 = (val[0] <= x2.max()) and (val[0] >= x2.min())
        not_already_found = tolerant_val not in tolerant_sols
        if meets_tol and within_x1 and within_x2 and not_already_found:
            tolerant_sols.append(val)
    return tuple(tolerant_sols)


def _is_azeotrope(x, y, tol=1e-8):
    from numpy import array
    identity_x = array([0, 1])
    identity_y = identity_x
    intersections = _one_d_intersections(x, y, identity_x, identity_y)
    if intersections == []:
        return False
    for val in intersections:
        middle = (val[0] >= tol) and (val[0] <= (1 - tol))
        if middle:
            return val[0]
    return False


@dataclass
class EquilibriumLine(XYLine):
    azeo_x: float = field(init=False, default=None)

    def __post_init__(self): 
        self.__init_check__()

        azeo_x = _is_azeotrope(self.x, self.y)

        if azeo_x:
            self.azeo_x = azeo_x

    def from_function(f):
        from numpy import linspace
        x = linspace(0, 1, 1000)
        y = f(x)

        return EquilibriumLine(x, y)

    def fit_curve(self, function=None, deg=10):
        from numpy import linspace

        if function is not None:
            from scipy.optimize import curve_fit
            covs, _ = curve_fit(function, self.x, self.y)

            f = lambda x: function(x, *covs)
        else:
            from pystill.solvers import poly_fit
            f = poly_fit(self.x, self.y, deg)

        self.x = linspace(0, 1, 1000)
        self.y = f(self.x)


@dataclass 
class AntoineEquation():
    A: float
    B: float
    C: float
    log_type: typing.Literal["ln", "log"] = "ln"

    def P(self, T):
        if self.log_type == "ln":
            from numpy import exp
            return exp(self.A - self.B / (T + self.C))
        elif self.log_type == "log":
            return 10**(self.A - self.B / (T + self.C))


_ideal_gamma = lambda x, T: 1

@dataclass 
class RaultsLawEquilibrium():
    P_1: typing.Callable
    P_2: typing.Callable
    gamma_1: typing.Callable = _ideal_gamma
    gamma_2: typing.Callable = _ideal_gamma

    def equilibrium(self, x, P):
        from scipy.optimize import fsolve

        def P_T(T):
            return x * self.gamma_1(x, T) * self.P_1(T) + (1 - x) * self.gamma_2(x, T) * self.P_2(T)

        T = fsolve(lambda T: P - P_T(T), 298)[0]
        y = x * self.gamma_1(x, T) * self.P_1(T) / P 
        return y
    
    def equilibrium_line(self, P):
        from numpy import linspace, zeros

        x = linspace(0, 1, 100)
        y = zeros(x.shape[0])
  
        for i in range(x.shape[0]):
            y[i] = self.equilibrium(x[i], P)

        line = EquilibriumLine(x, y)

        return line


class NRTL(RaultsLawEquilibrium):
    b_12: float
    b_21: float
    alpha: float

    def __init__(self, P_1, P_2, b_12, b_21, alpha):
        self.P_1 = P_1
        self.P_2 = P_2
        self.b_12 = b_12
        self.b_21 = b_21
        self.alpha = alpha

        from numpy import exp

        tau_12 = lambda T: self.b_12 / R / T
        tau_21 = lambda T: self.b_21 / R / T
        G_12 = lambda T: exp(-self.alpha * tau_12(T))
        G_21 = lambda T: exp(-self.alpha * tau_21(T))

        ln_gamma_1 = lambda x, T: (1 - x)**2 * (tau_21(T) * (G_21(T) / (x + (1 - x) * G_21(T)))**2 + G_12(T) * tau_12(T) / ((1 - x) + x * G_12(T))**2)

        ln_gamma_2 = lambda x, T: x**2 * (tau_12(T) * (G_12(T) / ((1 - x) + x * G_12(T)))**2 + G_21(T) * tau_21(T) / (x + (1 - x) * G_21(T))**2)

        self.gamma_1 = lambda x, T: exp(ln_gamma_1(x, T))
        self.gamma_2 = lambda x, T: exp(ln_gamma_2(x, T))