import dataclasses, typing, numpy, scipy

from objects import BinaryVaporLiquidEquilibriumLine


@dataclasses.dataclass 
class AntoineParameters():
    A: float
    B: float
    C: float
    log_type: typing.Literal["ln", "log"] = "ln"

    def P(self, T):
        if self.log_type == "ln":
            return numpy.exp(self.A - self.B / (T + self.C))
        elif self.log_type == "log":
            return 10**(self.A - self.B / (T + self.C))

def _gamma(x, T, b_12, b_21, alpha):
    R = 8.314

    tau_12 = lambda T: b_12 / R / T
    tau_21 = lambda T: b_21 / R / T
    G_12 = lambda T: numpy.exp(-alpha * tau_12(T))
    G_21 = lambda T: numpy.exp(-alpha * tau_21(T))

    ln_gamma_1 = lambda x, T: (1 - x)**2 * (tau_21(T) * (G_21(T) / (x + (1 - x) * G_21(T)))**2 + G_12(T) * tau_12(T) / ((1 - x) + x * G_12(T))**2)

    ln_gamma_2 = lambda x, T: x**2 * (tau_12(T) * (G_12(T) / ((1 - x) + x * G_12(T)))**2 + G_21(T) * tau_21(T) / (x + (1 - x) * G_21(T))**2)

    ln_gamma = numpy.array([ln_gamma_1(x, T), ln_gamma_2(x, T)])

    return numpy.exp(ln_gamma)

@dataclasses.dataclass 
class BinaryNRTLParameters():
    A: AntoineParameters = dataclasses.field(repr=False)
    B: AntoineParameters = dataclasses.field(repr=False)
    b_12: float
    b_21: float
    alpha: float

    def gamma(self, x, T):
        return _gamma(x, T, self.b_12, self.b_21, self.alpha)

    def equilibrium(self, x, P):
        P_T = lambda T: x * self.gamma(x, T)[0] * self.A.P(T) + (1 - x) * self.gamma(x, T)[1] * self.B.P(T)
        T = scipy.optimize.fsolve(lambda T: P - P_T(T), 298)[0]
        y = x * self.gamma(x, T)[0] * self.A.P(T) / P 
        return y

    def equilibrium_line(self, P):
        x = numpy.linspace(0, 1, 1000)
        y = numpy.zeros(x.shape[0])
  
        for i in range(x.shape[0]):
            y[i] = self.equilibrium(x[i], P)

        line = BinaryVaporLiquidEquilibriumLine(x, y, lambda x: self.equilibrium(x, P))

        return line