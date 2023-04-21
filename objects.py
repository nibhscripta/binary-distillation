import scipy, dataclasses, numpy, matplotlib, warnings, typing, pydantic

@dataclasses.dataclass 
class VaporLiquidEquilibriumLine():
    x: numpy.ndarray
    y: numpy.ndarray

@dataclasses.dataclass 
class BinaryDistillatioOperatingLine(VaporLiquidEquilibriumLine):
    x_F: float
    x_D: float
    x_W: float

@dataclasses.dataclass 
class McCabeThieleAnalysis():
    x: numpy.ndarray
    y: numpy.ndarray
    N: float
    eq: VaporLiquidEquilibriumLine
    op: BinaryDistillatioOperatingLine

    def plot(self):
        plot = matplotlib.pyplot.plot
        plot(self.eq.x, self.eq.y, label="Equilibrium Line")
        plot(self.op.x, self.op.y, label="Operating Line")
        plot([self.op.x_F, self.op.x[999]], [self.op.x_F, self.op.y[999]], label="Feed Line")
        plot(self.eq.x, self.eq.x, "tab:grey", linestyle="--")
        plot(self.x, self.y, "k", label="Stages")
        matplotlib.pyplot.xlabel("Liquid Composition")
        matplotlib.pyplot.ylabel("Vapor Composition")

    def __str__(self):
        return f"Column with {self.N} theoretical stages."

@pydantic.dataclasses.dataclass 
class AntoineParameters():
    A: float
    B: float
    C: float
    log_type: typing.Literal["ln", "log"]

    def P(self, T):
        if self.log_type == "ln":
            return numpy.exp(self.A - self.B / (T + self.C))
        elif self.log_type == "log":
            return 10**(self.A - self.B / (T + self.C))
        
@dataclasses.dataclass 
class RaoultsLawVaporLiquidEquilibrium():
    P_A: AntoineParameters
    P_B: AntoineParameters
    P: float
    Line: VaporLiquidEquilibriumLine = dataclasses.field(init=False)

    def __str__(self):
        return f"Equilibrium at {self.P}. P_A: {self.P_A} P_B: {self.P_B}"

    def __post_init__(self):
        x = numpy.linspace(0, 1, 1000)
        y = numpy.zeros(x.shape[0])
        P = self.P
        for i in range(x.shape[0]):
            x_i = x[i]
            P_T = lambda T: x_i * self.P_A.P(T) + (1 - x_i) * self.P_B.P(T)
            T = scipy.optimize.fsolve(lambda T: P - P_T(T), 298)[0]
            y[i] = x_i * self.P_A.P(T) / P 
        
        self.Line = VaporLiquidEquilibriumLine(x, y)