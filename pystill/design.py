import warnings

from dataclasses import dataclass, field

from pystill.equilibrium import EquilibriumLine, XYLine


def minimum_reflux_ratio(eq, x_F, x_D, q):
    from numpy import interp
    from scipy.optimize import fsolve

    if q == 1:
        xp = x_F
    else:
        # feed line
        f_F = lambda x: q / (q - 1) * x - x_F / (q - 1)

        # intersection between feed and equilibrium lines; evaluates to 0 at x'
        f_p = lambda x: interp(x, eq.x, eq.y) - f_F(x)

        # x'
        xp = fsolve(f_p, x_F)[0]
    
    # y'
    yp = interp(xp, eq.x, eq.y)

    # minimum reflux ration
    return (x_D - yp) / (yp - xp)



@dataclass
class OperatingLine():
    x_F: float
    q: float
    x: list = field(repr=False, init=False)
    y: list = field(repr=False, init=False)


@dataclass
class EnrichingLine(OperatingLine):
    x_D: float
    R: float

    def __post_init__(self):
        from numpy import array
        from scipy.optimize import fsolve

        f_e = lambda x: self.R / (self.R + 1) * x + self.x_D / (self.R + 1)

        if self.q == 1:
            xpp = self.x_F
        else:
            # feed line
            f_F = lambda x: self.q / (self.q - 1) * x - self.x_F / (self.q - 1)

            # x''
            xpp = fsolve(lambda x: f_F(x) - f_e(x), self.x_F)[0]

        self.x = array([xpp, self.x_D])
        self.y = f_e(self.x)


@dataclass
class StrippingLine(OperatingLine):
    x_W: float
    B: float

    def __post_init__(self):
        from numpy import array
        from scipy.optimize import fsolve

        f_s = lambda x: (1 + 1 / self.B) * x - self.x_W / self.B

        if self.q == 1:
            xpp = self.x_F
        else:
            # feed line
            f_F = lambda x: self.q / (self.q - 1) * x - self.x_F / (self.q - 1)

            # x''
            xpp = fsolve(lambda x: f_F(x) - f_s(x), self.x_F)[0]

        self.x = array([self.x_W, xpp])
        self.y = f_s(self.x)


def step_off_bottom(op, eq, E:float = 1, B_E=1):
    from numpy import interp

    if E > 1 or E <= 0:
        raise ValueError("Efficiency must be between 0 and 1 or 1.")

    if B_E > 1 or B_E <= 0:
        raise ValueError("Reboiler efficiency must be between 0 and 1 or 1.")

    x = [op.x[0]]
    y = [op.x[0]]

    for i in range(1000):
        x.append(x[-1])
        y_eq = interp(x[-1], eq.x, eq.y)
        if i == 0:
            y.append(y[-1] - (y[-1] - y_eq) * B_E)
        else:
            y.append(y[-1] - (y[-1] - y_eq) * E)
        y.append(y[-1])
        x.append(interp(y[-1], op.y, op.x))
        if y[2*i+1] > op.y[-1]:
            partial_stage = (op.y[-1] - y[-3]) / (y[-1] - y[-3])
            y[-1] = op.y[-1]
            y[-2] = op.y[-1]
            x[-1] = op.x[-1]
            N = i + partial_stage
            break

        if i >= 999:
            N = i+1
            warnings.warn(f"Cannot get around pinch point.")
        
    return x, y, N


@dataclass
class DistillationColumn():
    x_F: float
    x_D: float
    x_W: float
    q: float
    equilibrium: EquilibriumLine = field(repr=False)
    R: float = None
    B: float = None
    N: float = field(init=False)
    R_min: float = field(init=False)
    e: EnrichingLine = field(init=False, repr=False)
    s: StrippingLine = field(init=False, repr=False)  
    f: XYLine = field(init=False, repr=False)
    stages: XYLine = field(default=None, repr=False)         

    def __post_init__(self):
        # validate specification
        if (self.x_D < self.x_F) or (self.x_F < self.x_W):
            raise TypeError("Feed composition must be above the worm composition and below the distillate composition.")

        # validate composition spec where an azeotrope exists
        if self.equilibrium.azeo_x is not None:
            x_azeo = self.equilibrium.azeo_x
            if (self.x_F < x_azeo) and (self.x_D < x_azeo) and (self.x_W < x_azeo):
                pass
            elif (self.x_F > x_azeo) and (self.x_D > x_azeo) and (self.x_W > x_azeo):
                pass
            else: 
                raise TypeError("Column compositions must all fall on one side of the azeotrope composition.")

        self.R_min = minimum_reflux_ratio(self.equilibrium, self.x_F, self.x_D, self.q)

        if self.R is None and self.B is None:
            self.R = 1.3 * self.R_min
        elif self.B is not None and self.R is not None:
            raise TypeError("Specify either relfux ratio or boilup ratio but not both.")
        
        if self.R is not None:
            self.e = EnrichingLine(self.x_F, self.q, self.x_D, self.R)

            m_s = (self.e.y[0] - self.x_W) / (self.e.x[0] - self.x_W)

            self.B = 1 / (m_s - 1)

            self.s = StrippingLine(self.x_F, self.q, self.x_W, self.B)
        elif self.B is not None:
            self.s = StrippingLine(self.x_F, self.q, self.x_W, self.B)

            m_e = (self.s.y[-1] - self.x_D) / (self.s.x[-1] - self.x_D)

            self.R = m_e / (1 - m_e)

            self.e = EnrichingLine(self.x_F, self.q, self.x_D, self.R)

        xpp = self.e.x[0]
        ypp = self.e.y[0]

        from numpy import interp

        yp = interp(xpp, self.equilibrium.x, self.equilibrium.y)

        if ypp >= yp:
            raise TypeError(f"Operating line cannot exceed equilibrium. y''={ypp} which is greater than y'={yp}.")

        self.f = XYLine([self.x_F, xpp], [self.x_F, ypp])

    def plot(self):
        from matplotlib.pyplot import plot

        plot(self.equilibrium.x, self.equilibrium.y, label="Equilibrium Line")
        plot(self.e.x, self.e.y, label="Enriching Line")
        plot(self.s.x, self.s.y, label="Stripping Line")
        plot(self.f.x, self.f.y, label="Feed Line")
        plot([0, 1], [0, 1], "tab:grey", linestyle="--")
        if self.stages is not None:
            plot(self.stages.x, self.stages.y, "k", label="Stages")

    def design_stages(self, E=1, B_E=1):
        op = XYLine([*self.s.x, *self.e.x], [*self.s.y, *self.e.y])
        x, y, N = step_off_bottom(op, self.equilibrium, E, B_E)

        self.stages = XYLine(x, y)
        self.N = N

# Legacy stage analysis

def step_off_top(op_line, eq, E:float = 1):
    from numpy import interp
    op = OperatingLine([*op_line.s.x, *op_line.e.x], [*op_line.s.y, *op_line.e.y])

    if E > 1 or E <= 0:
        raise ValueError("Efficiency must be between 0 and 1 or 1.")
    
    # x values at each stage
    x = [op.x[-1]]
    # y values at each stage
    y = [op.x[-1]]

    for i in range(1000):
        y.append(y[-1])
        x_eq = interp(y[-1], eq.y, eq.x)
        x.append(x[-1] - (x[-1] - x_eq) * E)
        x.append(x[-1])
        y.append(interp(x[-1], op.x, op.y))
        if x[2*i+1] < op.x[0]:
            partial_stage = (x[-3] - op.x[0]) / (x[-3]- x[-1])
            x[-1] = op.x[0]
            y[-1] = op.x[0]
            x[-2] = op.x[0]
            y[-2] = y[-3]
            N = i + partial_stage 
            break

        if i >= 999:
            N = i+1
            warnings.warn(f"Cannot get around pinch point.")

    return x, y, N