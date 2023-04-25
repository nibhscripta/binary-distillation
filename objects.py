import scipy, dataclasses, numpy, matplotlib, warnings, typing



def _evaluate_minimum_reflux_ratio(eq, x_F, x_D, q):
    '''
    Finds the minimum reflux ratio of a binary distillation column by finding the point at which the feed and enriching line intersect the equilibrium curve.

    equilibrium_line: Set of x/y data that describes relationship between the composition of a binary mixture in the liquid and vapor phases.

    x_F: Feed composition

    x_D: Distillate composition

    q: Number which describes the slope of the feed line.
    '''
    if q == 1:
        xp = x_F
    else:
        # feed line
        f_F = lambda x: q / (q - 1) * x - x_F / (q - 1)

        # intersection between feed and equilibrium lines; evaluates to 0 at x'
        f_p = lambda x: numpy.interp(x, eq.x, eq.y) - f_F(x)

        # x'
        xp = scipy.optimize.fsolve(f_p, x_F)[0]
    
    # y'
    yp = numpy.interp(xp, eq.x, eq.y)

    # minimum reflux ration
    return (x_D - yp) / (yp - xp)



@dataclasses.dataclass
class OperatingLine():
    x: numpy.ndarray
    y: numpy.ndarray
    f: typing.Optional[typing.Callable]
    xpp: float
    ypp: float



def _operating_lines_specified_reflux(x_F, x_D, x_W, q, R):
    '''
    Creates a enriching and stripping lines for a specified reflux ratio.

    x_F: Feed composition

    x_D: Distillate composition

    x_W: Worm/bottom composition

    q: Number which describes the slope of the feed line.

    R: reflux ratio
    '''
    # enriching line
    f_e = lambda x: R / (R + 1) * x + x_D / (R + 1)

    if q == 1:
        xpp = x_F
    else:
        # feed line
        f_F = lambda x: q / (q - 1) * x - x_F / (q - 1)

        # x''
        xpp = scipy.optimize.fsolve(lambda x: f_F(x) - f_e(x), x_F)[0]

    # y''
    ypp = f_e(xpp)

    x_e = numpy.array([xpp, x_D])
    y_e = f_e(x_e)

    # stripping line
    f_s = lambda x: (y_e[0] - x_W) / (x_e[0] - x_W) * (x - x_W) + x_W
    x_s = numpy.array([x_W, xpp])
    y_s = f_s(x_s)

    return OperatingLine(x_s, y_s, f_s, xpp, ypp), OperatingLine(x_e, y_e, f_e, xpp, ypp)



def _operating_lines_specified_boilup(x_F, x_D, x_W, q, B):
    '''
    Creates a enriching and stripping lines for a specified boilup ratio.

    x_F: Feed composition

    x_D: Distillate composition

    x_W: Worm/bottom composition

    q: Number which describes the slope of the feed line.

    B: Boilup ratio
    '''
    # stripping line
    f_s = lambda x: (1 + 1 / B) * x - x_W / B

    if q == 1:
        xpp = x_F
    else:
        # feed line
        f_F = lambda x: q / (q - 1) * x - x_F / (q - 1)

        # x''
        xpp = scipy.optimize.fsolve(lambda x: f_F(x) - f_s(x), x_F)[0]

    # y''
    ypp = f_s(xpp)

    x_s = numpy.array([x_W, xpp])
    y_s = f_s(x_s)

    # enriching line
    f_e = lambda x: (x_D - y_s[-1]) / (x_D - xpp) * (x - x_D) + x_D
    x_e = numpy.array([xpp, x_D])
    y_e = f_e(x_e)

    return OperatingLine(x_s, y_s, f_s, xpp, ypp), OperatingLine(x_e, y_e, f_e, xpp, ypp)



@dataclasses.dataclass
class BinaryVaporLiquidEquilibriumLine():
    x: numpy.ndarray = dataclasses.field(init=False)
    y: numpy.ndarray = dataclasses.field(init=False)
    f: typing.Optional[typing.Callable] = dataclasses.field(init=False)

    def __init__(self, *args):
        if len(args) == 1:
            self.f = args[0]
            self.x = numpy.linspace(0, 1, 1000)
            self.y = self.f(self.x)
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]



@dataclasses.dataclass
class BinaryDistillationOperatingLine():
    x_F: float
    x_D: float
    x_W: float
    equilibrium: BinaryVaporLiquidEquilibriumLine
    q: float
    R: float = None
    B: float = None
    R_min: float = dataclasses.field(init=False)
    e: OperatingLine = dataclasses.field(init=False)
    s: OperatingLine = dataclasses.field(init=False)
    x: numpy.ndarray = dataclasses.field(init=False)
    y: numpy.ndarray = dataclasses.field(init=False) 

    def __post_init__(self):
        self.R_min = _evaluate_minimum_reflux_ratio(self.equilibrium, self.x_F, self.x_D, self.q)

        if self.R is None and self.B is None:
            self.R = 1.3 * self.R_min
            self.s, self.e = _operating_lines_specified_reflux(self.x_F, self.x_D, self.x_W, self.q, self.R)
        elif self.R is not None:
            self.s, self.e = _operating_lines_specified_reflux(self.x_F, self.x_D, self.x_W, self.q, self.R)
        elif self.B is not None:
            self.s, self.e = _operating_lines_specified_boilup(self.x_F, self.x_D, self.x_W, self.q, self.B)

        self.x = numpy.array([*self.s.x, *self.e.x])
        self.y = numpy.array([*self.s.y, *self.e.y])

    def __str__(self):
        return f"x_F: {self.x_F}, x_D: {self.x_D}, x_W: {self.x_W}, q: {self.q}, R: {self.R}, B: {self.B}, R_min: {self.R_min}"