import scipy, dataclasses, numpy, matplotlib, warnings, typing

from polynomial_fit import polynomial_fit



def _one_d_intersections(x1, y1, x2, y2, iterations=10, tol=1e-8):
    f = lambda x: numpy.interp(x, x1, y1) - numpy.interp(x, x2, y2)
    sols = []
    for i in range(iterations):
        x = scipy.optimize.fsolve(f, x1[int(x1.shape[0] * (i/iterations))])[0]
        y = numpy.interp(x, x1, y1)
        if [x, y] not in sols:
            sols.append([x, y])
    tolerant_sols = []
    tolerant_val = None
    for i, val in enumerate(sols):
        y_1i = numpy.interp(val[0], x1, y1)
        y_2i = numpy.interp(val[0], x2, y2)
        meets_tol = abs(y_2i - y_1i) <= tol
        within_x1 = (val[0] <= x1.max()) and (val[0] >= x1.min())
        within_x2 = (val[0] <= x2.max()) and (val[0] >= x2.min())
        not_already_found = tolerant_val not in tolerant_sols
        if meets_tol and within_x1 and within_x2 and not_already_found:
            tolerant_sols.append(val)
    return tuple(tolerant_sols)



def _is_azeotrope(x, y, tol=1e-8):
    identity_x = numpy.array([0, 1])
    identity_y = identity_x
    intersections = _one_d_intersections(x, y, identity_x, identity_y)
    if intersections == []:
        return False
    for val in intersections:
        middle = (val[0] >= tol) and (val[0] <= (1 - tol))
        if middle:
            return val[0]
    return False



def _valid_binary_specification_with_azeotrope(x_F, x_D, x_W, x_azeo):
    if (x_F < x_azeo) and (x_D < x_azeo) and (x_W < x_azeo):
        return True
    elif (x_F > x_azeo) and (x_D > x_azeo) and (x_W > x_azeo):
        return True
    else: 
        return False



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
    x: numpy.ndarray = dataclasses.field(repr=False)
    y: numpy.ndarray = dataclasses.field(repr=False)
    f: typing.Optional[typing.Callable] = dataclasses.field(repr=False)



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

    return OperatingLine(x_s, y_s, f_s), OperatingLine(x_e, y_e, f_e), xpp, ypp



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

    return OperatingLine(x_s, y_s, f_s), OperatingLine(x_e, y_e, f_e), xpp, ypp



def _feed_quality(q):
    if q > 1:
        return "Subcooled liquid"
    if q < 0:
        return "Superheated vapor"
    if q == 1:
        return "Saturated liquid"
    if q == 0:
        return "Saturated vapor"
    return "Vapor/Liquid mixture"



@dataclasses.dataclass
class BinaryVaporLiquidEquilibriumLine():
    x: numpy.ndarray = dataclasses.field(init=False, repr=False)
    y: numpy.ndarray = dataclasses.field(init=False, repr=False)
    f: typing.Optional[typing.Callable] = None

    def __init__(self, *args):
        if len(args) == 1:
            self.f = args[0]
            self.x = numpy.linspace(0, 1, 1000)
            self.y = self.f(self.x)
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]
        
        if self.x.min() < 0:
            raise TypeError("Liquid composition must be between 0 and 1.")
        if self.x.max() > 1:
            raise TypeError("Liquid composition must be between 0 and 1.")
        if self.y.min() < 0:
            raise TypeError("Vapor composition must be between 0 and 1.")
        if self.y.max() > 1:
            raise TypeError("Vapor composition must be between 0 and 1.")

    def fit_curve(self, function=None):
        if function is not None:
            covs, _ = scipy.optimize.curve_fit(function, self.x, self.y)

            self.f = lambda x: function(x, *covs)
        else:
            self.f = polynomial_fit(self.x, self.y)

        self.x = numpy.linspace(0, 1, 1000)
        self.y = self.f(self.x)



@dataclasses.dataclass
class BinaryDistillationOperatingLine():
    x_F: float
    x_D: float
    x_W: float
    equilibrium: BinaryVaporLiquidEquilibriumLine = dataclasses.field(repr=False)
    q: float
    R: float = None
    B: float = None
    azeo_x: float = None
    R_min: float = dataclasses.field(init=False)
    feed_quailty: str = dataclasses.field(init=False) 
    e: OperatingLine = dataclasses.field(init=False, repr=False)
    s: OperatingLine = dataclasses.field(init=False, repr=False)
    x: numpy.ndarray = dataclasses.field(init=False, repr=False)
    y: numpy.ndarray = dataclasses.field(init=False, repr=False) 
    xpp: float = dataclasses.field(init=False, repr=False)
    ypp: float = dataclasses.field(init=False, repr=False) 

    def __post_init__(self):
        azeo_x = _is_azeotrope(self.equilibrium.x, self.equilibrium.y)
        if azeo_x:
            self.azeo_x = azeo_x
            if not _valid_binary_specification_with_azeotrope(self.x_F, self.x_D,self.x_W, azeo_x):
                raise TypeError(f"Azeotrope found at x={azeo_x}. Distillation specification is invalid for an azeotrope.")

        self.R_min = _evaluate_minimum_reflux_ratio(self.equilibrium, self.x_F, self.x_D, self.q)

        if self.x_F > self.x_D:
            raise TypeError("Feed composition must be below distillate composition.")
        if self.x_F < self.x_W:
            raise TypeError("Feed composition must be above worm composition.")

        if self.R is None and self.B is None:
            self.R = 1.3 * self.R_min
            self.s, self.e, self.xpp, self.ypp = _operating_lines_specified_reflux(self.x_F, self.x_D, self.x_W, self.q, self.R)
        elif self.R is not None:
            if self.R <= self.R_min:
                raise TypeError("Infeasible system. R is below R_min.")
            self.s, self.e, self.xpp, self.ypp = _operating_lines_specified_reflux(self.x_F, self.x_D, self.x_W, self.q, self.R)
        elif self.B is not None:
            self.s, self.e, self.xpp, self.ypp = _operating_lines_specified_boilup(self.x_F, self.x_D, self.x_W, self.q, self.B)

        self.x = numpy.array([*self.s.x, *self.e.x])
        self.y = numpy.array([*self.s.y, *self.e.y])

        if self.ypp > numpy.interp(self.xpp, self.equilibrium.x, self.equilibrium.y):
            raise TypeError("Infeasible system. Operating line exceeds equilibrium line.")
        
        self.feed_quailty = _feed_quality(self.q)

    def __str__(self):
        return f"x_F: {self.x_F}, x_D: {self.x_D}, x_W: {self.x_W}, q: {self.q}, R: {self.R}, B: {self.B}, R_min: {self.R_min}"