import scipy, dataclasses, numpy, matplotlib, warnings

from objects import VaporLiquidEquilibriumLine, McCabeThieleAnalysis, BinaryDistillatioOperatingLine

def evaluate_equilibrium_line(equilibrium_relationshiop):
    '''
    Create an equilibrium line from an arbitray funcntion that describes relationship between the composition of a binary mixture in the liquid and vapor phases. 

    equilibrium_relationshiop: y=f(x)
    '''
    try:
        equilibrium_relationshiop(0)
    except:
        raise TypeError("Argument equilibrium_relationship must take one positional argument only.")
    x = numpy.linspace(0, 1, 1000)

    return VaporLiquidEquilibriumLine(x, equilibrium_relationshiop(x))



def _evaluate_minimum_reflux_ratio(equilibrium_line: VaporLiquidEquilibriumLine, x_F, x_D, q):
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
        f_p = lambda x: numpy.interp(x, equilibrium_line.x, equilibrium_line.y) - f_F(x)

        # x'
        xp = scipy.optimize.fsolve(f_p, x_F)[0]
    
    # y'
    yp = numpy.interp(xp, equilibrium_line.x, equilibrium_line.y)

    # minimum reflux ration
    return (x_D - yp) / (yp - xp)



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

    x_e = numpy.linspace(xpp, x_D, 1000)
    y_e = f_e(x_e)
    # enriching line array
    e = VaporLiquidEquilibriumLine(x_e, y_e)

    # stripping line
    f_s = lambda x: (e.y[0] - x_W) / (e.x[0] - x_W) * (x - x_W) + x_W
    x_s = numpy.linspace(x_W, xpp, 1000)
    y_s = f_s(x_s)
    # stripping line array
    s = VaporLiquidEquilibriumLine(x_s, y_s)

    return s, e, xpp, ypp



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

    x_s = numpy.linspace(x_W, xpp, 1000)
    y_s = f_s(x_s)
    # stripping line array
    s = VaporLiquidEquilibriumLine(x_s, y_s)

    # enriching line
    f_e = lambda x: (x_D - s.y[-1]) / (x_D - xpp) * (x - x_D) + x_D
    x_e = numpy.linspace(xpp, x_D, 1000)
    y_e = f_e(x_e)
    # enriching line array
    e = VaporLiquidEquilibriumLine(x_e, y_e)

    return s, e, xpp, ypp



def binary_distillation_operating_line(x_F, x_D, x_W, q, equilibrium_line: VaporLiquidEquilibriumLine, R=None, B=None):
    '''
    Creates a unified operating line spanning from the worm composition to the distillate composition.

    x_F: Feed composition

    x_D: Distillate composition

    x_W: Worm/bottom composition

    q: Number which describes the slope of the feed line.

    equilibrium_line: Set of x/y data that describes relationship between the composition of a binary mixture in the liquid and vapor phases.

    R (Optional): Reflux ratio

    B (Optional): Boilup ratio
    '''

    # If no reflux or boilup ratio is specified, find the mimimum reflux ration and use 1.3 times the minimum for R.
    if R is None and B is None:
        warnings.warn("No reflux or boilup ratio specified. Using 1.3 times R_min.")

        R = 1.3 * _evaluate_minimum_reflux_ratio(equilibrium_line, x_F, x_D, q)       
    
    # reflux or boilup ratio cannot both be specified
    if R is not None and B is not None:
        raise Warning("System is overspecified. Specify either reflux or boilup ratio, not both.")
    
    # for specified reflux ratio
    if R is not None:
        s, e, xpp, ypp = _operating_lines_specified_reflux(x_F, x_D, x_W, q, R)

    # for specified boilup ratio
    elif B is not None:
        s, e, xpp, ypp = _operating_lines_specified_boilup(x_F, x_D, x_W, q, B)

    # operating line cannot extend past equilivrium curve, violating the 1st law of thermodynamics
    if ypp > numpy.interp(xpp, equilibrium_line.x, equilibrium_line.y):
        warnings.warn("Process is infeasible. Operating line extends above equilibrium line.")

    # pack operating lines
    x = numpy.array([*s.x, *e.x])
    y = numpy.array([*s.y, *e.y])
    op = BinaryDistillatioOperatingLine(x, y, x_F, x_D, x_W)

    return op
    

def step_off_stages(operating_line: VaporLiquidEquilibriumLine, equilibrium_line: VaporLiquidEquilibriumLine, E:float = 1):
    '''
    Perform a McCabe Thiele analysis by stepping down to the distillate composition to the worm composition.

    operating_line: unified operating line for a binary distillation column.

    equilibrium_line: Set of x/y data that describes relationship between the composition of a binary mixture in the liquid and vapor phases.

    E: Murphree tray efficiency.
    '''
    if E > 1 or E <= 0:
        raise ValueError("Efficiency must be between 0 and 1 or 1.")
    
    # x values at each stage
    x = [operating_line.x[-1]]
    # y values at each stage
    y = [operating_line.x[-1]]

    for i in range(1000):
        y.append(y[2*i])
        x_eq = numpy.interp(y[2*i], equilibrium_line.y, equilibrium_line.x)
        x.append(x[2*i-1] - (x[2*i-1] - x_eq) * E)
        x.append(x[2*i+1])
        y.append(numpy.interp(x[2*i+1], operating_line.x, operating_line.y))
        if x[2*i+1] < operating_line.x[0]:
            partial_stage = (x[2*i] - operating_line.x[0]) / (x[2*i]- x[2*i+1])
            x[2*i+2] = operating_line.x[0]
            y[2*i+2] = operating_line.x[0]
            x[2*i+1] = operating_line.x[0]
            y[2*i+1] = y[2*i]
            N = i + partial_stage 
            break

        if i >= 999:
            N = i+1
            warnings.warn(f"Cannot get around pinch point.")
    
    return McCabeThieleAnalysis(x, y, N, equilibrium_line, operating_line)