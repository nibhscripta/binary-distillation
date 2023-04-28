import scipy, dataclasses, numpy, matplotlib, warnings



def _azeotrope_example_function(x):
    c = [14.64671599, -39.42525966, 39.32333535, -17.6497924, 2.98923655, 1.11520416]
    poly = lambda x, A, B, C, D, E, F: A * x**6 + B * x**5 + C * x**4 + D * x**3 + E * x**2 + F * x
    return poly(x, *c)



def plot_mccabe_thiele_analysis(eq, op, stages, op_lines=None):
    matplotlib.pyplot.plot(eq.x, eq.y, "tab:blue", label="Equilibrium Line")
    if op_lines == "both":
        matplotlib.pyplot.plot(op.e.x, op.e.y, "tab:orange", label="Enriching Line")
        matplotlib.pyplot.plot(op.s.x, op.s.y, "tab:green", label="Stripping Line")
    else:
        matplotlib.pyplot.plot(op.x, op.y, "tab:orange", label="Operating Line")
    matplotlib.pyplot.plot([op.x_F, op.xpp,], [op.x_F, op.ypp], "tab:red", label="Feed Line")
    matplotlib.pyplot.plot(stages.x, stages.y, "k", label="Stages")
    matplotlib.pyplot.plot(eq.x, eq.x, "tab:grey", linestyle="--")

    matplotlib.pyplot.xlabel("Liquid Composition")
    matplotlib.pyplot.ylabel("Vapor Composition")



@dataclasses.dataclass 
class BinaryStageAnalysis():
    x: float = dataclasses.field(repr=False)
    y: float = dataclasses.field(repr=False)
    N: float



def step_off_top(op, eq, E:float = 1):
    '''
    Perform a McCabe Thiele analysis by stepping down to the distillate composition to the worm composition.

    operating_line: unified operating line for a binary distillation column.

    equilibrium_line: Set of x/y data that describes relationship between the composition of a binary mixture in the liquid and vapor phases.

    E: Murphree tray efficiency.
    '''
    if E > 1 or E <= 0:
        raise ValueError("Efficiency must be between 0 and 1 or 1.")
    
    # x values at each stage
    x = [op.x[-1]]
    # y values at each stage
    y = [op.x[-1]]

    for i in range(1000):
        y.append(y[2*i])
        x_eq = numpy.interp(y[2*i], eq.y, eq.x)
        x.append(x[2*i-1] - (x[2*i-1] - x_eq) * E)
        x.append(x[2*i+1])
        y.append(numpy.interp(x[2*i+1], op.x, op.y))
        if x[2*i+1] < op.x[0]:
            partial_stage = (x[2*i] - op.x[0]) / (x[2*i]- x[2*i+1])
            x[2*i+2] = op.x[0]
            y[2*i+2] = op.x[0]
            x[2*i+1] = op.x[0]
            y[2*i+1] = y[2*i]
            N = i + partial_stage 
            break

        if i >= 999:
            N = i+1
            warnings.warn(f"Cannot get around pinch point.")

    return BinaryStageAnalysis(x, y, N)