import dataclasses
import numpy
import matplotlib.pyplot
import warnings

from specifications import OperatingLine



def plot_mccabe_thiele_analysis(eq, op, stages):
    matplotlib.pyplot.plot(eq.x, eq.y, "tab:blue", label="Equilibrium Line")
    matplotlib.pyplot.plot(op.e.x, op.e.y, "tab:orange", label="Enriching Line")
    matplotlib.pyplot.plot(op.s.x, op.s.y, "tab:green", label="Stripping Line")
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



def step_off_top(op_line, eq, E:float = 1):
    '''
    Perform a McCabe Thiele analysis by stepping down to the distillate composition to the worm composition.

    operating_line: unified operating line for a binary distillation column.

    equilibrium_line: Set of x/y data that describes relationship between the composition of a binary mixture in the liquid and vapor phases.

    E: Murphree tray efficiency.
    '''
    op = OperatingLine([*op_line.s.x, *op_line.e.x], [*op_line.s.y, *op_line.e.y])

    if E > 1 or E <= 0:
        raise ValueError("Efficiency must be between 0 and 1 or 1.")
    
    # x values at each stage
    x = [op.x[-1]]
    # y values at each stage
    y = [op.x[-1]]

    for i in range(1000):
        y.append(y[-1])
        x_eq = numpy.interp(y[-1], eq.y, eq.x)
        x.append(x[-1] - (x[-1] - x_eq) * E)
        x.append(x[-1])
        y.append(numpy.interp(x[-1], op.x, op.y))
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

    return BinaryStageAnalysis(x, y, N)



def step_off_bottom(op, eq, E:float = 1, B_E=1):
    if E > 1 or E <= 0:
        raise ValueError("Efficiency must be between 0 and 1 or 1.")

    if B_E > 1 or B_E <= 0:
        raise ValueError("Reboiler efficiency must be between 0 and 1 or 1.")

    x = [op.x[0]]
    y = [op.x[0]]

    for i in range(1000):
        x.append(x[-1])
        y_eq = numpy.interp(x[-1], eq.x, eq.y)
        if i == 0:
            y.append(y[-1] - (y[-1] - y_eq) * B_E)
        else:
            y.append(y[-1] - (y[-1] - y_eq) * E)
        y.append(y[-1])
        x.append(numpy.interp(y[-1], op.y, op.x))
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
        
    return BinaryStageAnalysis(x, y, N)