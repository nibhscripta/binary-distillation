from objects import BinaryDistillationOperatingLine, BinaryVaporLiquidEquilibriumLine
from operations import step_off_top, plot_mccabe_thiele_analysis

from numpy import sin, pi
from matplotlib.pyplot import plot, xlabel, ylabel, legend, show, grid


# Define specifications
f = lambda x: sin(pi / 2 * x)
x_F = 0.42
x_D = 0.97
x_W = 0.011
q = 1.5
R = 2.5
E = 0.8

# define equilibrium line
eq = BinaryVaporLiquidEquilibriumLine(f)
# define operating line
op = BinaryDistillationOperatingLine(x_F, x_D, x_W, eq, q, R=R)
# step off stages
stages = step_off_top(op, eq, E)

# plot
plot_mccabe_thiele_analysis(eq, op, stages)
grid(which="both", axis="both")
legend()
show()

print(stages)