# Installation

`!pip install git+https://github.com/nibhscripta/binary-distillation`

# Basic Usage

```python
# imports
from matplotlib.pyplot import show, legend

from pystill import DistillationColumn, EquilibriumLine

# Function which describes the equilibrium relationship
f = lambda x: x**0.5

# Create an equilibrium line from the function
eq = EquilibriumLine.from_function(f)

# Specify compositions
x_F = 0.5
x_D = 0.9
x_W = 0.1
# Feed quality
q = 1

# Create a distillation column object
op = DistillationColumn(x_F, x_D, x_W, q, eq)

# Perform a stage analysis on the distillation column object
op.design_stages()

# PLot the McCabe Thiele analysis
op.plot()
legend()
show()
```

Result:

![McCabe Thiele Analysis](https://github.com/nibhscripta/binary-distillation/blob/main/McCabe_Thiele.png)

# Additional Usage 

## Defining Equilibrium Lines from Data

Equilibrium lines can be defined from data. Supply two lists containing values for the liquid and vapor compositions at equilibrium. Optionally, fit a curve to the data. By defaults, `EquilibriumLine.fit_curve` fits a polynomial.

```python
from pystill import EquilibriumLine
# data
x = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.96, 0.98, 1]
y = [0, 0.192, 0.377, 0.527, 0.656, 0.713, 0.746, 0.771, 0.794, 0.822, 0.858, 0.912, 0.942, 0.959, 0.978, 1]

# define equilibrium line
eq = EquilibriumLine(x, y)
# fit a curve to the data
eq.fit_curve()
```

## Defining an Equilibrium Line Using Rault's Law

Rauolt's Law is a common way of modelling vapor/liquid equilibrium. `RaultsLawEquilibrium` is a class that defines a system of two molecules. This class requires that two functions be supplied that relate the vapor pressure of the two molecules in the system to temperature. Specify your own, or use the `AntoineEquation` class to define constants for the Antoine's equation.

```python
from pystill import RaultsLawEquilibrium, AntoineEquation

# Define Antoine's Equation constants for molecules A and B
A = AntoineEquation(16.5785, 3638.27, 239.500)
B = AntoineEquation(16.3872, 3885.70, 230.170)

# Define a RaultsLawEquilibrium class using the Antoine's Equation for molecules A and B
rl = RaultsLawEquilibrium(A.P, B.P)

# Create an equilibrium line at P=101.325 using the equilibrium_line method
eq = rl.equilibrium_line(101.325)
```

By default, `RaultsLawEquilibrium` uses 1 as activity coefficients of the two molecules. However, it is possible to specify your own. After creating an instance of the `RaultsLawEquilibrium` class, redifine the methods `gamma_1` and `gamma_2` with functions that describe the activity coefficients of the two molecules. These functions must take arguments `(x, T)` where x is the composition of the first molecule in the liquid phase and T is the temperature. 

```python
from pystill import RaultsLawEquilibrium, AntoineEquation

# Define Antoine's Equation constants for molecules A and B
A = AntoineEquation(16.5785, 3638.27, 239.500)
B = AntoineEquation(16.3872, 3885.70, 230.170)

# Define a RaultsLawEquilibrium class using the Antoine's Equation for molecules A and B
rl = RaultsLawEquilibrium(A.P, B.P)

# gamma relations for molecules A and B
from numpy import exp

rl.gamma_1 = lambda x, T: exp(1 - x)
rl.gamma_2 = lambda x, T: exp(x)

# Create an equilibrium line at P=101.325 using the equilibrium_line method
eq = rl.equilibrium_line(101.325)
```

## Defining an Equilibrium Line Using the NRTL Equation

It is possible to create an equilibrium line by using the NRTL equation. It is created in a similar way to the equilibrium lines from Rault's Law. Specify functions that describe the vapor pressure of each molecule as a function of temperature, and specify values for $b_{12}$, $b_{21}$, and $\alpha$ in the NRTL equation.

```python
from pystill import AntoineEquation, NRTL

A = AntoineEquation(16.5785, 3638.27, 239.500)
B = AntoineEquation(16.3872, 3885.70, 230.170)

# Define an NRTL system
b_12 = -253.88
b_21 = 845.21
alpha = 0.2994

nrtl = NRTL(A.P, B.P, b_12, b_21, alpha)

eq = nrtl.equilibrium_line(101.325)
```