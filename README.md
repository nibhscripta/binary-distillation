# binary-distillation

## Creating an equilibrium line

```python
from objects import BinaryVaporLiquidEquilibriumLine
from numpy import sin, pi

f = lambda x: sin(pi / 2 * x)
equilibrium = BinaryVaporLiquidEquilibriumLine(f)
```

Or, use data.

```python
x = numpy.array([0, ..., 1])
y = numpy.array([0, ..., 1])
equilibrium = BinaryVaporLiquidEquilibriumLine(x, y)
```

Optionally, fit the data to a curve.

```python
equilibrium.fit_curve()
```

Or, specify an equation to fit the data to.

```python
f = lambda x, A, B, C: A * x**3 + B * x**2 + C * x
equilibrium.fit_curve(f)
```

```BinaryVaporLiquidEquilibriumLine.fit``` uses ```scipy.optimize.curve_fit```, and so the arguments of the input function must be the independent variable followed by the coefficients of the function.

## Creating an operating line

```python
from objects import BinaryDistillationOperatingLine

x_F = 0.5
x_D = 0.8
x_W = 0.1
q = 1

operating = BinaryDistillationOperatingLine(x_F, x_D, x_W, equilibrium, q)
```

```BinaryDistillationOperatingLine``` depends on an existing, defined equilibrium line object.

Optionally, specify either the reflux or boilup ratio.

```python
RR = 2.5

operating = BinaryDistillationOperatingLine(x_F, x_D, x_W, equilibrium, q, R=RR)
```

## Performing a McCabe-Thiele analysis

```python
from operations import step_off_top

stages = step_off_top(operating, equilibrium)
```

```step_off_top``` returns an object that contains the x/y values for each point in the stage calculations and the number of theoretical stages.

Optionally, specify the Murphree tray efficiency.

```python
efficiency = 0.75

stages = step_off_top(operating, equilibrium, E=efficiency)
```

Plot the results.

```python
from operations import plot_mccabe_thiele_analysis

plot_mccabe_thiele_analysis(operating, equilibrium, stages)
```

Retrieve the number of theoretical stages.

```python
print(stages.N)
```