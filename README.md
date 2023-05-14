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
plot()
```