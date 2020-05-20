from plotting import Plot
from potentials_and_gradients import double_well_1d_potential

import numpy as np
import matplotlib.pyplot as plt

D = np.linspace(-2, 2, 1000)
V = double_well_1d_potential(D)

plot = Plot(file_name='double_well_1d_potential.png')
plot.potential(D, V)
