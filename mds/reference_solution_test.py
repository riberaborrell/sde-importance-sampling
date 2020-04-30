from reference_solution import langevin_1d_reference_solution
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient
from plotting import Plot

import matplotlib.pyplot as plt
import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

beta = 1

sol = langevin_1d_reference_solution(
    beta=beta,
    target_set_min=0.9,
    target_set_max=1.1,
)

sol.compute_reference_solution()

X = sol.omega_h
V = double_well_1d_potential(X)
DV = double_well_1d_gradient(X)
F = sol.F
Vbias = 2 * F

U = sol.u_opt
DVbias = - np.sqrt(2) * U

pl = Plot(
    file_name='potential_and_gradient_reference_solution',
    file_type='png',
    dir_path=FIGURES_PATH,
)
pl.tilted_potential_and_gradient(X, V, DV, Vbias, DVbias)
