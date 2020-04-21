#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from decorators import timer
from plotting import Plot

from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient, \
                                     bias_potential, \
                                     bias_gradient
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
METADYNAMICS_DATA_PATH = os.path.join(MDS_PATH, 'data/metadynamics/')
METADYNAMICS_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/metadynamics/')

@timer
def main():

    # grid x coordinate
    X = np.linspace(-2, 2, 100)
    V = double_well_1d_potential(X)
    dV = double_well_1d_gradient(X)
    
    # set bias functions and weights (left well)
    omegas = np.array([4, 2, 2])
    mus = np.array([-1, -1.4, -0.3])
    sigmas = np.array([0.3, 0.3, 0.3])
    
    # set bias functions and weights (right well)
    #omegas = np.array([4, 2, 2])
    #mus = np.array([1, 1.4, 0.3])
    #sigmas = np.array([0.3, 0.3, 0.3])

    # plot tilted potential and gradient
    Vbias = bias_potential(X, omegas, mus, sigmas)
    dVbias = bias_gradient(X, omegas, mus, sigmas)
    pl = Plot(
        file_name='fake_tilted_potential_and_gradient_left_well',
        file_type='png',
        dir_path=METADYNAMICS_FIGURES_PATH,
    )
    pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)
    
        
    # save bias
    np.savez(
        os.path.join(METADYNAMICS_DATA_PATH, 'langevin1d_fake_bias_potential.npz'),
        omegas=omegas,
        mus=mus,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    main()
