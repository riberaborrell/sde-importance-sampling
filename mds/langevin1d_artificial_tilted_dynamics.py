#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from decorators import timer
from plotting import Plot

from tools import double_well_1d_potential, \
                  gradient_double_well_1d_potential, \
                  bias_potential, \
                  bias_potential2, \
                  gradient_bias_potential, \
                  gradient_bias_potential2
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

@timer
def main():

    # grid x coordinate
    X = np.linspace(-2, 2, 100)
    V = double_well_1d_potential(X)
    dV = gradient_double_well_1d_potential(X)

    # preallocate mean and standard deviation for the bias funcitons
    mus = np.array([-1, -1.1])
    sigmas = np.array([0.4, 0.4])
    omegas = np.array([0.4, 0.3])

    # plot tilted potential
    Vbias = bias_potential2(
        X=X,
        mus=mus,
        sigmas=sigmas,
        omegas=omegas,
    )
    pl = Plot(file_name='tilted_pot', file_type='png')
    pl.tilted_potential(X, V, Vbias)
    
    # plot gradient of the tilted potential
    dVbias = gradient_bias_potential2(
        X=X,
        mus=mus,
        sigmas=sigmas,
        omegas=omegas,
    )
    pl = Plot(file_name='tilted_grad', file_type='png')
    pl.gradient_tilted_potential(X, dV, dVbias)
        
    # save bias
    np.savez(
        os.path.join(DATA_PATH, 'langevin1d_tilted_potential.npz'),
        omegas=omegas,
        mus=mus,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    main()
