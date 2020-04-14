import argparse

import sampling
from metadynamics import get_a_from_metadynamics, \
                         get_a_from_fake_metadynamics

from potentials_and_gradients import double_well_1d_potential, \
                                     gradient_double_well_1d_potential

from plotting import Plot


import numpy as np
from datetime import datetime
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

def get_parser():
    parser = argparse.ArgumentParser(description='drifted 1D overdamped Langevin')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=2,
        help='Set the parameter beta for the 1D MD SDE. Default: 2',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1.,
        help='Set the value of the process at time t=0. Default: -1',
    )
    parser.add_argument(
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
    )
    parser.add_argument(
        '--num-trajectories',
        dest='M',
        type=int,
        default=10**4,
        help='Set number of trajectories to sample. Default: 10.000',
    )
    return parser


def main():
    args = get_parser().parse_args()

    #a, mus, sigmas = get_a_from_fake_metadynamics(args.beta)
    a, mus, sigmas = get_a_from_metadynamics(
         beta=args.beta,
         m=10,
         J_min=-1.9,
         J_max=0.9,
         #J_min=-0.9,
         #J_max=1.9,
    )
    samp = sampling.langevin_1d(
        beta=args.beta,
        xzero=args.xzero,
        target_set=args.target_set,
        num_trajectories=args.M, 
        seed=args.seed, 
        is_drifted=True,
        do_reweighting=True,
    )
    samp._a = a
    samp._mus = mus
    samp._sigmas = sigmas 


    # plot tilted potential and gradient
    X = np.linspace(-2, 2, 100)

    V = double_well_1d_potential(X)
    dV = gradient_double_well_1d_potential(X)
    
    # preallocate gradient and bias potential
    Vbias = np.zeros(len(X))
    dVbias = np.zeros(len(X))
    for i, x in enumerate(X):
        # evaluate gradien and bias potential at x
        Vbias[i] = samp.bias_potential(x)
        u_at_x = samp.control(x)
        dVbias[i] = samp.bias_gradient(u_at_x)

    pl = Plot(
        file_name='tilted_potential_and_gradient',
        file_type='png',
        dir_path=FIGURES_PATH,
    )
    pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)

    samp.preallocate_variables()
    samp.sample()
    samp.compute_statistics()
    samp.save_statistics()


if __name__ == "__main__":
    main()
