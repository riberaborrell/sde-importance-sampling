import sampling
#from metadynamics import get_a_from_metadynamics, \
#                         get_a_from_fake_metadynamics
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient
from plotting import Plot

import argparse
from datetime import datetime

import numpy as np

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
        '--M',
        dest='M',
        type=int,
        default=10**4,
        help='Set number of trajectories to sample. Default: 10.000',
    )
    parser.add_argument(
        '--dt',
        dest='dt',
        type=float,
        default=0.001,
        help='Set dt. Default: 0.001',
    )
    parser.add_argument(
        '--N',
        dest='N',
        type=int,
        default=10**5,
        help='Set number of time steps. Default: 100.000',
    )
    return parser


def main():
    args = get_parser().parse_args()
    
    # initialize langevin_1d object
    samp = sampling.langevin_1d(
        beta=args.beta,
        is_drifted=True,
    )

    # set bias potential
    samp.set_bias_potential_from_metadynamics(
         m=10,
         J_min=-1.9,
         J_max=0.9,
         #J_min=-0.9,
         #J_max=1.9,
    )

    # set sampling and Euler-Majurama parameters
    samp.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M, 
        target_set=args.target_set,
        dt=args.dt,
        N=args.N,
    )
    samp.preallocate_variables(
        is_sampling_problem=True,
        do_reweighting=True,
    )
    samp.sample()
    samp.compute_statistics()
    samp.save_statistics()
    exit()
    
    # plot tilted potential and gradient
    X = np.linspace(-2, 2, 100)
    V = double_well_1d_potential(X)
    dV = double_well_1d_gradient(X)
    Vbias = samp.bias_potential(X)
    U = samp.control(X)
    dVbias = samp.bias_gradient(U)

    pl = Plot(
        file_name='tilted_potential_and_gradient',
        file_type='png',
        dir_path=FIGURES_PATH,
    )
    pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)
    

if __name__ == "__main__":
    main()
