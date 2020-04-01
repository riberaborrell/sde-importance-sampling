import argparse

import sampling
from plotting import Plot

import numpy as np
from datetime import datetime
import os

def get_parser():
    parser = argparse.ArgumentParser(description='1D overdamped Langevin')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--is-drifted',
        dest='is_drifted',
        type=bool,
        default=True,
        help='Set if the 1D MD SDE is drifted. Default: True',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=5,
        help='Set the parameter beta for the 1D MD SDE. Default: 5',
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

    samp = sampling.langevin_1d(
        seed=args.seed, 
        beta=args.beta,
        xzero=args.xzero,
        is_drifted=args.is_drifted,
        target_set=args.target_set,
        num_trajectories=args.M, 
    )
    if args.is_drifted:
        samp.get_a_coefficients_metadynamcs(
            m=20,
            J_min=-1.9,
            J_max=0.9,
        )

    # plot tilted potential and gradient
    X = np.linspace(-2, 2, 100)
    tilted_V = np.zeros(len(X))
    tilted_dV = np.zeros(len(X))
    for i, x in enumerate(X):
        tilted_V[i] = samp.tilted_potential(x)
        tilted_dV[i] = samp.tilted_gradient(x)
    pl = Plot(file_name='tilted_potential_and_gradient', file_type='png')
    pl.potential_and_gradient(X, tilted_V, tilted_dV)


if __name__ == "__main__":
    main()
