import sampling

import argparse
from datetime import datetime

import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(description='not drifted 1D overdamped Langevin')
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
        is_drifted=False,
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
    samp.preallocate_variables(is_sampling_problem=True)
    samp.sample()
    samp.compute_statistics()
    samp.save_statistics()


if __name__ == "__main__":
    main()
