import argparse

import sampling
from metadynamics import get_a_from_metadynamics

import numpy as np
from datetime import datetime
import os

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
        default=10**3,
        help='Set number of trajectories to sample. Default: 1.000',
    )
    return parser


def main():
    args = get_parser().parse_args()

    a, mus, sigmas = get_a_from_metadynamics(
         beta=args.beta,
         m=20,
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

    samp.preallocate_variables()
    samp.sample()
    samp.compute_statistics()
    samp.save_statistics()


if __name__ == "__main__":
    main()
