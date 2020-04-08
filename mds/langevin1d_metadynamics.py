import argparse

from decorators import timer
from metadynamics import metadynamics_algorithm

import numpy as np
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
METADYNAMICS_DATA_PATH = os.path.join(MDS_PATH, 'data/metadynamics')
METADYNAMICS_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/metadynamics')

def get_parser():
    parser = argparse.ArgumentParser(description='Metadynamics')
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
        '--well-set',
        nargs=2,
        dest='well_set',
        type=float,
        default=[-1.9, -0.1],
        help='Set the well set interval. Default: [-1.9, -0.1]',
    )
    parser.add_argument(
        '--k',
        dest='k',
        type=int,
        default=100,
        help='Steps before adding a bias function. Default: 100',
    )
    return parser

@timer
def main():
    args = get_parser().parse_args()

    omegas, mus, sigmas = metadynamics_algorithm(
        beta=args.beta,
        xzero=args.xzero,
        well_set=args.well_set,
        k=args.k,
        do_plots=True,
        seed=args.seed,
    )
        
    # save bias
    np.savez(
        os.path.join(METADYNAMICS_DATA_PATH, 'langevin1d_metadynamic.npz'),
        omegas=omegas,
        mus=mus,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    main()
