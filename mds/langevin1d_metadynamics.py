from decorators import timer
from metadynamics import metadynamics_algorithm
import sampling

import argparse
import numpy as np
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

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
        '--well-set',
        nargs=2,
        dest='well_set',
        type=float,
        default=[-1.7, 0],
        help='Set the well set interval. Default: [-1.8, -0.1]',
    )
    parser.add_argument(
        '--k',
        dest='k',
        type=int,
        default=100,
        help='Steps before adding a bias function. Default: 100',
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
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
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
        dt=args.dt,
        N=args.N,
        do_plots=args.do_plots,
        seed=args.seed,
    )
    
    # initialize langevin_1d object
    samp = sampling.langevin_1d(
        beta=args.beta,
        is_drifted=True,
    )
    
    # set bias potential
    a = omegas * args.beta / 2
    samp.set_bias_potential(a, mus, sigmas)
    
    # plot potential and gradient
    if args.do_plots:
        samp.plot_potential_and_gradient(file_name='potential_and_gradient_metadynamics')
        
    # save bias
    np.savez(
        os.path.join(DATA_PATH, 'langevin1d_bias_potential_metadynamics.npz'),
        omegas=omegas,
        mus=mus,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    main()
