from decorators import timer

import sampling

import argparse
import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
    
def get_parser():
    parser = argparse.ArgumentParser(description='fake metadynamics algorithm')
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 1D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--omegas',
        nargs='+',
        dest='omegas',
        type=float,
        default=[0.6, 0.8, 1.0, 0.6, 0.4],
        help='Set the weights of the bias potential. Default: 1',
    )
    parser.add_argument(
        '--mus',
        nargs='+',
        dest='mus',
        type=float,
        default=[-1.6, -1.3, -1., -0.7, -0.4],
        help='Set the means of the gaussians of the bias potential. \
              Default: []',
    )
    parser.add_argument(
        '--sigmas',
        nargs='+',
        dest='sigmas',
        type=float,
        default=[0.2, 0.2, 0.2, 0.2, 0.2],
        help='Set the weights of the bias potential. Default: 1',
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
    args.omegas = np.array(args.omegas)
    args.mus = np.array(args.mus)
    args.sigmas = np.array(args.sigmas)
    
    # initialize langevin_1d object
    samp = sampling.langevin_1d(beta=args.beta)
    
    # set bias potential
    a = args.omegas / 2
    samp.set_bias_potential(a, args.mus, args.sigmas)
    
    # plot potential and gradient
    if args.do_plots:
        samp.plot_potential_and_gradient(file_name='potential_and_gradient_fake_metadynamics')
    
    # save bias
    np.savez(
        os.path.join(DATA_PATH, 'langevin1d_bias_potential_fake_metadynamics.npz'),
        omegas=args.omegas,
        mus=args.mus,
        sigmas=args.sigmas,
    )


if __name__ == "__main__":
    main()
