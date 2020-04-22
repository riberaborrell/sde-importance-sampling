from decorators import timer
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient, \
                                     bias_potential, \
                                     bias_gradient
import sampling

import argparse
import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')
    
def get_parser():
    parser = argparse.ArgumentParser(description='fake metadynamics algorithm')
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=2,
        help='Set the parameter beta for the 1D MD SDE. Default: 2',
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

    # set bias functions and weights (left well)
    omegas = np.array([4, 2, 2])
    mus = np.array([-1, -1.4, -0.3])
    sigmas = np.array([0.3, 0.3, 0.3])
    
    # set bias functions and weights (right well)
    #omegas = np.array([4, 2, 2])
    #mus = np.array([1, 1.4, 0.3])
    #sigmas = np.array([0.3, 0.3, 0.3])

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
        samp.plot_potential_and_gradient(file_name='potential_and_gradient_fake_metadynamics')
    
    # save bias
    np.savez(
        os.path.join(DATA_PATH, 'langevin1d_bias_potential_fake_metadynamics.npz'),
        omegas=omegas,
        mus=mus,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    main()
