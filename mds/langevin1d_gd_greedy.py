import gradient_descent
from plotting import Plot
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient
import sampling

import argparse
import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

def get_parser():
    parser = argparse.ArgumentParser(description='Gradient Descent')
    parser.add_argument(
        '--learning-rate',
        dest='lr',
        type=float,
        default=0.002,
        help='Set learning rate. Default: 0.002',
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=10,
        help='Set number of epochs. Default: 10',
    )
    parser.add_argument(
        '--M',
        dest='M',
        type=int,
        default=10**2,
        help='Set number of trajectories to sample. Default: 100',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()
    
    # initialize gradient descent object
    soc = gradient_descent.gradient_descent(
        lr=args.lr,
        epochs=args.epochs,
        M=args.M,
    )

    soc.set_parameters_greedy(m=10)
    soc.gradient_descent()
    soc.save_statistics()
    
    # plot tilted potential and gradient
    if args.do_plots:
        samp = sampling.langevin_1d(
            beta=2,
            is_drifted=True,
        )
        
        samp.set_bias_potential(
            a=soc._as[-1],
            mus=soc._mus,
            sigmas=soc._sigmas,
        )

        samp.plot_potential_and_gradient(
            file_name='potential_and_gradient_gd_greedy',
        )

if __name__ == "__main__":
    main()
