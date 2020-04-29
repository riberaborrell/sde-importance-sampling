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
        default=1.,
        help='Set learning rate. Default:',
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=15,
        help='Set number of epochs. Default: 15',
    )
    parser.add_argument(
        '--M',
        dest='M',
        type=int,
        default=2*10**3,
        help='Set number of trajectories to sample. Default: 2000',
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
    
    soc._do_plots = True
    soc.gradient_descent()
    soc.save_statistics()
    
    # plot tilted potential and gradient
    if args.do_plots:
        soc.plot_tilted_potentials()

if __name__ == "__main__":
    main()
