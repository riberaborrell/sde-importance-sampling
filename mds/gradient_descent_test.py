import gradient_descent
import sampling
from plotting import Plot
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient

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
        default=0.004,
        help='Set learning rate. Default: 0.004',
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
    
    a = soc._as[-1]
    mus = soc._mus
    sigmas = soc._sigmas
    losses = soc._losses

    print(a)
    
    
    # initialize sample object
    samp = sampling.langevin_1d(
        beta=2,
        is_drifted=True,
    )
    samp.set_bias_potential(a, mus, sigmas)

    # plot tilted potential and gradient
    X = np.linspace(-2, 2, 100)
    V = double_well_1d_potential(X)
    dV = double_well_1d_gradient(X)
    Vbias = samp.bias_potential(X)
    U = samp.control(X)
    dVbias = samp.bias_gradient(U)

    pl = Plot(
        file_name='tilted_potential_and_gradient_gradient_descent',
        file_type='png',
        dir_path=FIGURES_PATH,
    )
    pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)

if __name__ == "__main__":
    main()
