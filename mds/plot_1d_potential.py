from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient
from utils import get_data_path

import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='Plots the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=['sym_1well', 'sym_2well', 'asym_2well', 'sym_3well', 'asym_3well'],
        default='sym_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
    )
    parser.add_argument(
        '--alphas',
        dest='alphas',
        nargs='+',
        type=float,
        default=np.array([1]),
        help='Set the parameter alpha for the chosen potentials. Default: [1]',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # get parameters
    potential_name = args.potential_name
    alphas = np.array(args.alphas)
    alphas.sort()

    # plot title
    if potential_name == 'sym_2well':
        title = r'$V(x ; \alpha) = \alpha (x^2 - 1)^2$'
    else:
        title = ''

    # plot potentials
    X = np.linspace(-3, 3, 1000)
    if alphas.shape[0] == 1:
        # compute potential
        potential, gradient = get_potential_and_gradient(potential_name, alphas[0])
        V = potential(X)

        # get plot path
        dir_path = get_data_path(potential_name, alphas[0])
        file_name = 'potential'

        # plot
        plot = Plot(dir_path, file_name)
        plot.set_ylim(bottom=0, top=alphas[0] * 10)
        plot.set_title(title)
        plot.potential(X, V)
    else:
        # compute potential
        Vs = np.zeros((alphas.shape[0], X.shape[0]))
        for i, alpha in enumerate(alphas):
            potential, gradient = get_potential_and_gradient(potential_name, alpha)
            Vs[i] = potential(X)

        # get plot path
        dir_path = get_data_path(potential_name)
        file_name = 'potentials'

        # plot
        plot = Plot(dir_path, file_name)
        plot.set_ylim(bottom=0, top=alphas.max() * 5)
        plot.set_title(title)
        plot.potentials(X, Vs, alphas)

if __name__ == "__main__":
    main()
