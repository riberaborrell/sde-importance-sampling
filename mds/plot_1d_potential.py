from mds.plots_1d import Plot1d
#from mds.potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES, POTENTIAL_TITLES, POTENTIAL_LABELS
from mds.potentials_and_gradients_1d import get_potential_and_gradient, POTENTIAL_NAMES
from mds.utils import get_example_data_path

import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='Plots the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='1d_sym_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
    )
    parser.add_argument(
        '--alphas',
        dest='alphas',
        nargs='+',
        type=float,
        default=[1],
        help='Set the parameters for the chosen potential. Default: [1]',
    )
    parser.add_argument(
        '--num-plots',
        dest='num_plots',
        type=int,
        default=1,
        help='Set number of plots',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # get parameters
    potential_name = args.potential_name
    alphas = np.array(args.alphas)
    num_plots = args.num_plots

    # plot potentials
    x = np.linspace(-3, 3, 1000)
    if num_plots == 1:
        # compute potential
        alpha = alphas
        potential, \
        gradient, \
        pot_formula, \
        grad_formula, \
        parameters = get_potential_and_gradient(potential_name, alpha)
        V = potential(x)

        # get plot path
        dir_path = get_example_data_path(potential_name, alpha)

        # plot
        plt1d = Plot1d(dir_path, 'potential')
        plt1d.set_ylim(bottom=0, top=alphas[0] * 10)
        plt1d.set_title(pot_formula)
        plt1d.potential(x, V, parameters)
    else:
        # compute potential
        assert alphas.shape[0] % num_plots == 0, ''

        Vs = np.zeros((num_plots, x.shape[0]))
        alpha_dim = int(alphas.shape[0] / num_plots)
        alphas = alphas.reshape((num_plots, alpha_dim))
        labels = []
        for i in range(num_plots):
            potential, \
            gradient, \
            pot_formula, \
            grad_formula, \
            parameters = get_potential_and_gradient(potential_name, alphas[i])
            labels.append(parameters)
            Vs[i] = potential(x)

        # get plot path
        dir_path = get_example_data_path(potential_name)

        # plot
        plt1d = Plot1d(dir_path, 'potentials')
        plt1d.set_ylim(bottom=0, top=alphas.max() * 2)
        plt1d.set_title(pot_formula)
        plt1d.potentials(x, Vs, labels)

if __name__ == "__main__":
    main()
