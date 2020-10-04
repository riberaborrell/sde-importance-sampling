from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES, POTENTIAL_TITLES, POTENTIAL_LABELS
from utils import get_example_data_path

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

    # plot title
    title = POTENTIAL_TITLES[potential_name]

    # plot potentials
    X = np.linspace(-3, 3, 1000)
    if num_plots == 1:
        # compute potential
        alpha = alphas
        potential, gradient = get_potential_and_gradient(potential_name, alpha)
        V = potential(X)

        # get plot path
        dir_path = get_example_data_path(potential_name, alpha)
        file_name = 'potential'

        #label = POTENTIAL_LABELS[potential_name].format(tuple(alpha))
        label = r'a = {:2.1f}'.format(alpha[0])

        # plot
        plot = Plot(dir_path, file_name)
        plot.set_ylim(bottom=0, top=alphas[0] * 10)
        plot.set_title(title)
        plot.potential(X, V, label)
    else:
        # compute potential
        assert alphas.shape[0] % num_plots == 0, ''

        Vs = np.zeros((num_plots, X.shape[0]))
        alpha_dim = int(alphas.shape[0] / num_plots)
        alphas = alphas.reshape((num_plots, alpha_dim))
        for i in range(num_plots):
            potential, gradient = get_potential_and_gradient(potential_name, alphas[i])
            Vs[i] = potential(X)

        # get plot path
        dir_path = get_example_data_path(potential_name)
        file_name = 'potentials'

        #labels = None
        labels = [r'a = {:2.1f}'.format(float(alpha)) for alpha in alphas]

        # plot
        plot = Plot(dir_path, file_name)
        plot.set_ylim(bottom=0, top=alphas.max() * 5)
        plot.set_title(title)
        plot.potentials(X, Vs, labels)

if __name__ == "__main__":
    main()
