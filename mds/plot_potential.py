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
        choices=['sym_1well', 'sym_2well', 'asym_2well'],
        default='sym_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
    )
    return parser

def main():
    args = get_parser().parse_args()

    potential, gradient = get_potential_and_gradient(args.potential_name)

    X = np.linspace(-3, 3, 1000)
    V = potential(X)

    dir_path = get_data_path(args.potential_name)
    file_name = 'potential'
    plot = Plot(dir_path, file_name)
    plot.potential(X, V)


if __name__ == "__main__":
    main()
