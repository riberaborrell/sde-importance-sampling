from mds.plots_2d import Plot2d
from mds.potentials_and_gradients_2d import get_potential_and_gradient, POTENTIAL_NAMES
from mds.utils import get_example_data_path

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import os

def get_parser():
    parser = argparse.ArgumentParser(description='3D Plot of the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='2d_4well',
        help='Set the type of potential to plot. Default: symmetric quadruple well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1, 1, 1, 1],
        help='Set the parameter alpha for the chosen potentials. Default: [1, 1, 1, 1]',
    )
    return parser

def main():
    args = get_parser().parse_args()
    potential_name = args.potential_name
    alpha = np.array(args.alpha)

    potential, \
    gradient, \
    pot_formula, \
    grad_formula, \
    parameters = get_potential_and_gradient(potential_name, alpha)

    dir_path = get_example_data_path(potential_name, alpha)

    # set grid
    xmin, xmax = (-2, 2)
    ymin, ymax = (-2, 2)
    h = 0.01
    x = np.arange(xmin, xmax, h)
    y = np.arange(ymin, ymax, h)
    Nx = x.shape[0]
    Ny = y.shape[0]
    xx, yy = np.meshgrid(x, y, sparse=True, indexing='ij')
    X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')

    # set bounds for the z axis
    vmin, vmax = (0, 10 * alpha.max())

    # evaluate potential
    pos = np.dstack((X, Y)).reshape((Nx * Ny, 2))
    V = potential(pos).reshape((Nx, Ny))

    # surface plot
    plt2d = Plot2d(dir_path, 'potential_surface')
    plt2d.set_title(pot_formula)
    plt2d.surface(xx, yy, V, vmin, vmax)

    # contour plot
    levels = np.logspace(-2, 1, 20, endpoint=True)
    plt2d = Plot2d(dir_path, 'potential_contour')
    plt2d.set_title(pot_formula)
    plt2d.contour(X, Y, V, vmin, vmax, levels)

    # evaluate coarsed gradient
    #k = int(xx.shape[0] / 20)
    k = 20
    x = x[::k]
    y = y[::k]
    Nx = x.shape[0]
    Ny = y.shape[0]
    X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')
    pos = np.dstack((X, Y)).reshape((Nx * Ny, 2))
    grad = gradient(pos).reshape((Nx, Ny, 2))
    U = grad[:, :, 0]
    V = grad[:, :, 1]

    #gradient plot
    plt2d = Plot2d(dir_path, 'gradient_vector_field')
    plt2d.set_title(grad_formula)
    plt2d.vector_field(X, Y, U, V)


if __name__ == "__main__":
    main()
