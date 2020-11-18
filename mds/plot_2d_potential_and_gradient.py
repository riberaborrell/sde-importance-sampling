from mds.base_parser_2d import get_base_parser
from mds.plots_2d import Plot2d
from mds.potentials_and_gradients_2d import get_potential_and_gradient
from mds.utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Plots the potential landscape surface and contour and' \
                         'the gradient vector field'
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
    xmin, xmax = (-3, 3)
    ymin, ymax = (-3, 3)
    h = 0.01
    x = np.around(np.arange(xmin, xmax + h, h), decimals=3)
    y = np.around(np.arange(ymin, ymax + h, h), decimals=3)
    Nx = x.shape[0]
    Ny = y.shape[0]
    xx, yy = np.meshgrid(x, y, sparse=True, indexing='ij')
    X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')

    # set bounds for the z axis
    zmin, zmax = (0, 10 * alpha.max())

    # evaluate potential and gradient
    pos = np.dstack((X, Y)).reshape(Nx * Ny, 2)
    pot = potential(pos).reshape(Nx, Ny)
    grad = gradient(pos).reshape(Nx, Ny, 2)
    U = -grad[:, :, 0]
    V = -grad[:, :, 1]

    # potential surface plot
    plt2d = Plot2d(dir_path, 'potential_surface')
    #plt2d.set_title(pot_formula)
    plt2d.set_xlim(-2, 2)
    plt2d.set_ylim(-2, 2)
    plt2d.set_zlim(zmin, zmax)
    plt2d.surface(xx, yy, pot)

    # potential contour plot
    levels = np.logspace(-2, 1, 20, endpoint=True)
    plt2d = Plot2d(dir_path, 'potential_contour')
    #plt2d.set_title(pot_formula)
    plt2d.set_xlim(-2, 2)
    plt2d.set_ylim(-2, 2)
    plt2d.set_zlim(zmin, zmax)
    plt2d.contour(X, Y, pot, levels)

    # minus gradient vector field plot
    plt2d = Plot2d(dir_path, 'gradient_vector_field')
    #plt2d.set_title(grad_formula)
    plt2d.set_xlim(-2, 2)
    plt2d.set_ylim(-2, 2)
    plt2d.vector_field(X, Y, U, V, scale=100)

    # zoomed minus gradient vector field plot
    plt2d = Plot2d(dir_path, 'gradient_vector_field_zoomed')
    #plt2d.set_title(grad_formula)
    plt2d.set_xlim(-1.25, 1.25)
    plt2d.set_ylim(-1.25, 1.25)
    plt2d.vector_field(X, Y, U, V, scale=25)


if __name__ == "__main__":
    main()
