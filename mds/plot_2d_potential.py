from potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES
from utils import get_example_data_path

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

    potential, _ = get_potential_and_gradient(potential_name, alpha)

    dir_path = get_example_data_path(potential_name, alpha)

    xmin, xmax = (-2, 2)
    ymin, ymax = (-2, 2)
    h = 0.01
    x = np.arange(xmin, xmax, h)
    y = np.arange(ymin, ymax, h)

    zlim_bottom, zlim_top = (0, 10 * alpha.max())

    # surface plot
    # create potential
    xx, yy = np.meshgrid(x, y, sparse=True, indexing='ij')
    V = potential(xx, yy)

    # clip data outside zlims
    idx_x, idx_y = np.where((V < zlim_bottom) | (V > zlim_top))
    V[idx_x, idx_y] = np.nan

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xx,
        yy,
        V,
        cmap=cm.coolwarm,
        vmin=zlim_top,
        vmax=zlim_bottom,
        linewidth=0,
        antialiased=False,
    )
    ax.set_zlim(zlim_bottom, zlim_top)
    fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    file_path = os.path.join(dir_path, 'potential_surface.png')
    plt.savefig(file_path)

    # contour plot
    # create potential
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')
    V = potential(xx, yy)

    # clip data outside zlims
    zlim_bottom, zlim_top = (0, 10)
    idx_x, idx_y = np.where((V < zlim_bottom) | (V > zlim_top))
    V[idx_x, idx_y] = np.nan

    fig, ax = plt.subplots()
    levels = np.logspace(-2, 1, 20, endpoint=True)
    cs = ax.contourf(
        xx,
        yy,
        V,
        vmin=0,
        vmax=10,
        levels=levels,
        cmap=cm.coolwarm,
    )
    cbar = fig.colorbar(cs)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    file_path = os.path.join(dir_path, 'potential_contour.png')
    plt.savefig(file_path)


if __name__ == "__main__":
    main()
