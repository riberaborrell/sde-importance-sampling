from potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def get_parser():
    parser = argparse.ArgumentParser(description='3D Plot of the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='2d_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1, 1, 1, 1],
        help='Set the parameter alpha for the chosen potentials. Default: [1, 1, 1, 1]',
    )
    parser.add_argument(
        '--plot-surface',
        dest='plot_surface',
        action='store_true',
        help='Do surface plot. Default: False',
    )
    parser.add_argument(
        '--plot-contour',
        dest='plot_contour',
        action='store_true',
        help='Do contour plot. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()
    potential_name = args.potential_name
    alpha = np.array(args.alpha)

    potential, _ = get_potential_and_gradient(potential_name, alpha)

    xmin, xmax = (-2, 2)
    ymin, ymax = (-2, 2)
    h = 0.01
    x = np.arange(xmin, xmax, h)
    y = np.arange(ymin, ymax, h)

    zlim_bottom, zlim_top = (0, 10 * alpha.max())

    # surface plot
    if args.plot_surface:
        # create potential
        xx, yy = np.meshgrid(x, y, sparse=True)
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
        plt.show()

    # contour plot
    if args.plot_contour:
        # create potential
        xx, yy = np.meshgrid(x, y)
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
        plt.show()


if __name__ == "__main__":
    main()
