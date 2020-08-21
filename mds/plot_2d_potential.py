import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def get_parser():
    parser = argparse.ArgumentParser(description='3D Plot of the potential landscape')
    parser.add_argument(
        '--alphas',
        dest='alphas',
        nargs=2,
        type=float,
        default=[1, 1],
        help='Set the parameter alpha for the chosen potentials. Default: [1, 1]',
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
    alphas = np.array(args.alphas)

    xmin, xmax = (-2, 2)
    ymin, ymax = (-2, 2)
    h = 0.01
    x = np.arange(xmin, xmax, h)
    y = np.arange(ymin, ymax, h)

    zlim_bottom, zlim_top = (0, 10 * alphas.max())

    # surface plot
    if args.plot_surface:
        # create potential
        xx, yy = np.meshgrid(x, y, sparse=True)
        V = alphas[0] * (xx**2 - 1)**2 + alphas[1] * (yy**2 - 1)**2

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
        V = alphas[0] * (xx**2 - 1)**2 + alphas[1] * (yy**2 - 1)**2

        # clip data outside zlims
        zlim_bottom, zlim_top = (0, 10)
        idx_x, idx_y = np.where((V < zlim_bottom) | (V > zlim_top))
        V[idx_x, idx_y] = np.nan

        fig, ax = plt.subplots()
        cs = ax.contourf(
            xx,
            yy,
            V,
            vmin=0,
            vmax=10,
            #locator = ticker.LogLocator(10),
            cmap=cm.coolwarm,
        )
        cbar = fig.colorbar(cs)
        plt.show()


if __name__ == "__main__":
    main()
