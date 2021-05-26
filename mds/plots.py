import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import os

LEGEND_LOCATION_STRINGS = [
    'best',
    'upper right',
    'upper left',
    'lower left',
    'lower right',
    'right',
    'center left',
    'center right',
    'lower center',
    'upper center',
    'center',
]

class Plot:
    def __init__(self, dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

        # initialize matplotlib object
        self.plt = plt

        # title and label
        self.title = ''
        self.label = ''

        # axes labels
        self.xlabel = None
        self.ylabel = None
        self.zlabel = None

        # axes bounds
        self.xmin = None
        self.xmax= None
        self.ymin = None
        self.ymax= None
        self.zmin = None
        self.zmax = None

        # axes ticks
        self.yticks = None

        # logplot
        self.logplot = False

        # legend
        self.legend_loc = 'best'

        # transparency
        self.alpha = 1

    @property
    def file_path(self):
        if self.file_name:
            return os.path.join(
                self.dir_path, self.file_name + '.' + self.file_type
            )
        else:
            return None

    def set_title(self, title):
        self.plt.title(title)

    def set_label(self, label):
        self.label = label

    def set_xlim(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def set_ylim(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax

    def set_zlim(self, zmin, zmax):
        self.zmin = zmin
        self.zmax = zmax

    def set_scientific_notation(self, axis='both'):
        assert axis in ['x', 'y', 'both']

        self.plt.ticklabel_format(axis=axis, style='sci', scilimits=(-1, 1))

    def set_logplot(self):
        self.logplot = True

    def set_legend_location(self, loc):
        assert loc in LEGEND_LOCATION_STRINGS, ''

        self.legend_loc = loc

    def set_line_transparency(self, alpha):
        self.alpha = alpha

    def one_line_plot(self, x, y, color=None, label=None):
        assert x.ndim == y.ndim == 1, ''
        assert x.shape[0] == y.shape[0], ''

        plt = self.plt

        if not self.logplot:
            plt.plot(
                x,
                y,
                color=color,
                linestyle='-',
                label=label,
                alpha=self.alpha,
            )
        else:
            plt.semilogy(
                x,
                y,
                color=color,
                linestyle='-',
                label=label,
                alpha=self.alpha,
            )

        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.grid(True)

        if label is not None:
            plt.legend(loc=self.legend_loc, fontsize=8)

        plt.savefig(self.file_path)
        plt.close()

    def multiple_lines_plot(self, x, ys, colors=None, linestyles=None, labels=None):
        assert x.ndim == 1, ''
        assert ys.ndim == 2, ''
        assert x.shape[0] == ys.shape[1], ''

        num_plots = ys.shape[0]
        if colors is not None:
            assert num_plots == len(colors), ''
        else:
            colors = [None for i in range(num_plots)]
        if linestyles is not None:
            assert num_plots == len(linestyles), ''
        else:
            linestyles = ['-' for i in range(num_plots)]
        if labels is not None:
            assert num_plots == len(labels), ''
        else:
            labels = [None for i in range(num_plots)]

        plt.title(self.title)
        for i in range(num_plots):
            if not self.logplot:
                plt.plot(
                    x,
                    ys[i],
                    color=colors[i],
                    linestyle=linestyles[i],
                    label=labels[i],
                    alpha=self.alpha,
                )
            else:
                plt.semilogy(
                    x,
                    ys[i],
                    color=colors[i],
                    linestyle=linestyles[i],
                    label=labels[i],
                    alpha=self.alpha,
                )


        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.grid(True)
        if any(label is not None for label in labels):
            plt.legend(loc=self.legend_loc, fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def one_bar_plot(self, x, height, color=None, label=None):
        assert x.ndim == height.ndim == 1, ''
        assert x.shape[0] == height.shape[0], ''

        plt.title(self.title)

        plt.bar(
            x,
            height,
            width=0.8,
            color=color,
            label=label,
            align='center',
            alpha=self.alpha,
        )
        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        #plt.xlim(left=epochs[0] -0.8, right=epochs[-1] + 0.8)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        #plt.set_yticks([-1,0,1])
        if label is not None:
            plt.legend(loc=self.legend_loc, fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def surface(self, X, Y, Z):
        assert X.ndim == Y.ndim == Z.ndim == 2, ''
        assert Z.shape == (X.shape[0], Y.shape[1]), ''

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # clip data outside vmin and vmax
        if self.zmin is not None and self.zmax is not None:
            idx_x, idx_y = np.where((Z < self.zmin) | (Z > self.zmax))
            Z[idx_x, idx_y] = np.nan
        elif self.zmin is not None:
            idx_x, idx_y = np.where(Z < self.zmin)
            Z[idx_x, idx_y] = np.nan
        elif self.zmax is not None:
            idx_x, idx_y = np.where(Z > self.zmax)
            Z[idx_x, idx_y] = np.nan

        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            vmin=self.zmin,
            vmax=self.zmax,
            antialiased=False,
        )
        ax.set_title(self.title)
        ax.set_xlabel(r'$x_1$', fontsize=16)
        ax.set_ylabel(r'$x_2$', fontsize=16)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_zlim(self.zmin, self.zmax)

        fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
        #fig.subplots_adjust(wspace=0.2, hspace=0)
        plt.savefig(self.file_path)
        plt.close()

    def contour(self, X, Y, Z, levels=None):
        assert X.ndim == Y.ndim == Z.ndim == 2, ''
        assert Z.shape == X.shape == Y.shape, ''

        fig, ax = plt.subplots()
        cs = ax.contourf(
            X,
            Y,
            Z,
            vmin=self.zmin,
            vmax=self.zmax,
            levels=levels,
            cmap=cm.coolwarm,
        )
        ax.set_title(self.title)
        ax.set_xlabel(r'$x_1$', fontsize=16)
        ax.set_ylabel(r'$x_2$', fontsize=16)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        cbar = fig.colorbar(cs)
        plt.savefig(self.file_path)
        plt.close()

    def reduce_quiver_arrows(self, X, Y, U, V):
        # get the indices of the given limits
        x = X[:, 0]
        y = Y[0, :]
        idx_xmin = np.where(x == self.xmin)[0][0]
        idx_xmax = np.where(x == self.xmax)[0][0]
        idx_ymin = np.where(y == self.ymin)[0][0]
        idx_ymax = np.where(y == self.ymax)[0][0]
        X = X[idx_xmin:idx_xmax+1, idx_ymin:idx_ymax+1]
        Y = Y[idx_xmin:idx_xmax+1, idx_ymin:idx_ymax+1]
        U = U[idx_xmin:idx_xmax+1, idx_ymin:idx_ymax+1]
        V = V[idx_xmin:idx_xmax+1, idx_ymin:idx_ymax+1]
        return X, Y, U, V

    def coarse_quiver_arrows(self, X, Y, U, V, kx, ky):
        # show every k row and column
        X = X[::kx, ::ky]
        Y = Y[::kx, ::ky]
        U = U[::kx, ::ky]
        V = V[::kx, ::ky]
        return X, Y, U, V

    def vector_field(self, X, Y, U, V, kx=None, ky=None, scale=None, width=0.005):
        fig, ax = plt.subplots()
        if (self.xmin is not None and
            self.xmax is not None and
            self.ymin is not None and
            self.ymax is not None):
            X, Y, U, V = self.reduce_quiver_arrows(X, Y, U, V)
        if kx is None:
            kx = X.shape[0] // 25
        if ky is None:
            ky = Y.shape[1] // 25
        X, Y, U, V = self.coarse_quiver_arrows(X, Y, U, V, kx, ky)

        C = np.sqrt(U**2 + V**2)

        # modify color map
        colormap = cm.get_cmap('viridis_r', 100)
        colormap = colors.ListedColormap(
            colormap(np.linspace(0.20, 0.95, 75))
        )

        # initialize norm object and make rgba array
        norm = colors.Normalize(vmin=np.min(C), vmax=np.max(C))
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)

        quiv = ax.quiver(
            X,
            Y,
            U,
            V,
            C,
            cmap=colormap,
            angles='xy',
            scale_units='xy',
            scale=scale,
            width=width,
        )

        ax.set_title(self.title)
        ax.set_xlabel(r'$x_1$', fontsize=16)
        ax.set_ylabel(r'$x_2$', fontsize=16)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        plt.colorbar(sm)
        plt.savefig(self.file_path)
        plt.close()
