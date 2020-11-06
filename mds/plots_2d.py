from mds.utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import os

class Plot2d:
    dir_path = get_example_data_path()
    def __init__(self, dir_path=dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

        # title and label
        self.title = ''
        self.label = ''

        # axes bounds
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None

    @property
    def file_path(self):
        if self.file_name:
            return os.path.join(
                self.dir_path, self.file_name + '.' + self.file_type
            )
        else:
            return None

    def set_title(self, title):
        self.title = title

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
        plt.savefig(self.file_path)

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

    def coarse_quiver_arrows(self, X, Y, U, V, k):
        # show every k row and column
        X = X[::k, ::k]
        Y = Y[::k, ::k]
        U = U[::k, ::k]
        V = V[::k, ::k]
        return X, Y, U, V

    def vector_field(self, X, Y, U, V, k=None, scale=None):
        fig, ax = plt.subplots()
        if self.xmin and self.xmax and self.ymin and self.ymax:
            X, Y, U, V = self.reduce_quiver_arrows(X, Y, U, V)
        if k is not None:
            X, Y, U, V = self.coarse_quiver_arrows(X, Y, U, V, k)

        C = np.sqrt(U**2 + V**2)

        # initialize norm object and make rgba array
        norm = colors.Normalize(vmin=np.min(C), vmax=np.max(C))
        sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)

        quiv = ax.quiver(
            X,
            Y,
            U,
            V,
            C,
            cmap=cm.coolwarm,
            angles='xy',
            scale_units='xy',
            scale=scale,
        )

        ax.set_title(self.title)
        ax.set_xlabel(r'$x_1$', fontsize=16)
        ax.set_ylabel(r'$x_2$', fontsize=16)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        plt.colorbar(sm)
        plt.savefig(self.file_path)
