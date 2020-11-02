from mds.utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os

class Plot2d:
    dir_path = get_example_data_path()
    def __init__(self, dir_path=dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path
        self.title = None
        self.label = ''

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

    def surface(self, X, Y, Z, vmin=None, vmax=None):
        assert X.ndim == Y.ndim == Z.ndim == 2, ''
        assert Z.shape == (X.shape[0], Y.shape[1]), ''

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # clip data outside vmin and vmax
        if vmin is not None and vmax is not None:
            idx_x, idx_y = np.where((Z < vmin) | (Z > vmax))
            Z[idx_x, idx_y] = np.nan
        elif vmin is not None:
            idx_x, idx_y = np.where(Z < vmin)
            Z[idx_x, idx_y] = np.nan
        elif vmax is not None:
            idx_x, idx_y = np.where(Z > vmax)
            Z[idx_x, idx_y] = np.nan

        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            vmin=vmin,
            vmax=vmax,
            antialiased=False,
        )
        if self.title is not None:
            ax.set_title(self.title)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlim(vmin, vmax)
        fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
        plt.savefig(self.file_path)

    def contour(self, X, Y, Z, vmin=None, vmax=None, levels=None):
        assert X.ndim == Y.ndim == Z.ndim == 2, ''
        assert Z.shape == X.shape == Y.shape, ''

        fig, ax = plt.subplots()
        cs = ax.contourf(
            X,
            Y,
            Z,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cm.coolwarm,
        )
        if self.title is not None:
            ax.set_title(self.title)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        cbar = fig.colorbar(cs)
        plt.savefig(self.file_path)

    def vector_field(self, X, Y, U, V):
        fig, ax = plt.subplots()
        C = np.sqrt(U**2 + V**2)
        quiv = ax.quiver(X, Y, U, V, C, angles='xy', scale_units='xy')
        if self.title is not None:
            ax.set_title(self.title)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        #fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
        plt.savefig(self.file_path)
