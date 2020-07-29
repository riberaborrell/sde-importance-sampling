from utils import get_data_path

import numpy as np
import matplotlib.pyplot as plt

import os

class Plot:
    dir_path = get_data_path()
    def __init__(self, dir_path=dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

    @property
    def file_path(self):
        if self.file_name:
            return os.path.join(
                self.dir_path, self.file_name + '.' + self.file_type
            )
        else:
            return None

    def set_ylim(self, bottom=None, top=None):
        self.bottom = bottom
        self.top = top
        tick_sep = (top - bottom) / 10
        self.yticks = np.arange(bottom, top + tick_sep, tick_sep)

    def potential(self, X, V):
        plt.title('Potential')
        plt.plot(X, V, 'b-', label=r'$V(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=15, bottom=0)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def potential_and_tilted_potential(self, X, V, Vbias, Vopt=None):
        plt.title('Potential, bias potential and tilted potential')
        plt.plot(X, V, 'b-', label=r'$V(x)$')
        plt.plot(X, Vbias, 'r-', label=r'$V_{b}(x)$')
        plt.plot(X, V + Vbias, c='purple', linestyle='-', label=r'$\tilde{V}(x)$')
        if Vopt is not None:
            plt.plot(X, Vopt, 'c-', label=r'$\tilde{V}_{opt}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_potential_wrt_betas(self, X, V, betas, Vbias):
        assert betas.shape[0] == Vbias.shape[0], ''
        plt.title(r'Tilted potential $\tilde{V}(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(X, V + Vbias[i, :], '-', label=label)
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def drift_and_tilted_drift(self, X, dV, dVbias, dVopt=None):
        plt.title('Drift, bias drift and tilted drift')
        plt.plot(X, -dV, 'b-', label=r'$ - \nabla V(x)$')
        plt.plot(X, -dVbias, 'r-', label=r'$ - \nabla V_{bias}(x)$')
        plt.plot(X, -dV -dVbias, c='purple', linestyle='-', label=r'$ - \nabla \tilde{V}(x)$')
        if dVopt is not None:
            plt.plot(X, -dVopt, 'c-', label=r'$ - \nabla \tilde{V}_{opt}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_drift_wrt_betas(self, X, dV, betas, dVbias):
        assert betas.shape[0] == dVbias.shape[0], ''
        plt.title(r'Tilted drift $ - \nabla \tilde{V}(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(X, -dV - dVbias[i, :], '-', label=label)
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()


    def free_energy(self, X, J):
        plt.title('Free energy / Performance function')
        plt.plot(X, J, 'b-', label='F(x)')
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def free_energy_wrt_betas(self, X, betas, J):
        assert betas.shape[0] == J.shape[0], ''
        plt.title('Free energy / Performance function $F(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(X, J[i, :], '-', label=label)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def control(self, X, u):
        plt.title('Control')
        plt.plot(X, u, 'b-', label='u(x)')
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def control_wrt_betas(self, X, betas, u):
        assert betas.shape[0] == u.shape[0], ''
        plt.title(r'Control $u(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(X, u[i, :], '-', label=label)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def potential_and_gradient(self, X, V, dV):
        plt.title('Potential and Gradient')
        plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(X, dV, 'r-', label=r'Gradient $\nabla V(X)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=8, bottom=-8)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_potential_and_gradient(self, X, V, dV, Vbias, dVbias, Vopt=None, dVopt=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        ax1.plot(X, V, 'b-', label=r'$V(x)$')
        ax1.plot(X, Vbias, 'r-', label=r'$V_{bias}(x)$')
        ax1.plot(X, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
        if Vopt is not None:
            ax1.plot(X, Vopt, 'c-', label=r'$\tilde{V}_{opt}(x)$')
        ax1.set_xlabel('x', fontsize=16)
        ax1.set_xlim(left=-1.8, right=1.8)
        ax1.set_ylim(top=8, bottom=-4)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True)
        ax2.plot(X, dV, 'b-.', label=r'$\nabla V(x)$')
        ax2.plot(X, dVbias, 'r-.', label=r'$\nabla V_{bias}(x)$')
        ax2.plot(X, dV + dVbias, 'm-.', label=r'$\nabla \tilde{V}(x)$')
        if dVopt is not None:
            ax2.plot(X, dVopt, 'c-', label=r'$\nabla \tilde{V}_{opt}(x)$')
        ax2.set_xlabel('x', fontsize=16)
        ax2.set_xlim(left=-1.8, right=1.8)
        ax2.set_ylim(top=12, bottom=-12)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)
        fig.savefig(self.file_path)
        plt.close()

    def gradient_descent_tilted_potentials(self, X, Vs, Vopt):
        for i, V in enumerate(Vs):
            label = r'epoch = {:d}'.format(i)
            plt.plot(X, V, linestyle='-', label=label)
        plt.plot(X, Vopt, linestyle='dashed', label='optimal')

        plt.xlabel('x', fontsize=16)
        plt.xlim(left=-1.8, right=1.8)
        plt.ylim(top=8, bottom=0)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def ansatz_functions(self, X, v):
        m = v.shape[0]
        for j in range(m):
            plt.plot(X, v[j])
        plt.title(r'$v_{j}(x; \mu, \sigma)$')
        plt.xlabel('x', fontsize=16)
        plt.savefig(self.file_path)
        plt.close()

    def control_basis_functions(self, X, b):
        m = b.shape[0]
        for j in range(m):
            plt.plot(X, b[j])
        plt.title(r'$b_{j}(x; \mu, \sigma)$')
        plt.xlabel('x', fontsize=16)
        plt.savefig(self.file_path)
        plt.close()

    #def ansatz_functions(self, X, ans, weig_ans, V, Vbias):
    #plt.plot(X, weig_ans[j], 'g-', label=r'$a_j v_j(x)$')
    #plt.plot(X, V, 'b-', label=r'$V(x)$')
    #plt.plot(X, Vbias, 'r-', label=r'$V_{bias}(x)$')
    #plt.plot(X, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
    #plt.ylim(top=4, bottom=0)

    #TODO plot potential and gradient in 1D.

    #TODO plot tilted potential and basis functions


    def trajectory(self, x, y):
        plt.plot(x, y, 'r', label='EM solution path')
        plt.xlabel('t', fontsize=16)
        plt.ylabel('X', fontsize=16)
        #plt.xlim(right=600)
        plt.ylim(bottom=-1.8)
        plt.ylim(top=1.8)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()
