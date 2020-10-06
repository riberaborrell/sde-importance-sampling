from utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt

import os

class Plot:
    dir_path = get_example_data_path()
    def __init__(self, dir_path=dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

        self.title = ''
        self.bottom = None
        self.top = None
        self.yticks = None

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

    def set_ylim(self, bottom=None, top=None):
        self.bottom = bottom
        self.top = top
        tick_sep = (top - bottom) / 10
        self.yticks = np.arange(bottom, top + tick_sep, tick_sep)

    def potential(self, x, V, label=None):
        plt.title(self.title)
        if label is not None:
            plt.plot(x, V, 'b-', label=label)
        else:
            plt.plot(x, V, 'b-')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.grid(True)
        plt.legend()
        plt.savefig(self.file_path)
        plt.close()

    def potentials(self, x, Vs, labels=None):
        if labels is not None:
            assert Vs.shape[0] == len(labels), ''
        plt.title(self.title)
        for i in range(Vs.shape[0]):
            if labels is not None:
                plt.plot(x, Vs[i], '-', label=labels[i])
            else:
                plt.plot(x, Vs[i], '-')

        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)

    def exp_fht(self, x, exp_fht=None, appr_exp_fht=None):
        plt.title('Expected first hitting time')
        if exp_fht is not None:
            plt.plot(x, exp_fht, 'c-', label=r'$E^x[\tau]$')
        if appr_exp_fht is not None:
            #todo
            pass
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def mgf(self, x, Psi=None, appr_Psi=None):
        plt.title('Moment generating function')
        if Psi is not None:
            plt.plot(x, Psi, 'c-', label=r'$\Psi(x)$')
        if appr_Psi is not None:
            plt.plot(x, appr_Psi, 'm-', label=r'$\tilde{\Psi}(x)$')
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def free_energy(self, x, F=None, appr_F=None):
        plt.title('Free energy / Performance function')
        if F is not None:
            plt.plot(x, F, 'c-', label='F(x)')
        if appr_F is not None:
            plt.plot(x, appr_F, 'm-', label=r'$\tilde{F}(x)$')
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def control(self, x, u_opt=None, u=None):
        plt.title('Control')
        if u_opt is not None:
            plt.plot(x, u_opt, 'c-', label=r'$u^*(x)$')
        if u is not None:
            plt.plot(x, u, 'm-', label=r'u(x)')
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def potential_and_tilted_potential(self, x, V, Vbias=None, Vbias_opt=None):
        plt.title('Potential, bias potential and tilted potential')
        plt.plot(x, V, 'b-', label=r'$V(x)$')
        if Vbias is not None:
            plt.plot(x, Vbias, 'r-', label=r'$V_{b}(x)$')
            plt.plot(x, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
        if Vbias_opt is not None:
            plt.plot(x, Vbias_opt, 'y-', label=r'$V_{b}^*(x)$')
            plt.plot(x, V + Vbias_opt, 'c-', label=r'$\tilde{V}^*(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def drift_and_tilted_drift(self, x, dV, dVbias=None, dVbias_opt=None):
        plt.title('Drift, bias drift and tilted drift')
        plt.plot(x, -dV, 'b-', label=r'$ - \nabla V(x)$')
        if dVbias is not None:
            plt.plot(x, -dVbias, 'r-', label=r'$ - \nabla V_{b}(x)$')
            plt.plot(x, -dV -dVbias, 'm-', label=r'$ - \nabla \tilde{V}(x)$')
        if dVbias_opt is not None:
            plt.plot(x, -dVbias_opt, 'y-', label=r'$ - \nabla V_{b}^*(x)$')
            plt.plot(x, -dV -dVbias_opt, 'c-', label=r'$ - \nabla \tilde{V}^*(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_potential_wrt_betas(self, x, V, betas, Vbias):
        assert betas.shape[0] == Vbias.shape[0], ''
        plt.title(r'Tilted potential $\tilde{V}(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(x, V + Vbias[i, :], '-', label=label)
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_drift_wrt_betas(self, x, dV, betas, dVbias):
        assert betas.shape[0] == dVbias.shape[0], ''
        plt.title(r'Tilted drift $ - \nabla \tilde{V}(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(x, -dV - dVbias[i, :], '-', label=label)
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def free_energy_wrt_betas(self, x, betas, F):
        assert betas.shape[0] == F.shape[0], ''
        plt.title('Free energy / Performance function $F(x)$')
        for i, beta in enumerate(betas):
            label = r'$\beta={:2.1f}$'.format(beta)
            plt.plot(x, F[i, :], '-', label=label)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def control_wrt_betas(self, x, betas, u):
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

    def potential_and_gradient(self, x, V, dV):
        plt.title('Potential and Gradient')
        plt.plot(x, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(x, dV, 'r-', label=r'Gradient $\nabla V(X)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=8, bottom=-8)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_potential_and_gradient(self, x, V, dV, Vbias, dVbias, Vopt=None, dVopt=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        ax1.plot(x, V, 'b-', label=r'$V(x)$')
        ax1.plot(x, Vbias, 'r-', label=r'$V_{bias}(x)$')
        ax1.plot(x, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
        if Vopt is not None:
            ax1.plot(x, Vopt, 'c-', label=r'$\tilde{V}_{opt}(x)$')
        ax1.set_xlabel('x', fontsize=16)
        ax1.set_xlim(left=-1.8, right=1.8)
        ax1.set_ylim(top=8, bottom=-4)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True)
        ax2.plot(x, dV, 'b-.', label=r'$\nabla V(x)$')
        ax2.plot(x, dVbias, 'r-.', label=r'$\nabla V_{bias}(x)$')
        ax2.plot(x, dV + dVbias, 'm-.', label=r'$\nabla \tilde{V}(x)$')
        if dVopt is not None:
            ax2.plot(x, dVopt, 'c-', label=r'$\nabla \tilde{V}_{opt}(x)$')
        ax2.set_xlabel('x', fontsize=16)
        ax2.set_xlim(left=-1.8, right=1.8)
        ax2.set_ylim(top=12, bottom=-12)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)
        fig.savefig(self.file_path)
        plt.close()

    def gradient_descent_tilted_potentials(self, x, Vs, Vopt):
        for i, V in enumerate(Vs):
            label = r'epoch = {:d}'.format(i)
            plt.plot(x, V, linestyle='-', label=label)
        plt.plot(x, Vopt, linestyle='dashed', label='optimal')

        plt.xlabel('x', fontsize=16)
        plt.xlim(left=-1.8, right=1.8)
        plt.ylim(top=8, bottom=0)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def ansatz_value_f(self, x, v):
        m = v.shape[0]
        for j in range(m):
            plt.plot(x, v[j])
        plt.title(self.title)
        plt.xlabel('x', fontsize=16)
        plt.savefig(self.file_path)
        plt.close()

    def ansatz_control(self, x, b):
        m = b.shape[0]
        for j in range(m):
            plt.plot(x, b[j])
        plt.title(self.title)
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

    def potential_2d(self, X, V):
        assert X.ndim == V.ndim == 2, ''
        plt.title(self.title)
        plt.plot(X, V, 'b-', label=r'$V(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=self.top, bottom=self.bottom)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()
