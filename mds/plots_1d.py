from mds.utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt

import os

class Plot1d:
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
        self.xmax= None
        self.ymin = None
        self.ymax= None
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

    def set_xlim(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def set_ylim(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax
        tick_sep = (ymax - ymin) / 10
        self.yticks = np.arange(ymin, ymax + tick_sep, tick_sep)

    def potential_or_gradient(self, x, V, label=None):
        plt.title(self.title)
        if label is not None:
            plt.plot(x, V, 'b-', label=label)
        else:
            plt.plot(x, V, 'b-')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.grid(True)
        plt.legend()
        plt.savefig(self.file_path)
        plt.close()

    def multiple_plots(self, x, ys, labels=None):
        assert x.ndim == 1, ''
        assert ys.ndim == 2, ''
        assert x.shape[0] == ys.shape[1], ''

        num_plots = ys.shape[0]
        if labels:
            assert num_plots == len(labels), ''

        plt.title(self.title)
        for i in range(num_plots):
            if labels:
                plt.plot(x, ys[i], '-', label=labels[i])
            else:
                plt.plot(x, ys[i], '-')

        plt.xlabel('x', fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def ansatz_value_f(self, x, v):
        assert v.ndim == 2, ''
        m = v.shape[1]
        for j in range(m):
            plt.plot(x, v[:, j])
        plt.title(self.title)
        plt.xlabel('x', fontsize=16)
        plt.savefig(self.file_path)
        plt.close()

    def ansatz_control(self, x, b):
        assert b.ndim == 2, ''
        m = b.shape[1]
        for j in range(m):
            plt.plot(x, b[:, j])
        plt.title(self.title)
        plt.xlabel('x', fontsize=16)
        plt.savefig(self.file_path)
        plt.close()

    def exp_fht(self, x, exp_fht=None, appr_exp_fht=None):
        plt.title(self.title)
        if exp_fht is not None:
            plt.plot(x, exp_fht, 'c-', label=r'$E^x[\tau]$')
        if appr_exp_fht is not None:
            #todo
            pass
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def psi(self, x, Psi=None, appr_Psi=None):
        plt.title(self.title)
        if Psi is not None:
            plt.plot(x, Psi, 'c-', label=r'$\Psi(x)$')
        if appr_Psi is not None:
            plt.plot(x, appr_Psi, 'm-', label=r'$\tilde{\Psi}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def free_energy(self, x, F=None, appr_F=None):
        plt.title(self.title)
        if F is not None:
            plt.plot(x, F, 'c-', label='F(x)')
        if appr_F is not None:
            plt.plot(x, appr_F, 'm-', label=r'$\tilde{F}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def control(self, x, u_opt=None, u=None):
        plt.title(self.title)
        if u_opt is not None:
            plt.plot(x, u_opt, 'c-', label=r'$u^*(x)$')
        if u is not None:
            plt.plot(x, u, 'm-', label=r'u(x)')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_potential(self, x, V, Vbias=None, Vbias_opt=None):
        plt.title(self.title)
        if Vbias is not None:
            plt.plot(x, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
        if Vbias_opt is not None:
            plt.plot(x, V + Vbias_opt, 'c-', label=r'$\tilde{V}^*(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def potential_and_tilted_potential(self, x, V, Vbias=None, Vbias_opt=None):
        plt.title(self.title)
        plt.plot(x, V, 'b-', label=r'$V(x)$')
        if Vbias is not None:
            plt.plot(x, Vbias, 'r-', label=r'$V_{b}(x)$')
            plt.plot(x, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
        if Vbias_opt is not None:
            plt.plot(x, Vbias_opt, 'y-', label=r'$V_{b}^*(x)$')
            plt.plot(x, V + Vbias_opt, 'c-', label=r'$\tilde{V}^*(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def drift_and_tilted_drift(self, x, dV, dVbias=None, dVbias_opt=None):
        plt.title(self.title)
        plt.plot(x, -dV, 'b-', label=r'$ - \nabla V(x)$')
        if dVbias is not None:
            plt.plot(x, -dVbias, 'r-', label=r'$ - \nabla V_{b}(x)$')
            plt.plot(x, -dV -dVbias, 'm-', label=r'$ - \nabla \tilde{V}(x)$')
        if dVbias_opt is not None:
            plt.plot(x, -dVbias_opt, 'y-', label=r'$ - \nabla V_{b}^*(x)$')
            plt.plot(x, -dV -dVbias_opt, 'c-', label=r'$ - \nabla \tilde{V}^*(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
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
        #plt.ylim(self.ymin, self.ymax)
        plt.ylim(8, -8)
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
        #plt.ylim(self.ymin, self.ymax)
        #ax1.set_ylim(top=8, bottom=-4)
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

    def gd_appr_free_energies(self, x, epochs, F_appr, F=None):
        plt.title('Free energies / Performance functions')
        for i, epoch in enumerate(epochs):
            label = r'epoch = {:d}'.format(epoch)
            plt.plot(x, F_appr[i], linestyle='-', label=label)
        if F is not None:
            plt.plot(x, F, color='c', linestyle='dashed', label=r'$F(x)$')

        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def gd_controls(self, x, epochs, u, u_opt=None):
        plt.title('Controls')
        for i, epoch in enumerate(epochs):
            label = r'epoch = {:d}'.format(epoch)
            plt.plot(x, u[i], linestyle='-', label=label)
        if u_opt is not None:
            plt.plot(x, u_opt, color='c', linestyle='dashed', label=r'$u^*(x)$')

        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def gd_tilted_potentials(self, x, V, epochs, Vbias, Vbias_opt=None):
        plt.title('Tilted potentials')
        for i, epoch in enumerate(epochs):
            label = r'epoch = {:d}'.format(epoch)
            plt.plot(x, V + Vbias[i], linestyle='-', label=label)
        if Vbias_opt is not None:
            plt.plot(x, V + Vbias_opt, color='c', linestyle='dashed', label=r'$\tilde{V}^*(x)$')

        plt.xlabel('x', fontsize=16)
        plt.ylim(self.ymin, self.ymax)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def gd_losses_bar(self, epochs, losses, value_f_ref=None, value_f_mc=None):
        plt.title(self.title)
        plt.bar(
            x=epochs,
            height=losses,
            width=0.8,
            color='b',
            align='center',
            label=r'$J(x_0)$',
        )
        if value_f_ref is not None:
            plt.plot(
                [epochs[0]-0.4, epochs[-1] + 0.4],
                [value_f_ref, value_f_ref],
                color='cyan',
                linestyle='dotted',
                label=r'num sol HJB pde',
            )
        if value_f_mc is not None:
            plt.plot(
                [epochs[0]-0.4, epochs[-1] + 0.4],
                [value_f_mc, value_f_mc],
                color='orange',
                linestyle='dashed',
                label=r'MC Sampling',
            )
        plt.xlabel('epochs', fontsize=16)
        plt.xlim(left=epochs[0] -0.8, right=epochs[-1] + 0.8)
        plt.ylim(self.ymin, self.ymax)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def gd_losses_line(self, epochs, losses, value_f_ref=None, value_f_mc=None):
        plt.title(self.title)
        plt.plot(epochs, losses, 'b-', label=r'$J(x_0)$')
        if value_f_ref is not None:
            plt.plot([epochs[0], epochs[-1]], [value_f_ref, value_f_ref],
                     color='cyan', linestyle='dotted', label=r'num sol HJB pde')
        if value_f_mc is not None:
            plt.plot([epochs[0], epochs[-1]], [value_f_mc, value_f_mc],
                     color='orange', linestyle='dashed', label=r'MC Sampling')
        plt.xlabel('epochs', fontsize=16)
        #plt.xlim(left=epochs[0], right=epochs[-1])
        #plt.xticks(epochs)
        plt.ylim(self.ymin, self.ymax)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def gd_time_steps_bar(self, epochs, N):
        plt.title(self.title)
        plt.bar(x=epochs, height=N, width=0.8, color='b', align='center', label=r'$N$')
        plt.xlabel('epochs', fontsize=16)
        plt.xlim(left=epochs[0] -0.8, right=epochs[-1] + 0.8)
        plt.ylim(self.ymin, self.ymax)
        #plt.set_yticks([-1,0,1])
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def gd_time_steps_line(self, epochs, N):
        plt.title(self.title)
        plt.plot(epochs, N, 'b-', label=r'$N$')
        plt.xlabel('epochs', fontsize=16)
        #plt.xlim(left=epochs[0], right=epochs[-1])
        #plt.xticks(epochs)
        plt.ylim(self.ymin, self.ymax)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def trajectory(self, x, y):
        plt.title(self.title)
        plt.plot(x, y, 'r', label='EM solution path')
        plt.xlabel('t', fontsize=16)
        plt.ylabel('X', fontsize=16)
        plt.ylim(-1.8, 1.8)
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
