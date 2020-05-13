import numpy as np
import matplotlib.pyplot as plt

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

class Plot:
    def __init__(self, file_name=None, file_type='png', dir_path=FIGURES_PATH):
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

    def potential_and_gradient(self, X, V, dV):
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
        ax1.set_ylim(top=8, bottom=0)
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
    
    def tilted_potential(self, X, V, Vbias):
        plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(X, Vbias, 'r-', label=r'bias Potential $V_{bias}(x)$')
        plt.plot(X, V + Vbias, c='purple', linestyle='-', label=r'tilted Potential $\tilde{V}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.xlim(left=-1.8, right=1.8)
        plt.ylim(top=8, bottom=0)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_gradient(self, X, dV, dVbias):
        plt.plot(X, dV, 'b-', label=r'gradient $\nabla V(x)$')
        plt.plot(X, dVbias, 'r-', label=r'bias gradient $\nabla V_{bias}(x)$')
        plt.plot(X, dV + dVbias, c='purple', linestyle='-', label=r'tilted gradient $\nabla \tilde{V}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.xlim(left=-1.8, right=1.8)
        plt.ylim(top=12, bottom=-12)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
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

    def performance_function(self, X, J):
        plt.plot(X, J, 'b-', label='J(x)')
        plt.xlim(left=-1.8, right=1.8)
        plt.ylim(top=3, bottom=0)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

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
