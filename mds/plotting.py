import numpy as np
import matplotlib.pyplot as plt

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

class Plot:
    def __init__(self, file_name, file_type):
        self.file_path = os.path.join(
            FIGURES_PATH,
            file_name + '.' + file_type
        )

    def potential_and_gradient(self, X, V, dV):
        plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(X, dV, 'r-', label=r'Gradient $\nabla V(X)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=1, bottom=-1)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()
    
    def tilted_potential_and_gradient(self, X, V, dV, Vbias, dVbias):
        plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(X, Vbias, 'r-', label=r'bias Potential $V_{bias}(x)$')
        plt.plot(X, V + Vbias, 'm-', label=r'tilted Potential $\tilde{V}(x)$')
        plt.plot(X, dV, 'b-.', label=r'gradient $\nabla V(x)$')
        plt.plot(X, dVbias, 'r-.', label=r'bias gradient $\nabla V_{bias}(x)$')
        plt.plot(X, dV + dVbias, 'm-.', label=r'tilted gradient $\nabla \tilde{V}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=1, bottom=-1)
        plt.legend(loc='lower right', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()
    
    def tilted_potential_and_gradient2(self, X, V, dV, Vbias, dVbias):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        ax1.plot(X, V, 'b-', label=r'$V(x)$')
        ax1.plot(X, Vbias, 'r-', label=r'$V_{bias}(x)$')
        ax1.plot(X, V + Vbias, 'm-', label=r'$\tilde{V}(x)$')
        ax1.set_xlabel('x', fontsize=16)
        ax1.set_ylim(top=1, bottom=0)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True)
        ax2.plot(X, dV, 'b-.', label=r'$\nabla V(x)$')
        ax2.plot(X, dVbias, 'r-.', label=r'$\nabla V_{bias}(x)$')
        ax2.plot(X, dV + dVbias, 'm-.', label=r'$\nabla \tilde{V}(x)$')
        ax2.set_xlabel('x', fontsize=16)
        ax2.set_ylim(top=1, bottom=-1)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)
        fig.savefig(self.file_path)

    def tilted_potential_and_gradient(self, X, V, dV, Vbias, dVbias):
        plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(X, Vbias, 'r-', label=r'bias Potential $V_{bias}(x)$')
        plt.plot(X, V + Vbias, 'm-', label=r'tilted Potential $\tilde{V}(x)$')
        plt.plot(X, dV, 'b-.', label=r'gradient $\nabla V(x)$')
        plt.plot(X, dVbias, 'r-.', label=r'bias gradient $\nabla V_{bias}(x)$')
        plt.plot(X, dV + dVbias, 'm-.', label=r'tilted gradient $\nabla \tilde{V}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=1, bottom=-1)
        plt.legend(loc='lower right', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def tilted_potential(self, X, V, Vbias):
        plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
        plt.plot(X, Vbias, 'r-', label=r'bias Potential $V_{bias}(x)$')
        plt.plot(X, V + Vbias, c='purple', linestyle='-', label=r'tilted Potential $\tilde{V}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=1, bottom=0)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def gradient_tilted_potential(self, X, dV, dVbias):
        plt.plot(X, dV, 'b-', label=r'gradient $\nabla V(x)$')
        plt.plot(X, dVbias, 'r-', label=r'bias gradient $\nabla V_{bias}(x)$')
        plt.plot(X, dV + dVbias, c='purple', linestyle='-', label=r'tilted gradient $\nabla \tilde{V}(x)$')
        plt.xlabel('x', fontsize=16)
        plt.ylim(top=4, bottom=-4)
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
