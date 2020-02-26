import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rc

#rc('text', usetex=True)

def plot_potential_and_gradient(X, V, dV):
    plt.plot(X, V, 'b-', label=r'Potential $V(x)$')
    plt.plot(X, dV, 'r-', label=r'Gradient $\nabla V(X)$')
    plt.xlabel('x', fontsize=16)
    plt.ylim(top=1, bottom=-1)
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

#TODO plot potential and gradient in 1D.

#TODO plot tilted potential and basis functions


def plot_trajectory(x, y):
    plt.plot(x, y, 'r', label='EM solution path')
    plt.xlabel('t', fontsize=16)
    plt.ylabel('X', fontsize=16)
    plt.xlim(right=60)
    plt.ylim(bottom=-1.8)
    plt.ylim(top=1.8)
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
