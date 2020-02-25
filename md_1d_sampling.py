import argparse
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_potential_and_gradient
from tools import double_well_1d_potential, \
                  gradient_double_well_1d_potential

parser = argparse.ArgumentParser(description='Brownian path simulation')
parser.add_argument(
    '--seed',
    dest='seed',
    type=int,
    help='Set the seed for RandomState',
)
parser.add_argument(
    '--beta',
    dest='beta',
    type=float,
    default=1.,
    help='Set the parameter beta for the 1D MD SDE. Default: 1',
)
parser.add_argument(
    '--xzero',
    dest='xzero',
    type=float,
    default=-1.,
    help='Set the value of the process at time t=0. Default: 1',
)
args = parser.parse_args()

# set random seed
if args.seed:
    np.random.seed(args.seed)

# grid x coordinate
X = np.linspace(-1.5, 1.5, 100)

# potential and gradient
V = double_well_1d_potential(X)
dV = gradient_double_well_1d_potential(X)
#plot_potential_and_gradient(X, V, dV)

# time interval, time steps and number of time steps
tzero = 0
T = 6*10 # 60 seconds
N = 6*10**5 # 60 thousand time steps
dt = (T - tzero) / N # 0,0001

# Euler-Maruyama
R = 4
L = int(N/R)
dt_coarse = R*dt
t_coarse = np.linspace(tzero, T, L+1)

# 1D MD SDE is dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x,
beta = args.beta
Xem = np.zeros(L+1)
Xtemp = args.xzero
Xem[0] = Xtemp

# Brownian increments
dB = np.sqrt(dt)*np.random.normal(0, 1, N)


for j in np.arange(1, L+1):
    # stop simulation if particle entered target set T
    if (Xtemp > 0.9 and Xtemp < 1.1):
        print('The trajectory HAS arrived to the target set!')
        break
    
    # compute the increments of the Brownian motion 
    Binc = np.sum(dB[(j-1)*R+1: j*R])
    
    # compute drift and diffusion coefficients
    drift = - gradient_double_well_1d_potential(Xtemp)*dt
    diffusion = np.sqrt(2/beta) * Binc

    # compute X_t
    Xtemp = Xtemp + drift + diffusion
    Xem[j] = Xtemp

# final time and final number of steps
n = j*R
t = n*dt

if j == L:
    print('The trajectory has NOT arrived to the target set!')
else:
    print('Steps needed: {:8.2f} \n'.format(n))
    print('Time needed: {:8.2f} \n'.format(t))

# plot trajectory
plt.plot(t_coarse, Xem, 'r', label='EM solution path')
plt.xlabel('t', fontsize=16)
plt.ylabel('X', fontsize=16)
plt.legend(loc='upper left', fontsize=8)
plt.show()
