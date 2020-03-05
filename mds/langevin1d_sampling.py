import argparse
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_potential_and_gradient, \
                     plot_trajectory
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
    default=10,
    help='Set the parameter beta for the 1D MD SDE. Default: 1',
)
parser.add_argument(
    '--xzero',
    dest='xzero',
    type=float,
    default=-1.,
    help='Set the value of the process at time t=0. Default: 1',
)
parser.add_argument(
    '--target-set',
    nargs=2,
    dest='target_set',
    type=float,
    default=[0.9, 1.1],
    help='Set the target set interval. Default: [0.9, 1.1]',
)
args = parser.parse_args()


# set random seed
if args.seed:
    np.random.seed(args.seed)

# check target set
target_set_min = args.target_set[0]
target_set_max = args.target_set[1]
if target_set_min >= target_set_max:
    #TODO raise error
    print("The target set interval is not valid")
    exit() 

# grid x coordinate
X = np.linspace(-2, 2, 100)

# potential and gradient
V = double_well_1d_potential(X)
dV = gradient_double_well_1d_potential(X)
plot_potential_and_gradient(X, V, dV)
exit()

# time interval, time steps and number of time steps
tzero = 0
T = 6*10 # 60 seconds
N = 6*10**5 # 60 thousand time steps
dt = (T - tzero) / N # 0,0001

# Euler-Maruyama
R = 1
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
    if (Xtemp > target_set_min and Xtemp < target_set_max):
        break
    
    # compute the increments of the Brownian motion 
    Binc = np.sum(dB[(j-1)*R: j*R])
    
    # compute drift and diffusion coefficients
    drift = - gradient_double_well_1d_potential(Xtemp)*dt_coarse
    diffusion = np.sqrt(2/beta) * Binc

    # compute X_t
    Xtemp = Xtemp + drift + diffusion
    Xem[j] = Xtemp


if j == L:
    print('The trajectory has NOT arrived to the target set!')
else:
    print('The trajectory HAS arrived to the target set!')

    # final time and final number of steps
    n = j*R
    t = n*dt
    
    Xem[j:L+1] = np.nan
    print('Steps needed: {:8.2f} \n'.format(n))
    print('Time needed: {:8.2f} \n'.format(t))

# plot trajectory
plot_trajectory(t_coarse, Xem)
