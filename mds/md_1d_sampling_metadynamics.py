import argparse
import numpy as np
import matplotlib.pyplot as plt

import os

from plotting import Plot

from tools import double_well_1d_potential, \
                  gradient_double_well_1d_potential, \
                  bias_functions, \
                  bias_potential, \
                  bias_potential2, \
                  gradient_bias_potential, \
                  gradient_bias_potential2

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
    '--well-set',
    nargs=2,
    dest='well_set',
    type=float,
    default=[-1.9, -0.1],
    help='Set the target set interval. Default: [-1.9, -0.1]',
)
args = parser.parse_args()

# set random seed
if args.seed:
    np.random.seed(args.seed)

# check target set
well_set_min = args.well_set[0]
well_set_max = args.well_set[1]
if well_set_min >= well_set_max:
    #TODO raise error
    print("The well set interval is not valid")
    exit() 

    
# grid x coordinate
X = np.linspace(-2, 2, 100)

# potential and gradient
V = double_well_1d_potential(X)
dV = gradient_double_well_1d_potential(X)
pl = Plot(file_name='potential_and_gradient', file_type='png')
pl.potential_and_gradient(X, V, dV)

# time interval, time steps and number of time steps
tzero = 0
T = 6*10 # 60 seconds
N = 6*10**3 # 6000 thousand time steps
dt = (T - tzero) / N # 0,01
t = np.linspace(tzero, T, N+1)

# steps before adding a bias function
k = 500 
if np.mod(N, k) != 0:
    print("N has to be a multipel of the number of steps k")
    exit()
# bias functions
i = 0

# preallocate mean and standard deviation for the bias funcitons
mus = np.zeros(int(N/k))
sigmas = np.zeros(int(N/k))

# set the weights of the bias functions
omegas = 0.1 * np.ones(int(N/k))

# 1D MD SDE: dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x
beta = args.beta
Xem = np.zeros(N+1) 
Xtemp = args.xzero
Xem[0] = Xtemp

Xhelp = np.zeros(k+1)

# Brownian increments
dB = np.sqrt(dt)*np.random.normal(0, 1, N)

for n in np.arange(1, N+1): 
    # stop simulation if particle leave the well set T
    if (Xtemp < well_set_min or Xtemp > well_set_max):
        Xem[n:N+1] = np.nan
        break

    # every k-steps add new bias function
    if (np.mod(n, k) == 0):
        mus[i] = np.mean(Xhelp)
        sigmas[i] = np.max(np.abs(Xhelp - mus[i])) / 5
        Xhelp = np.zeros(k+1)
        i += 1

        # plot tilted potential

        Vbias = bias_potential2(
            X=X,
            mus=mus[:i],
            sigmas=sigmas[:i],
            omegas=omegas[:i],
        )
        pl = Plot(file_name='tilted_pot_i_' + str(i), file_type='png')
        pl.tilted_potential(X, V, Vbias)
        
        # plot gradient of the tilted potential
        dVbias = gradient_bias_potential2(
            X=X,
            mus=mus[:i],
            sigmas=sigmas[:i],
            omegas=omegas[:i],
        )
        pl = Plot(file_name='tilted_grad_i_' + str(i), file_type='png')
        pl.gradient_tilted_potential(X, dV, dVbias)
    
    # compute dVbias
    if i == 0:
        Vbias = 0
        dVbias = 0
    else:
        Vbias = bias_potential(
            x=Xtemp,
            omegas=omegas[:i],
            mus=mus[:i],
            sigmas=sigmas[:i],
        )
        dVbias = gradient_bias_potential(
            x=Xtemp,
            omegas=omegas[:i],
            mus=mus[:i],
            sigmas=sigmas[:i],
        )

    # compute drift and diffusion coefficients
    #drift = (- gradient_double_well_1d_potential(Xtemp) + Vbias)*dt
    drift = (- gradient_double_well_1d_potential(Xtemp) - dVbias)*dt
    diffusion = np.sqrt(2 / beta) * dB[n-1]

    # compute Xtemp
    Xtemp = Xtemp + drift + diffusion
    Xem[n] = Xtemp
    
    # update Xhelp
    Xhelp[np.mod(n, k)] = Xtemp


if n == N:
    print('The trajectory has NOT left the well set!')
else:
    print('The trajectory HAS left the well set!')
    print('Steps needed: {:8.2f} \n'.format(n))

# plot trajectory
pl = Plot(file_name='trajectory', file_type='png')
pl.trajectory(t, Xem)
        
print('Bias functions used: {:d}\n'.format(i))
