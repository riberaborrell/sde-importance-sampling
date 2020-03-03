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
    help='Set the well set interval. Default: [-1.9, -0.1]',
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

# check well set
well_set_min = args.well_set[0]
well_set_max = args.well_set[1]
if well_set_min >= well_set_max:
    #TODO raise error
    print("The well set interval is not valid")
    exit() 

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
pl = Plot(file_name='potential_and_gradient', file_type='png')
pl.potential_and_gradient(X, V, dV)

# time interval, time steps and number of time steps
tzero = 0
T = 6*10**2 # 60 seconds
N = 6*10**4 # 6000 thousand time steps
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
#omegas = 0.1 * np.ones(int(N/k))
omegas = np.zeros(int(N/k))
for j in range(int(N/k)):
    omegas[j] = 0.1 / (j + 1)


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


# Sampling with a fixed bias potential
omegas=omegas[:i]
mus=mus[:i]
sigmas=sigmas[:i]


N_traj = 10**4

# preallocate first hitting step/time array
FHs = np.zeros(N_traj)
FHt = np.zeros(N_traj)

# preallocate Girsanov Martingale
M_FHt = np.zeros(N_traj)
M_T = np.zeros(N_traj)

# preallocate observable of interest
# for f=1, g=0, and therefore W = tau t
I = np.ones(N_traj)

# sample of N_traj trajectories 
for i in np.arange(N_traj):
    # preallocate EM solution
    Xem = np.zeros(N+1) 
    Xtemp = args.xzero
    Xem[0] = Xtemp

    # preallocate martingale terms, M_t = e^(M1_t + M2_t)
    M1em = np.zeros(N+1)
    M2em = np.zeros(N+1)

    # not been in target set yet
    has_been_in_target_set = False

    for n in np.arange(1, N+1):
        # compute gradient of the bias potential evaluated at x
        dVbias = gradient_bias_potential(
            x=Xtemp,
            omegas=omegas,
            mus=mus,
            sigmas=sigmas,
        )

        # compute Xem
        drift = (- gradient_double_well_1d_potential(Xtemp) - dVbias)*dt
        diffusion = np.sqrt(2 / beta) * dB[n-1]
        Xtemp = Xtemp + drift + diffusion
        Xem[n] = Xtemp

        # evaluate the control drift
        Ut = -1 * dVbias / np.sqrt(2 / beta)

        # compute martingale terms
        M1em[n] = M1em[n-1] - Ut * dB[n-1]
        M2em[n] = M2em[n-1] - (1/2)* (Ut**2) * dt 

        # check if we have arrived to the target set
        if (has_been_in_target_set == False and
            Xtemp >= target_set_min and 
            Xtemp <= target_set_max):
            
            # change flag
            has_been_in_target_set = True

            # save first hitting time/step
            FHs[i] = n 
            FHt[i] = n * dt

            # compute martingale at the fht
            M_FHt[i] = np.exp(M1em[n] + M2em[n])
            
            # compute quantity of interest at the fht
            I[i] = np.exp(-beta * FHt[i] * dt) * M_FHt[i] 

    # recall that 
    # if not has_been_in_target_set:
    #    FHs[i] = 0 
    #    FHt[i] = 0
    #    M_FHt[i] = 0
    #    I[i] = 0
    
    # compute Martingale at time T
    M_T[i] = np.exp(M1em[N] + M2em[N])

# compute mean and variance of I
mean_I = np.mean(np.array([x for x in I if x > 0 ]))
var_I = np.var(np.array([x for x in I if x > 0 ]))
re_I = np.sqrt(var_I) / mean_I

print('Expectation of exp(-beta*tau*t)Mt: {:2.8f}'.format(mean_I))
print('Variance of exp(-beta*tau*t)Mt: {:2.8f}'.format(var_I))
print('Relative error of exp(-beta*tau*t)Mt: {:2.8f}'.format(re_I))
