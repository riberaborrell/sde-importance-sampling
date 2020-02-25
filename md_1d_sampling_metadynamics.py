import argparse
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_potential_and_gradient
from tools import double_well_1d_potential, \
                  gradient_double_well_1d_potential, \
                  bias_functions, \
                  bias_potential, \
                  gradient_bias_potential

parser = argparse.ArgumentParser(description='Brownian path simulation')
parser.add_argument(
    '--seed',
    dest='seed',
    type=int,
    help='Set the seed for RandomState',
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

# time steps
tzero = 0
T = 1
N = 10**4
dt = (T - tzero) / N

# steps before adding a bias function
k = 10 
#TODO check that N (mod k) = 0

# bias functions
i = 0

means = np.empty(int(N/k))
sigmas = np.empty(int(N/k))

# weights
omegas = 0.1 * np.ones(int(N/k))

# 1D MD SDE: dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x
Xt = np.empty(N+1) 
Xt[0] = 1
beta = 3

Xhelp = np.empty(k+1)


# diffusion term
dBt = np.sqrt(2 * dt / beta) * np.random.normal(0, 1, N)

# initialize step number
n = 0

# loop until ?
while (Xt[n] > 0.9 and Xt[n] < 1.1):
    
    # every k-steps add new bias function
    if (n > 0 and np.mod(n, k) == 0):
        means[i] = np.mean(Xhelp)
        sigmas[i] = np.max(np.abs(Xhelp - means[i]))
        Xhelp = np.empty(k+1)
        i += 1
    
    # stop loop after a limit number of time steps
    if n > N:
        #TODO write error
        print('Trajektorie zu lang!')
        break
    
    # compute dVbias
    if i == 0:
        dVbias = 0
    else:
        b = bias_functions(
            x=Xt[n],
            K=i,
            means=means[:i],
            sigmas=sigmas[:i],
        )
        dVbias = gradient_bias_potential(
            x=Xt[n],
            b=b,
            omegas=omegas[:i],
            means=means[:i],
            sigmas=sigmas[:i],
        )
    
    # solve SDE
    Xt[n+1]= Xt[n] + (dVbias - gradient(Xt[n]))*dt + dBt[n]
    
    # update Xhelp
    Xhelp[np.mod(n, k)] = Xt[n]

    # update steps
    n += 1
        
print('Steps needed to leave region O: {:8.2f} \n'.format(n))
print('Bias functions needed for the bias potential: {:d}\n'.format(i))
#exit()

# Evaluate the bias potential and its gradient for every point in the grid
Vbias = np.empty(X.shape[0]) 
dVbias = np.empty(X.shape[0]) 
j = 0
for x in X:
    b = bias_functions(
        x=x,
        K=i,
        means=means[:i],
        sigmas=sigmas[:i],
    )
    Vbias[j] = bias_potential(
        b=b, 
        omegas=omegas[:i],
    )
    dVbias[j] = gradient_bias_potential(
        x=x,
        b=b,
        omegas=omegas[:i],
        means=means[:i],
        sigmas=sigmas[:i],
    )
    j += 1

# plot bias potential and gradient
plot_potential_and_gradient(X, Vbias, dVbias)

















