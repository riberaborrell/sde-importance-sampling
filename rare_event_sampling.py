import argparse
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_potential_and_gradient


def potential(x):
    '''This method returns a potential function evaluated at the point x
    '''
    return 0.5*(x**2 - 1)**2


def gradient(x):
    '''This method returns the gradient of a potential function evaluated
    at the point x
    '''
    return 2*x*(x**2 - 1)


def bias_functions(x, K, means, sigmas):
    '''This method computes Gaussian densities given a mean and a
    variance (sigma**2)


    Args:
        x (float) : posision
        K (int) : number of bias functions  
        means (array): mean of each gaussian  
        sigmas (array) : square root of the variance of each gaussian


    Returns:
        b (array) : Gaussian densities with mean and variance given by
        means and sigmas**2 respectively evaluated at the point x 

    '''
    b = np.empty(K)
    
    for i in np.arange(K):
        norm_factor = sigmas[i]*np.sqrt(2*np.pi)
        b[i] = np.exp(-0.5*(x - means[i])**2 / (sigmas[i]**2)) / norm_factor

    return b
    

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

# potential and gradient
X = np.linspace(-1.5, 1.5, 100)
V = potential(X)
dV = gradient(X)
#plot_potential_and_gradient(X, V, dV)

# time steps
tzero = 0
T = 1
N = 10**4
dt = (T - tzero) / N

# bias functions
k = 200 
#TODO check that N (mod k) = 0
num_bias_functions = 1
means = np.empty(int(N/k))
sigmas = np.empty(int(N/k))

# weights
omegas = 0.1 * np.ones(int(N/k))
tau = 1

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
        num_bias_functions += 1
        means[num_bias_functions] = np.mean(Xhelp)
        sigmas[num_bias_functions] = np.max(np.abs(Xhelp - means[num_bias_functions]))
        
        Xhelp = np.empty(k+1)
    
    # stop loop after a limit number of time steps
    if n > N:
        #TODO write error
        print('Trajektorie zu lang!')
        break
    
    # compute dVbias
    b = bias_functions(
        x=Xt[n],
        K=num_bias_functions,
        means=means[:num_bias_functions + 1],
        sigmas=sigmas[:num_bias_functions + 1],
    )
    dVbias = tau * omegas[:num_bias_functions + 1] * b
    
    
    # solve SDE
    Xt[n+1]= Xt[n] + (dVbias - gradient(Xt[n]))*dt + dBt
    
    # update Xhelp
    Xhelp[np.mod(n, k)] = Xt[n]

    # update steps
    n += 1
        
print('Steps needed in the first run {:8.2f} \n'.format(n))
