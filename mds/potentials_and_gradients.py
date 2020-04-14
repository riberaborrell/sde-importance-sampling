import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def double_well_1d_potential(x):
    '''This method returns a potential function evaluated at the point/s x
    Args:
        x (float or float array) : posicion/s
    '''
    return 2 * (x**2 - 1)**2

def gradient_double_well_1d_potential(x):
    '''This method returns the gradient of a potential function evaluated
    at the point/s x
    Args:
        x (float or float array) : posicion/s
    '''
    return 8 * x * (x**2 - 1)

def one_well_1d_potential(x):
    return (x - 1)**2

def gradient_one_well_1d_potential(x):
    return 2 * (x - 1)

#TODO deprecated. Use norm.pdf(x, mu, sigma)
def normal_pdf(x, mu=0, sigma=1):
    '''This method evaluates the normal probability density with mean
    mu and standard deviation sigma at the point x.

    Args:
        x (float or ndarray) : posision
        mu (float or ndarray): mean
        sigma (float or ndarray) : standard deviation
    '''
    norm_factor = np.sqrt(2 * np.pi) * sigma
    return np.exp(-0.5 * ((x - mu) / sigma) **2 ) / norm_factor

def derivative_normal_pdf(x, mu=0, sigma=1):
    '''This method evaluates the derivative of the normal probability
       density function with mean mu and standard deviation sigma at 
       the point x.

    Args:
        x (float or ndarray) : posision
        mu (float or ndarray): mean
        sigma (float or ndarray) : standard deviation
    '''
    return norm.pdf(x, mu, sigma) * (mu - x) / sigma**2

def bias_potential(x, omegas, mus, sigmas):
    '''This method computes the bias potential evaluated at the point x
    
    Args:
        x (float) : posision
        omegas (array) : weights
        means (array): mean of each gaussian
        sigmas (array) : standard deviation of each gaussian
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"
    
    # get bias functions (gaussians) evalutated at x
    b = norm.pdf(x, mus, sigmas)

    # scalar product
    Vbias = np.sum(omegas * b)

    return Vbias

def gradient_bias_potential(x, omegas, mus, sigmas):
    '''This method computes the gradient of the bias potential evaluated
       the point x
    
    Args:
        x (float) : posision
        omegas (array) : weights
        means (array): mean of each gaussian
        sigmas (array) : standard deviation of each gaussian
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"

    # get derivative of bias functions (gaussians) evalutated at x
    db = derivative_normal_pdf(x, mus, sigmas)

    # scalar product
    dVbias = np.sum(omegas * db)

    return dVbias

def bias_potential_grid(X, omegas, mus, sigmas):
    '''This method computes the bias potential at the given grid points
    
    Args:
        X (array) : posisions
        omegas (array) : weights
        means (array): mean of each gaussian
        sigmas (array) : standard deviation of each gaussian

    Returns:
        Vbiased (array) : Biased potential evaluated at the array X
    '''
    # preallocate bias potential
    Vbias = np.zeros(len(X))
    
    for i, x in enumerate(X):
        # evaluate bias potential at x
        Vbias[i] = bias_potential(x, omegas, mus, sigmas)

    return Vbias

def gradient_bias_potential_grid(X, omegas, mus, sigmas):
    '''This method computes the gradient of the bias potential
    
    Args:
        X (array) : posisions
        omegas (array) : weights
        means (array): mean of each gaussian density
        sigmas (array) : standard deviation of each gaussian density

    Returns:
        dVbiased (float) : gradient of the bias potential evaluated at the array X
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"

    # preallocate gradient of bias potential
    dVbias = np.zeros(len(X))

    for i, x in enumerate(X):
        # evaluate gradien of the bias potential at x
        dVbias[i] = gradient_bias_potential(x, omegas, mus, sigmas)
    return dVbias
