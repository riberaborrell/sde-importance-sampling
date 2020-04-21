import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def double_well_1d_potential(x):
    '''This method returns a potential function evaluated at the point/s x
    Args:
        x (float or float array) : posicion/s
    '''
    return 2 * (x**2 - 1)**2
    #return 0.5 * (x**2 - 1)**2

def double_well_1d_gradient(x):
    '''This method returns the gradient of a potential function evaluated
    at the point/s x
    Args:
        x (float or float array) : posicion/s
    '''
    return 8 * x * (x**2 - 1)
    #return 2 * x * (x**2 - 1)

def one_well_1d_potential(x):
    return (x - 1)**2

def one_well_1d_gradient(x):
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
        x (float or ndarry) : posision/s
        omegas (ndarray) : weights
        mus (ndarray): mean of each gaussian
        sigmas (ndarray) : standard deviation of each gaussian
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"
    
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)

    # get bias functions (gaussians) evalutated at x
    b = norm.pdf(x, mus, sigmas)

    # scalar product
    Vbias = np.dot(omegas, b)

    return Vbias

def bias_gradient(x, omegas, mus, sigmas):
    '''This method computes the gradient of the bias potential evaluated
       the point x
    
    Args:
        x (float) : posision
        omegas (array) : weights
        mus (array): mean of each gaussian
        sigmas (array) : standard deviation of each gaussian
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"
    
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)

    # get derivative of bias functions (gaussians) evalutated at x
    db = derivative_normal_pdf(x, mus, sigmas)

    # scalar product
    dVbias = np.dot(omegas, db)

    return dVbias
