import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def double_well_1d_potential(x):
    '''This method returns a potential function evaluated at the point x
    '''
    return 0.5*(x**2 - 1)**2

def gradient_double_well_1d_potential(x):
    '''This method returns the gradient of a potential function evaluated
    at the point x
    '''
    return 2*x*(x**2 - 1)

def normal_probability_density(x, mu, sigma):
    '''This method evaluates the normal probability density with mean
    mu and standard deviation sigma at the point x.

    Args:
        x (float) : posision
        mu (float): mean
        sigma (float) : standard deviation
    '''
    # norm_factor = np.sqrt(2 * np.pi) * sigma
    # return np.exp(-0.5 * ((x - mu) / sigma) **2 ) / norm_factor
    return norm.pdf(x, mu, sigma)

def derivative_normal_probability_density(x, mu, sigma):
    '''This method evaluates the derivative of the normal probability
       density with mean mu and standard deviation sigma at the point x.

    Args:
        x (float) : posision
        mu (float): mean
        sigma (float) : standard deviation
    '''
    return - ((x - mu) / sigma**2) * norm.pdf(x, mu, sigma) 

def bias_functions(x, mus, sigmas):
    '''This method computes Gaussian densities given a mean and a
    standard deviation

    Args:
        x (float) : posision
        means (array): mean of each gaussian density
        sigmas (array) : standard deviation of each gaussian density

    Returns:
        b (array) : Gaussian densities with mean and variance given by
        means and sigmas**2 respectively evaluated at the point x 

    '''
    assert mus.shape == sigmas.shape, "Error"

    # number of bias functions
    K = len(mus)

    # preallocate basis functions
    b = np.zeros(K)
    
    for i in np.arange(K):
        b[i] = norm.pdf(x, mus[i], sigmas[i]) 

    return b
    
def bias_potential(x, mus, sigmas, omegas):
    '''This method computes the bias potential
    
    Args:
        x (float) : posision
        means (array): mean of each gaussian density
        sigmas (array) : standard deviation of each gaussian density

    Returns:
        Vbiased (float) : Biased potential evaluated at the point x
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"
    
    # get bias functions evalutated at x
    b = bias_functions(
        x=x,
        mus=mus,
        sigmas=sigmas,
    )

    Vbias = np.sum(omegas*b)
    return Vbias

def gradient_bias_potential(x, omegas, mus, sigmas):
    '''This method computes the gradient of the bias potential
    
    Args:
        x (float) : posision
        omegas (array) : weights
        means (array): mean of each gaussian density
        sigmas (array) : standard deviation of each gaussian density

    Returns:
        dVbiased (float) : gradient of the bias potential evaluated at the point x
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"

    # get bias functions evalutated at x
    b = bias_functions(
        x=x,
        mus=mus,
        sigmas=sigmas,
    )

    dVbias = np.sum(-omegas * ((x - mus)/sigmas**2) * b)
    return dVbias

def bias_potential2(X, mus, sigmas, omegas):
    '''This method computes the bias potential
    
    Args:
        X (array) : posisions
        means (array): mean of each gaussian density
        sigmas (array) : standard deviation of each gaussian density
        omegas (array) : weights

    Returns:
        Vbiased (array) : Biased potential evaluated at the array X
    '''
    assert omegas.shape == mus.shape == sigmas.shape, "Error"

    # preallocate bias potential
    Vbias = np.zeros(len(X))
    
    for i, x in enumerate(X):
        # get bias functions evalutated at x
        b = bias_functions(
            x=x,
            mus=mus,
            sigmas=sigmas,
        )
        Vbias[i] = np.dot(omegas, b)
    return Vbias

def gradient_bias_potential2(X, omegas, mus, sigmas):
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
        # get bias functions evalutated at x
        b = bias_functions(
            x=x,
            mus=mus,
            sigmas=sigmas,
        )
        dVbias[i] = np.sum(-omegas * ((x - mus)/sigmas**2) * b)
    return dVbias
