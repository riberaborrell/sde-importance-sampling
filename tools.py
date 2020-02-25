import numpy as np
import matplotlib.pyplot as plt


def double_well_1d_potential(x):
    '''This method returns a potential function evaluated at the point x
    '''
    return 0.5*(x**2 - 1)**2


def gradient_double_well_1d_potential(x):
    '''This method returns the gradient of a potential function evaluated
    at the point x
    '''
    return 2*x*(x**2 - 1)


def bias_functions(x, K, means, sigmas):
    '''This method computes Gaussian densities given a mean and a
    standard deviation

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
    
def bias_potential(b, omegas):
    '''This method computes the bias potential
    
    Args:
        b (array) : bias functions evaluated at the point x
        omegas (array) : weights

    Returns:
        Vbiased (float) : Biased potential evaluated at the point x
    '''
    assert b.shape == omegas.shape, "Error"

    Vbias = np.sum(omegas*b)
    return Vbias

def gradient_bias_potential(x, b, omegas, means, sigmas):
    '''This method computes the gradient of the bias potential
    
    Args:
        x (float) : posision
        b (array) : bias functions evaluated at the point x
        omegas (array) : weights
        means (array): mean of each gaussian
        sigmas (array) : square root of the variance of each gaussian

    Returns:
        dVbiased (float) : gradient of the bias potential evaluated at the point x
    '''
    assert b.shape == omegas.shape == means.shape == sigmas.shape, "Error"

    dVbias = np.sum(-omegas * ((x - means)/sigmas) * b)
    return dVbias
