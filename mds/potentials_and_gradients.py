import numpy as np
from scipy import stats

def get_potential_and_gradient(potential_name):
    '''Given a potential name this methods returns the corresponding
       potential function and gradient function.
    Args:
        potential_name (str) : label of the potential
    '''
    if potential_name == 'sym_1well':
        potential = one_well_potential
        gradient = one_well_gradient

    elif potential_name == 'sym_2well':
        potential = symmetric_double_well_potential
        gradient = symmetric_double_well_gradient

    elif potential_name == 'asym_2well':
        potential = asymmetric_double_well_potential
        gradient = asymmetric_double_well_gradient

    return potential, gradient

def one_well_potential(x):
    ''' Potential V(x)=(x-1)^2 evaluated at x
    Args:
        x (float or float array) : posicion/s
    '''
    return (x - 1)**2

def one_well_gradient(x):
    ''' Gradient dV(x)=2(x-1) evaluated at x
    Args:
        x (float or float array) : posicion/s
    '''
    return 2 * (x - 1)

def symmetric_double_well_potential(x, alpha=1):
    ''' Potential V(x;alpha)=alpha(x^2-1)^2 evaluated at x
    Args:
        x (float or float array) : posicion/s
        alpha (float) : parameter
    '''
    return alpha * (x**2 - 1)**2

def symmetric_double_well_gradient(x, alpha=1):
    ''' Gradient dV(x;alpha)=4 alpha x(x^2-1) evaluated at x
    Args:
        x (float or float array) : posicion/s
        alpha (float) : parameter
    '''
    return 4 * alpha * x * (x**2 - 1)

def asymmetric_double_well_potential(x):
    ''' Potential V(x)=(x^2-1)^2 - 0.2x + 0.3 at x
    Args:
        x (float or float array) : posicion/s
    '''
    return (x**2 - 1)**2 - 0.2*x + 0.3

def asymmetric_double_well_gradient(x):
    ''' Gradient dV(x)=4x(x^2-1) -0.2  evaluated at x
    Args:
        x (float or float array) : posicion/s
    '''
    return 4 * x * (x**2 - 1) - 0.2


#TODO deprecated. Use stats.norm.pdf(x, mu, sigma)
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
    return stats.norm.pdf(x, mu, sigma) * (mu - x) / sigma**2

def bias_potential(x, omegas, mus, sigmas):
    '''This method computes the bias potential evaluated at the point x
    
    Args:
        x (float or ndarry) : posision/s
        omegas (ndarray) : weights
        mus (ndarray): mean of each gaussian
        sigmas (ndarray) : standard deviation of each gaussian
    '''
    assert omegas.shape == mus.shape == sigmas.shape
    
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)

    # get bias functions (gaussians) evalutated at x
    b = stats.norm.pdf(x, mus, sigmas)

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
    assert omegas.shape == mus.shape == sigmas.shape
    
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)

    # get derivative of bias functions (gaussians) evalutated at x
    db = derivative_normal_pdf(x, mus, sigmas)

    # scalar product
    dVbias = np.dot(omegas, db)

    return dVbias
