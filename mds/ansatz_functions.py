import numpy as np
import scipy.stats as stats

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

def fe_1d_1o_basis_function(x, omega_min, omega_max, num_elements, j):
    nodes = np.linspace(omega_min, omega_max, num_elements + 1)
    m = nodes.shape[0]
    h = nodes[1] - nodes[0]
    if j == 1:
        condlist = [x < nodes[j], x >= nodes[j]]
        functlist = [lambda x: (nodes[j] - x )/h, 0]
    elif j >= 2 and j <= m-1:
        condlist = [
            x < nodes[j-2],
            (x >= nodes[j-2]) & (x < nodes[j-1]),
            (x >= nodes[j-1]) & (x < nodes[j]),
            x >= nodes[j],
        ]
        functlist = [0, lambda x: (x - nodes[j-2])/h, lambda x: (nodes[j] - x )/h, 0 ]
    else:
        condlist = [x < nodes[j-2], x >= nodes[j-2]]
        functlist = [0, lambda x: (x - nodes[j-2])/h]

    phi = np.piecewise(x, condlist, functlist)
    return phi

def fe_1d_1o_basis_function_derivative(x, omega_min, omega_max, num_elements, j):
    nodes = np.linspace(omega_min, omega_max, num_elements + 1)
    m = nodes.shape[0]
    h = nodes[1] - nodes[0]
    if j == 1:
        condlist = [x < nodes[j], x >= nodes[j]]
        functlist = [-1 / h, 0]
    elif j >= 2 and j <= m-1:
        condlist = [
            x < nodes[j-2],
            (x >= nodes[j-2]) & (x < nodes[j-1]),
            (x >= nodes[j-1]) & (x < nodes[j]),
            x >= nodes[j],
        ]
        functlist = [0, 1 / h, -1 / h, 0 ]
    else:
        condlist = [x < nodes[j-2], x >= nodes[j-2]]
        functlist = [0, 1 / h]

    phi = np.piecewise(x, condlist, functlist)
    return phi

def fe_1d_1o_basis_function2(x, omega_min, omega_max, m, j):
    omega_h = np.linspace(omega_min, omega_max, m)
    h = omega_h[1] - omega_h[0]
    breakpoint()
    condlist = [
        (j >= 2) & (j <= m) & (x < omega_h[j-2]),
        (j >= 2) & (j <= m) & (x >= omega_h[j-2]) & (x < omega_h[j-1]),
        (j >= 1) & (j <= m-1) & (x >= omega_h[j-1]) & (x < omega_h[j]),
        (j >= 1) & (j <= m-1) & (x >= omega_h[j]),
    ]
    functlist = [
        0,
        lambda x: (x - omega_h[j-2])/h,
        lambda x: (omega_h[j] - x )/h,
        0,
    ]

    phi = np.piecewise(x, condlist, functlist)
    return phi


def fe_1d_1o_basis(x, omega_min, omega_max, num_elements):
    m = num_elements + 1
    nodal_basis = np.zeros((x.shape[0], m))
    for j in np.arange(1, m+1):
        nodal_basis[:, j-1] = fe_1d_1o_basis_function(x, omega_min, omega_max, num_elements, j)
    return nodal_basis

def fe_1d_1o_basis_derivative(x, omega_min, omega_max, num_elements):
    m = num_elements + 1
    nodal_basis_derivative = np.zeros((x.shape[0], m))
    for j in np.arange(1, m+1):
        nodal_basis_derivative[:, j-1] = fe_1d_1o_basis_function_derivative(x, omega_min, omega_max, num_elements, j)
    return nodal_basis_derivative
