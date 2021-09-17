import functools
import numpy as np
import torch

POTENTIAL_NAMES = [
    'nd_2well',
    'nd_2well_asym',
]

def get_potential_and_gradient(potential_name, n, alpha=None, tensor=False):
    '''Given a potential name this method returns the corresponding
       potential function and gradient function.
    Args:
        n (int) : dimension of the potential
        potential_name (str) : label of the potential
        alpha (array) : parameters of the potential
    '''
    assert potential_name in POTENTIAL_NAMES, ''
    assert alpha.ndim == 1, ''

    if potential_name in ['nd_2well', 'nd_2well_asym']:
        assert alpha.shape[0] == n, ''
        potential = functools.partial(double_well_nd_potential, alpha=alpha, tensor=tensor)
        gradient = functools.partial(double_well_nd_gradient, alpha=alpha, tensor=tensor)
        parameters = ''
        for i in range(n):
            parameters += r'$\alpha_{:d}={}$'.format(i+1, alpha[i])

    return potential, gradient, parameters

def double_well_nd_potential(x, alpha, tensor=False):
    ''' Potential V(x; alpha) evaluated at x
        x ((N, n)-array) : posicion
        alpha (n-array) : barrier height

        return pot (N-array)
    '''
    assert x.ndim == 2, ''
    assert alpha.ndim == 1, ''
    assert x.shape[1] == alpha.shape[0], ''

    N = x.shape[0]
    n = x.shape[1]

    potential = np.zeros(N)
    for i in range(n):
        potential += alpha[i] * np.power(np.power(x[:, i], 2) - 1, 2)
    return potential

def double_well_nd_gradient(x, alpha, tensor=False):
    ''' Gradient dV(x; alpha) evaluated at x
        x ((N, n)-array) : posicion
        alpha (n-array) : barrier height

        return grad ((N, n)-array)
    '''
    assert x.ndim == 2, ''
    assert alpha.ndim ==  1, ''
    assert x.shape[1] == alpha.shape[0], ''

    N = x.shape[0]
    n = x.shape[1]

    if not tensor:
        gradient = np.empty((N, n))
        for i in range(n):
            gradient[:, i] = alpha[i] * 4 * x[:, i] * (np.power(x[:, i], 2) - 1)
    else:
        gradient = torch.empty((N, n))
        for i in range(n):
            gradient[:, i] = alpha[i] * 4 * x[:, i] * (torch.pow(x[:, i], 2) - 1)

    return gradient

def line_nd_potential(x, tensor=False):
    ''' Potential V(x) evaluated at x
        x ((N, n)-array) : posicion

        return pot (N-array)
    '''
    assert x.ndim == 2, ''

    N = x.shape[0]
    n = x.shape[1]

    potential = np.zeros(N)
    for i in range(n):
        potential += alpha[i] * np.power(np.power(x[:, i], 2) - 1, 2)
    return potential
