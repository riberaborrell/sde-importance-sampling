import functools
import numpy as np

POTENTIAL_NAMES = [
    'nd_2well',
]

def get_potential_and_gradient(n, potential_name, alpha=None):
    '''Given a potential name this methods returns the corresponding
       potential function and gradient function.
    Args:
        n (int) : dimension of the potential
        potential_name (str) : label of the potential
        alpha (array) : parameters of the potential
    '''
    assert potential_name in POTENTIAL_NAMES, ''
    assert alpha.ndim == 1, ''

    if potential_name == 'nd_2well':
        assert alpha.shape[0] == n, ''
        potential = functools.partial(double_well_nd_potential, a=alpha)
        gradient = functools.partial(double_well_nd_gradient, a=alpha)
        parameters = ''
        for i in range(n):
            parameters += r'$\alpha_{:d}={}$'.format(i+1, alpha[i])

    return potential, gradient, parameters

def double_well_nd_potential(x, a):
    ''' Potential V(x; a) evaluated at x
        x ((N, n)-array) : posicion
        a (n-array) : barrier height

        return pot (N-array)
    '''
    assert x.ndim == 2, ''
    assert a.ndim == 1, ''
    assert x.shape[1] == a.shape[0], ''

    N = x.shape[0]
    n = x.shape[1]

    pot = np.zeros(N)
    for i in range(n):
        pot += a[i] * np.power(np.power(x[:, i], 2) - 1, 2)
    return pot

def double_well_nd_gradient(x, a):
    ''' Gradient dV(x; a) evaluated at x
        x ((N, n)-array) : posicion
        a (n-array) : barrier height

        return grad ((N, n)-array)
    '''
    assert x.ndim == 2, ''
    assert a.ndim ==  1, ''
    assert x.shape[1] == a.shape[0], ''

    N = x.shape[0]
    n = x.shape[1]

    grad = np.zeros((N, n))
    for i in range(n):
        grad[:, i] = a[i] * 4 * x[:, i] * (np.power(x[:, i], 2) - 1)

    return grad
