import functools
import numpy as np

POTENTIAL_NAMES = [
    '2d_1well',
    '2d_4well',
]

def get_potential_and_gradient(potential_name, alpha=None):
    '''Given a potential name this methods returns the corresponding
       potential function and gradient function.
    Args:
        potential_name (str) : label of the potential
        alpha (array) : parameters of the potential
    '''
    assert potential_name in POTENTIAL_NAMES, ''
    assert alpha.ndim == 1, ''

    if potential_name == '2d_1well':
        a = alpha[:2]
        b = alpha[2:]
        potential = functools.partial(one_well_2d_potential, a=a, b=b)
        gradient = functools.partial(one_well_2d_gradient, a=a, b=b)
        pot_formula = r'$V(x; a, b) = a_1(x - b_1)^2 + a_2(y - b_2)^2$'
        grad_formula = r'$\nabla V(x; a, b) = 2 a_1 (x - b_1) + 2 a_2(y - b_2)$'
        parameters = r'$a_1={}, a_2={}, b_1={}, b_2={}$'.format(
            a[0], a[1], b[0], b[1]
        )

    elif potential_name == '2d_4well':
        a = alpha[:2]
        b = alpha[2:]
        potential = functools.partial(quadruple_well_2d_potential, a=a, b=b)
        gradient = functools.partial(quadruple_well_2d_gradient, a=a, b=b)
        pot_formula = r'$V(x; a, b) = a_1(x^2 - b_1)^2 + a_2(y^2 - b_2)^2$'
        grad_formula = r'$\nabla V(x; a, b) = 4 a_1 x (x^2 - b_1)' + \
                                           '+ 4 a_2 y (y^2 - b_2)$'
        parameters = r'$a_1={}, a_2={}, b_1={}, b_2={}$'.format(
            a[0], a[1], b[0], b[1]
        )

    return potential, gradient, pot_formula, grad_formula, parameters

def one_well_2d_potential(x, a, b):
    ''' Potential V(x; a, b) evaluated at x
        x (array) : posicion
        a (array) : parameter
        b (float) : center
    '''
    assert x.shape[1] == 2, ''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    pot = a[0] * np.power(x[:, 0] - b[0], 2) \
        + a[1] * np.power(x[:, 1] - b[1], 2)
    return pot

def one_well_2d_gradient(x, a, b):
    ''' Gradient dV(x; a, b) evaluated at x
        x (array) : posicion
        a (array) : parameter
        b (float) : center
    '''
    assert x.shape[1] == 2, ''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    grad_x = (a[0] * 2 * (x[:, 0] - b[0])).reshape((x.shape[0], 1))
    grad_y = (a[1] * 2 * (x[:, 1] - b[1])).reshape((x.shape[0], 1))
    grad = np.hstack((grad_x, grad_y))
    return grad

def quadruple_well_2d_potential(x, a, b):
    ''' Potential V(x; a, b) evaluated at x
        x ((M, 2)-array) : posicion
        a (array) : barrier height
        b (float) : center
    '''
    assert x.ndim == 2, ''
    assert x.shape[1] == 2, ''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    pot = a[0] * np.power(np.power(x[:, 0], 2) - b[0], 2) \
        + a[1] * np.power(np.power(x[:, 1], 2) - b[1], 2)
    return pot

def quadruple_well_2d_gradient(x, a, b):
    ''' Gradient dV(x; a, b) evaluated at x
        x ((M, 2)-array) : posicion
        x (array) : posicion
        a (array) : barrier height
        b (float) : center
    '''
    assert x.ndim == 2, ''
    assert x.shape[1] == 2, ''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    grad_x = (a[0] * 4 * x[:, 0] * (np.power(x[:, 0], 2) - b[0])).reshape((x.shape[0], 1))
    grad_y = (a[1] * 4 * x[:, 1] * (np.power(x[:, 1], 2) - b[1])).reshape((x.shape[0], 1))
    grad = np.hstack((grad_x, grad_y))
    return grad
