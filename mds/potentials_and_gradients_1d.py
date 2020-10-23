import functools
import numpy as np

POTENTIAL_NAMES = [
    '1d_sym_1well',
    '1d_sym_2well',
    '1d_asym_2well',
]

def get_potential_and_gradient(potential_name, alpha=None):
    '''Given a potential name this methods returns the corresponding
       potential function, the gradient function, the formula and the
       chosen parameters.
    Args:
        potential_name (str) : label of the potential
        alpha (array) : parameters of the potential
    '''
    assert potential_name in POTENTIAL_NAMES, ''
    assert alpha.ndim == 1, ''

    if potential_name == '1d_sym_1well':
        assert alpha.shape[0] == 1, ''
        a = alpha[0]
        potential = functools.partial(one_well_1d_potential, a=a)
        gradient = functools.partial(one_well_1d_gradient, a=a)
        pot_formula = r'$V(x; a) = a x^2$'
        grad_formula = r'$\nabla V(x; a) = 2 a x$'
        parameters = r'a = {}'.format(a)

    elif potential_name == '1d_sym_2well':
        assert alpha.shape[0] == 1, ''
        a = alpha[0]
        potential = functools.partial(symmetric_double_well_1d_potential, a=a)
        gradient = functools.partial(symmetric_double_well_1d_gradient, a=a)
        pot_formula = r'$V(x; a) = a(x^2- 1)^2$'
        grad_formula = r'$\nabla V(x; a) = 4 a x (x^2- 1)$'
        parameters = r'a = {}'.format(a)

    elif potential_name == '1d_asym_2well':
        assert alpha.shape[0] == 4, ''
        a, b, c, d = alpha
        potential = asymmetric_double_well_1d_potential
        gradient = asymmetric_double_well_1d_gradient
        pot_formula = r'$V(x; a, b, c, d) = a(x^2 - b)^2 + c x + d$'
        grad_formula = r'$\nabla V(x; a, b, c, d) = 4 a x (x^2 - b) + c$'
        parameters = r'a = {}, b = {}, c = {}, d = {}'.format(a, b, c, d)

    return potential, gradient, pot_formula, grad_formula, parameters

def one_well_1d_potential(x, a=1):
    ''' Potential V(x; a) = a x^2 evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
    '''
    return a * x**2

def one_well_1d_gradient(x, a=1):
    ''' Gradient dV(x; a) = 2 a x evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
    '''
    return 2 * a * (x - b)

def symmetric_double_well_1d_potential(x, a=1):
    ''' Potential V(x; a) = a(x^2-1)^2 evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : barrier height parameter
    '''
    return a * (x**2 - 1)**2

def symmetric_double_well_1d_gradient(x, a=1):
    ''' Gradient dV(x; a) = 4 a x(x^2-1) evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
    '''
    return 4 * a * x * (x**2 - 1)

def asymmetric_double_well_1d_potential(x, a=1, b=1, c=-0.2, d=0.3):
    ''' Potential V(x; a, b, c, d) = a(x^2 - b)^2 + cx + d at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
        b (float) : parameter
        c (float) : parameter
        d (float) : parameter
    '''
    return a * (x**2 - b)**2 + c * x + d

def asymmetric_double_well_1d_gradient(x, a=1, b=1, c=-0.2, d=0.3):
    ''' Gradient dV(x; a, b, c, d) = 4 a x(x^2 - b) +c  evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
        b (float) : parameter
        c (float) : parameter
        d (float) : parameter
    '''
    return 4 * a * x * (x**2 - 1) + c

def symmetric_triple_well_1d_potential(x):
    #return 35*x**2 - 12*x**4 + x**6
    return 10*x**2 - 4*x**4 + x**6

def symmetric_triple_well_1d_gradient(x):
    return 70*x - 48*x**3 + 6*x**5

def asymmetric_triple_well_1d_potential(x):
    return (0.5*x**6 - 15*x**4 + 119*x**2 + 28*x + 50) / 200

def asymmetric_triple_well_1d_gradient(x):
    return (3*x**5 - 60*x**3 + 238*x + 28) / 200

