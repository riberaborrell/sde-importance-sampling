import functools

POTENTIAL_NAMES = [
    '1d_sym_1well',
    '1d_sym_2well',
    '1d_asym_2well',
    #'1d_sym_3well',
    #'1d_asym_3well',
    '2d_1well',
    '2d_4well',
]

POTENTIAL_TITLES = {
    '1d_sym_1well': r'$V(x;a,b) = a(x- b)^2$',
    '1d_sym_2well': r'$V(x;a) = a(x^2 - 1)^2$',
    '1d_asym_2well': r'$V(x) = (x^2 - 1)^2 - 0.2 x + 0.3$',
    #'1d_sym_3well': '',
    #'1d_asym_3well': '',
    '2d_1well': r'$V(x;a,b) = a_1(x - b_1)^2 + a_2(y - b_2)^2$',
    '2d_4well': r'$V(x;a,b) = a_1(x^2 - b_1)^2 + a_2(y^2 - b_2)^2$',
}

POTENTIAL_LABELS = {
    '1d_sym_1well': r'a = {}, b = {}',
    '1d_sym_2well': r'a = {}',
    '1d_asym_2well': '',
    '2d_1well': r'$a_1={}, a_2={}, b_1={}, b_2={}$',
    '2d_3well': r'$a_1={}, a_2={}, b_1={}, b_2={}$',
}

def get_potential_and_gradient(potential_name, alpha=None):
    '''Given a potential name this methods returns the corresponding
       potential function and gradient function.
    Args:
        potential_name (str) : label of the potential
        alpha (array) : parameters of the potential
    '''
    assert potential_name in POTENTIAL_NAMES, ''
    assert alpha.ndim == 1, ''

    if potential_name == '1d_sym_1well':
        a, b = alpha
        potential = functools.partial(one_well_1d_potential, a=a, b=b)
        gradient = functools.partial(one_well_1d_gradient, a=a, b=b)
    elif potential_name == '1d_sym_2well':
        a = alpha[0]
        potential = functools.partial(symmetric_double_well_1d_potential, a=a)
        gradient = functools.partial(symmetric_double_well_1d_gradient, a=a)
    elif potential_name == '1d_asym_2well':
        potential = asymmetric_double_well_1d_potential
        gradient = asymmetric_double_well_1d_gradient
    elif potential_name == '1d_sym_3well':
        potential = symmetric_triple_well_1d_potential
        gradient = symmetric_triple_well_1d_gradient
    elif potential_name == '1d_asym_3well':
        potential = asymmetric_triple_well_1d_potential
        gradient = asymmetric_triple_well_1d_gradient
    elif potential_name == '2d_1well':
        a = alpha[:2]
        b = alpha[2:]
        potential = functools.partial(one_well_2d_potential, a=a, b=b)
        gradient = None
    elif potential_name == '2d_4well':
        a = alpha[:2]
        b = alpha[2:]
        potential = functools.partial(quadruple_well_2d_potential, a=a, b=b)
        gradient = functools.partial(quadruple_well_2d_gradient, a=a, b=b)

    return potential, gradient

def one_well_1d_potential(x, a=1.0, b=0):
    ''' Potential V(x;a,b) = a(x-b)^2 evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
        b (float) : center
    '''
    return a * (x - b)**2

def one_well_1d_gradient(x, a=1.0, b=0):
    ''' Gradient dV(x;a,b) = 2a(x-b) evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
        b (float) : center
    '''
    return 2 * a * (x - b)

def symmetric_double_well_1d_potential(x, a=1.0):
    ''' Potential V(x;a) = a(x^2-1)^2 evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : barrier height parameter
    '''
    return a * (x**2 - 1)**2

def symmetric_double_well_1d_gradient(x, a=1.0):
    ''' Gradient dV(x;a) = 4 a x(x^2-1) evaluated at x
    Args:
        x (float or float array) : posicion/s
        a (float) : parameter
    '''
    return 4 * a * x * (x**2 - 1)

def asymmetric_double_well_1d_potential(x):
    ''' Potential V(x) = (x^2-1)^2 - 0.2x + 0.3 at x
    Args:
        x (float or float array) : posicion/s
    '''
    return (x**2 - 1)**2 - 0.2 * x + 0.3

def asymmetric_double_well_1d_gradient(x):
    ''' Gradient dV(x) = 4x(x^2-1) -0.2  evaluated at x
    Args:
        x (float or float array) : posicion/s
    '''
    return 4 * x * (x**2 - 1) - 0.2

def symmetric_triple_well_1d_potential(x):
    #return 35*x**2 - 12*x**4 + x**6
    return 10*x**2 - 4*x**4 + x**6

def symmetric_triple_well_1d_gradient(x):
    return 70*x - 48*x**3 + 6*x**5

def asymmetric_triple_well_1d_potential(x):
    return (0.5*x**6 - 15*x**4 + 119*x**2 + 28*x + 50) / 200

def asymmetric_triple_well_1d_gradient(x):
    return (3*x**5 - 60*x**3 + 238*x + 28) / 200

def one_well_2d_potential(x, y, a, b):
    ''' Potential V(x,y;a1,a2,b1,b2) evaluated at (x, y)
        x (float or float array) : x posicion/s
        y (float or float array) : y posicion/s
        a (array) : parameter
        b (float) : center
    '''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    return a[0] * (x - b[0])**2 + a[1] * (y - b[1])**2

def quadruple_well_2d_potential(x, y, a, b):
    ''' Potential V(x,y;a1,a2,b1,b2) evaluated at (x, y)
        x (float or float array) : x posicion/s
        y (float or float array) : y posicion/s
        a (array) : barrier height
        b (float) : center
    '''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    return a[0] * (x**2 - b[0])**2 + a[1] * (y**2 - b[1])**2

def quadruple_well_2d_gradient(x, y, a, b):
    ''' Gradient dV(x,y;a1,a2,b1,b2) evaluated at (x, y)
        x (float or float array) : x posicion/s
        y (float or float array) : y posicion/s
        a (array) : barrier height
        b (float) : center
    '''
    assert a.ndim == b.ndim == 1, ''
    assert a.shape[0] == b.shape[0] == 2, ''
    gradient_x = a[0] * 4 * x * (x**2 - b[0])
    gradient_y = a[1] * 4 * y * (y**2 - b[0])
    return gradient_x, gradient_y
