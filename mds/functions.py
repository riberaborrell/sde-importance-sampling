import functools
import numpy as np
import torch

def constant(x, a, tensor=False):
    ''' constant function
    '''
    assert type(a) == float, ''

    # 1-dimensional function
    if type(x) == float:
        return a

    # n-dimensional scalar function
    elif x.ndim == 1:
        return a

    # n-dimensional vector funcion
    elif x.ndim == 2:
        N = x.shape[0]
        if not tensor:
            return a * np.ones(N)
        else:
            return a * torch.ones(N)


def quadratic_one_well(x, nu, tensor=False):
    ''' quadratic one well centered at 1.
    '''

    # 1-dimensional function
    if type(x) == float and type(nu) == float:
        return nu * (x - 1) ** 2
    elif type(x) == float and type(nu) == np.ndarray:
        assert nu.ndim == 1 and nu.shape[0] == 1, ''
        return nu[0] * (x - 1) ** 2

    # check nu and x
    assert type(x) == np.ndarray or type(x) == torch.Tensor, ''
    assert type(nu) == np.ndarray and nu.ndim == 1, ''
    n = nu.shape[0]

    # n-dimensional scalar function
    if x.ndim == 1:
        assert x.shape[0] == n, ''
        if not tensor:
            return np.sum(nu * (x -1)**2)
        else:
            nu_tensor = torch.tensor(nu, requires_grad=False)
            return torch.sum(nu_tensor * torch.float_power((x -1), 2), axis=0)

    # n-dimensional vector funcion
    elif x.ndim == 2:
        assert x.shape[1] == n, ''
        N = x.shape[0]

        if not tensor:
            return np.sum(nu * (x -1)**2, axis=1)
        else:
            nu_tensor = torch.tensor(nu, requires_grad=False)
            return torch.sum(nu_tensor * torch.float_power((x -1), 2), axis=1)


def double_well(x, alpha, tensor=False):
    '''
    '''
    # 1-dimensional function
    if type(x) == np.float64 and type(alpha) == np.float64:
        return alpha * (x**2 - 1) ** 2
    elif type(x) == np.float64 and type(alpha) == np.ndarray:
        assert alpha.ndim == 1 and alpha.shape[0] == 1, ''
        return alpha[0] * (x**2 - 1) ** 2

    # check alpha and x
    assert type(x) == np.ndarray or type(x) == torch.Tensor, ''
    assert type(alpha) == np.ndarray and alpha.ndim == 1, ''
    n = alpha.shape[0]

    # n-dimensional scalar function
    if x.ndim == 1:
        assert x.shape[0] == n, ''
        if not tensor:
            return np.sum(alpha * (x**2 - 1) ** 2)
        else:
            alpha_tensor = torch.tensor(alpha, requires_grad=False)
            return torch.sum(
                alpha_tensor * torch.float_power((torch.float_power(x, 2) - 1), 2),
                axis=0,
            )
    # n-dimensional vector funcion
    elif x.ndim == 2:
        assert x.shape[1] == n, ''
        N = x.shape[0]

        if not tensor:
            return np.sum(alpha * (x ** 2 -1) **2, axis=1)
        else:
            alpha_tensor = torch.tensor(alpha, requires_grad=False)
            return torch.sum(
                alpha_tensor * torch.float_power((torch.float_power(x, 2) -1), 2),
                axis=1,
            )

def double_well_gradient(x, alpha, tensor=False):
    '''
    '''
    # 1-dimensional function
    if type(x) == float and type(alpha) == float:
        return 4 * alpha * x * (x**2 - 1)
    elif type(x) == float and type(alpha) == np.ndarray:
        assert alpha.ndim == 1 and alpha.shape[0] == 1, ''
        return 4 * alpha[0] * x * (x**2 - 1)

    # check alpha and x
    assert type(x) == np.ndarray or type(x) == torch.Tensor, ''
    assert type(alpha) == np.ndarray and alpha.ndim == 1, ''
    n = alpha.shape[0]

    # n-dimensional vector function
    if x.ndim == 1:
        assert x.shape[0] == n, ''
        if not tensor:
            return 4 * alpha * x * (x**2 - 1)
        else:
            alpha_tensor = torch.tensor(alpha, requires_grad=False)
            return 4 * alpha_tensor * x * (torch.float_power(x, 2) - 1)

    # n-dimensional vector funcion
    elif x.ndim == 2:
        assert x.shape[1] == n, ''
        N = x.shape[0]

        if not tensor:
            return 4 * alpha * x * (x ** 2 - 1)
        else:
            alpha_tensor = torch.tensor(alpha, requires_grad=False)
            return torch.sum(
                alpha_tensor * torch.float_power((torch.float_power(x, 2) -1), 2),
                axis=1,
            )
