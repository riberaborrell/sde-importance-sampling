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
    if type(x) == float:
        assert type(nu) == float, ''
        return nu * (x - 1) ** 2

    # n-dimensional scalar function
    elif x.ndim == 1:
        assert type(nu) == np.ndarray and nu.ndim == 1, ''
        if not tensor:
            return np.sum(nu * (x -1)**2)
        else:
            nu_tensor = torch.tensor(nu, requires_grad=False)
            return torch.sum(nu_tensor * torch.power((x -1), 2), axis=0)

    # n-dimensional vector funcion
    elif x.ndim == 2:
        assert type(nu) == np.ndarray and nu.ndim == 1, ''
        N = x.shape[0]
        n = x.shape[1]
        assert nu.shape[0] == n, ''

        if not tensor:
            return np.sum(nu * (x -1)**2, axis=1)
        else:
            pass
        #f= np.zeros(N)
        #for i in range(n):
        #    f += alpha[i] * np.power(np.power(x[:, i], 2) - 1, 2)
