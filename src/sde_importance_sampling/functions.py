import functools
import numpy as np
import scipy.stats as stats
import torch

FLOAT_TYPES = [float, np.float32, np.float64]

def constant(x, a):
    ''' constant function
    '''
    assert type(a) == float, ''

    # 1-dimensional function
    if type(x) in FLOAT_TYPES:
        return a

    # n-dimensional scalar function
    elif x.ndim == 1:
        return a

    # n-dimensional vector funcion
    elif x.ndim == 2:

        # set n points
        N = x.shape[0]

        if type(x) == np.array:
            return a * np.ones(N)
        elif type(x) == torch.Tensor:
            return a * torch.ones(N)


def quadratic_one_well(x, nu, tensor=False):
    ''' quadratic one well centered at 1.
    '''

    # 1-dimensional function
    if type(x) in FLOAT_TYPES and type(nu) in FLOAT_TYPES:
        return nu * (x - 1) ** 2
    elif type(x) in FLOAT_TYPES and type(nu) == np.ndarray:
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
            return torch.sum(nu_tensor * torch.pow((x -1), 2), axis=0)

    # n-dimensional vector funcion
    elif x.ndim == 2:
        assert x.shape[1] == n, ''
        N = x.shape[0]

        if not tensor:
            return np.sum(nu * (x -1)**2, axis=1)
        else:
            nu_tensor = torch.tensor(nu, requires_grad=False)
            return torch.sum(nu_tensor * torch.pow((x -1), 2), axis=1)


def double_well(x, alpha, tensor=False):
    ''' double well with minimus at +- 1 and maximum at 0.
    '''
    # 1-dimensional function
    if type(x) in FLOAT_TYPES and type(alpha) in FLOAT_TYPES:
        return alpha * (x**2 - 1) ** 2
    elif type(x) in FLOAT_TYPES and type(alpha) == np.ndarray:
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
    if type(x) in FLOAT_TYPES and type(alpha) in FLOAT_TYPES:
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
            alpha_tensor = torch.tensor(alpha, requires_grad=False, dtype=torch.float32)
            return 4 * alpha_tensor * x * (torch.pow(x, 2) - 1)

    # n-dimensional vector funcion
    elif x.ndim == 2:
        assert x.shape[1] == n, ''
        N = x.shape[0]

        if not tensor:
            return 4 * alpha * x * (x ** 2 - 1)
        else:
            alpha_tensor = torch.tensor(alpha, requires_grad=False, dtype=torch.float32)
            return 4 * alpha_tensor * x * ((torch.pow(x, 2) - 1))


def mvn_pdf(x, mean=None, cov=None, normalized=True):
    ''' Multivariate normal probability density function (nd Gaussian)
    v(x; mean, cov) evaluated at x
        x ((N, n)-array) : posicion
        mean ((n,)-array) : center of the gaussian
        cov ((n, n)-array) : covariance matrix
    '''
    # assume shape of x array to be (N, n)
    assert x.ndim == 2, ''
    N, n = x.shape

    # check center and covariance matrix
    if mean is None:
        mean = np.zeros(n)
    if cov is None:
        cov = np.eye(n)
    assert mean.shape == (n,), ''
    assert cov.shape == (n, n), ''

    # preallocate
    mvn_pdf = np.empty(N)

    # random variable with multivariate distribution
    rv = stats.multivariate_normal(mean, cov, allow_singular=False)

    # get corresponding probability density function
    if normalized:
        mvn_pdf[:] = rv.pdf(x)
    else:
        norm_factor = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(cov))
        mvn_pdf[:] = norm_factor * rv.pdf(x)

    return mvn_pdf


def mvn_pdf_gradient(x, mean=None, cov=None, normalized=True):
    ''' Gradient of the Multivariate normal probability density function (nd Gaussian)
    \nabla v(x; mean, cov) evaluated at x
        x ((N, n)-array) : posicion
        mean ((n,)-array) : center of the gaussian
        cov ((n, n)-array) : covariance matrix
    '''
    # assume shape of x array to be (N, n)
    assert x.ndim == 2, ''
    N, n = x.shape

    # check center and covariance matrix
    if mean is None:
        mean = np.zeros(n)
    if cov is None:
        cov = np.eye(n)
    assert mean.shape == (n,), ''
    assert cov.shape == (n, n), ''

    # preallocate
    mvn_pdf = np.empty(N)

    # random variable with multivariate distribution
    rv = stats.multivariate_normal(mean, cov, allow_singular=False)
    if normalized:
        mvn_pdf[:] = rv.pdf(x)
    else:
        norm_factor = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(cov))
        mvn_pdf[:] = norm_factor * rv.pdf(x)

    # covariance matrix inverse
    inv_cov = np.linalg.inv(cov)

    # gradient of the exponential term of the pdf
    grad_exp_term = - 0.5 * np.matmul(x - mean, inv_cov + inv_cov.T)

    grad_mvn_pdf = grad_exp_term * mvn_pdf[:, np.newaxis]
    return grad_mvn_pdf
