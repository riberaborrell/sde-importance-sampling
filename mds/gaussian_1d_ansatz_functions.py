from mds.plots_1d import Plot1d
from mds.utils import get_ansatz_data_path
from mds.validation import is_1d_valid_domain

import numpy as np
import scipy.stats as stats

class GaussianAnsatz:
    '''
    '''

    def __init__(self, domain, m=None, mus=None, sigmas=None):
        '''
        '''
        if domain is None:
            domain = np.array([-3, 3])
        is_1d_valid_domain(domain)
        if mus is not None and sigmas is not None:
            assert mus.ndim == sigmas.ndim == 1, ''
            assert mus.shape == sigmas.shape, ''
        self.domain = domain
        self.m = m
        self.mus = mus
        self.sigmas = sigmas
        self.dir_path = None

    def set_dir_path(self, example_dir_path):
        assert self.m is not None, ''
        assert self.sigmas is not None, ''

        m = self.m
        sigma = self.sigmas[0]
        self.dir_path = get_ansatz_data_path(example_dir_path, 'gaussian-ansatz', m, sigma)
        return self.dir_path

    def set_given_ansatz_functions(self, mus, sigmas):
        '''This method sets the number of ansatz functions, the mean and
           the standard deviation of the ansatz functions

        Args:
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert mus.shape == sigmas.shape

        self.m = mus.shape[0]
        self.mus = mus
        self.sigmas = sigmas

    def set_unif_dist_ansatz_functions(self, m=None, sigma=None):
        '''This method sets the mean and standard deviation of the m
           Gaussian ansatz functions. The means will be uniformly distributed
           in the domain and the standard deviation is given.

        Args:
            sigma (float) : standard deviation
        '''
        mus_min, mus_max = self.domain
        if m is None:
            m = self.m
        assert m >= 2, ''

        mus = np.around(np.linspace(mus_min, mus_max, m), decimals=2)
        if sigma is None:
            sigma = np.around(mus[1] - mus[0], decimals=2)

        self.m = m
        self.mus = mus
        self.sigmas = sigma * np.ones(m)

    # deprecated method
    def set_unif_dist_ansatz_functions_on_S(self, sigma, target_set):
        '''This method sets the mean and standard deviation of the m
           Gaussian ansatz functions. The means will be uniformly distributed
           in the domain and the standard deviation is given.

        Args:
            sigma (float) : standard deviation
        '''
        mus_min, mus_max = self.domain 
        # assume target_set is connected and contained in [mus_min, mus_max]
        target_set_min, target_set_max = target_set

        # set grid 
        h = 0.001
        N = int((mus_max - mus_min) / h) + 1
        X = np.around(np.linspace(mus_min, mus_max, N), decimals=3)

        # get indexes for nodes in/out the target set
        idx_ts = np.where((X >= target_set_min) & (X <= target_set_max))[0]
        idx_nts = np.where((X < target_set_min) | (X > target_set_max))[0]
        idx_l = np.where(X < target_set_min)[0]
        idx_r = np.where(X > target_set_max)[0]

        # compute ratio of nodes in the left/right side of the target set
        ratio_left = idx_l.shape[0] / idx_nts.shape[0]
        ratio_right = idx_r.shape[0] / idx_nts.shape[0]

        # assigm number of ansatz functions in each side
        m_left = int(np.round(m * ratio_left))
        m_right = int(np.round(m * ratio_right))
        assert m == m_left + m_right

        # distribute ansatz functions unif (in each side)
        mus_left = np.around(
            np.linspace(X[idx_l][0], X[idx_l][-1], m_left + 2)[:-2],
            decimals=3,
        )
        mus_right = np.around(
            np.linspace(X[idx_r][0], X[idx_r][-1], m_right + 2)[2:],
            decimals=3,
        )
        mus = np.concatenate((mus_left, mus_right), axis=0)

        # compute sigmas
        factor = 2
        sigma_left = factor * np.around(mus_left[1] - mus_left[0], decimals=3)
        sigma_right = factor * np.around(mus_right[1] - mus_right[0], decimals=3)
        sigmas_left = sigma_left * np.ones(m_left)
        sigmas_right = sigma_right * np.ones(m_right)
        sigmas = np.concatenate((sigmas_left, sigmas_right), axis=0)
        sigma_avg = np.around(np.mean(sigmas), decimals=3)

        print(m_left, m_right, m)
        print(mus_left[0], mus_left[-1])
        print(mus_right[0], mus_right[-1])
        print(sigma_left, sigma_right, sigma_avg)

        self.m = m
        self.mus = mus
        self.sigmas = sigmas

    #TODO deprecated. Use stats.norm.pdf(x, mu, sigma)
    def normal_pdf(self, x, mu=0, sigma=1):
        '''This method evaluates the normal probability density with mean
        mu and standard deviation sigma at the point x.

        Args:
            x (float or ndarray) : posision
            mu (float or ndarray): mean
            sigma (float or ndarray) : standard deviation
        '''
        norm_factor = np.sqrt(2 * np.pi) * sigma
        return np.exp(-0.5 * ((x - mu) / sigma) **2 ) / norm_factor

    def derivative_normal_pdf(self, x, mu=0, sigma=1):
        '''This method evaluates the derivative of the normal probability
           density function with mean mu and standard deviation sigma at
           the point x.

        Args:
            x (float or ndarray) : posision
            mu (float or ndarray): mean
            sigma (float or ndarray) : standard deviation
        '''
        return stats.norm.pdf(x, mu, sigma) * (mu - x) / sigma**2

    def basis_value_f(self, x):
        '''This method computes the ansatz functions for the value function evaluated at x

        Args:
            x (float or ndarray) : position/s
        '''
        mus = self.mus
        sigmas = self.sigmas

        if type(x) == np.ndarray:
            x = x.reshape(x.shape[0], 1)

        return stats.norm.pdf(x, mus, sigmas)

    def basis_control(self, x):
        '''This method computes the control basis functions evaluated at x

        Args:
            x (float or ndarray) : position/s
        '''
        mus = self.mus
        sigmas = self.sigmas

        if type(x) == np.ndarray:
            x = x.reshape(x.shape[0], 1)

        return - np.sqrt(2) * self.derivative_normal_pdf(x, mus, sigmas)

    def write_ansatz_parameters(self, f):
        '''
        '''
        f.write('Control parametrization (unif distr ansatz functions)\n')
        f.write('m: {:d}\n'.format(self.m))
        f.write('smallest mu: {:2.2f}\n'.format(np.min(self.mus)))
        f.write('biggest mu: {:2.2f}\n'.format(np.max(self.mus)))
        f.write('sigma: {:2.2f}\n\n'.format(self.sigmas[0]))

    def plot_gaussian_ansatz_functions(self):
        D_min, D_max = self.domain
        x = np.linspace(D_min, D_max, 1000)

        # basis value function
        v = self.basis_value_f(x)
        plt1d = Plot1d(dir_path=self.dir_path, file_name='basis_value_f')
        plt1d.set_title(r'$v_{j}(x; \mu, \sigma)$')
        plt1d.ansatz_value_f(x, v)

        # basis control
        b = self.basis_control(x)
        plt1d = Plot1d(dir_path=self.dir_path, file_name='basis_control')
        plt1d.set_title(r'$b_{j}(x; \mu, \sigma)$')
        plt1d.ansatz_control(x, b)


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
