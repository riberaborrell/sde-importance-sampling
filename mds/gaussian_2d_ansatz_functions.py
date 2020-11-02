from mds.plots_2d import Plot2d
from mds.utils import get_ansatz_data_path
#from mds.validation import is_1d_valid_domain

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class GaussianAnsatz:
    '''
    '''

    def __init__(self, domain=None, m_x=None, m_y=None):
        '''
        '''
        if domain is None:
            domain = np.array([[-3, 3], [-3, 3]])
        #TODO: validate 2d domain
        #is_1d_valid_domain(domain)
        self.domain = domain
        self.m_x = m_x
        self.m_y = m_y
        self.m = None
        self.sigma_x = None
        self.sigma_y = None
        self.means = None
        self.cov = None
        self.dir_path = None

    def set_dir_path(self, example_dir_path):
        assert self.m is not None, ''
        assert self.sigma_x is not None, ''
        assert self.sigma_y is not None, ''
        assert self.sigma_x == self.sigma_y, ''

        m = self.m
        sigma_x = self.sigma_x
        self.dir_path = get_ansatz_data_path(example_dir_path, 'gaussian-ansatz', m, sigma_x)
        return self.dir_path

    def set_unif_dist_ansatz_functions(self, m_x=None, m_y=None, sigma_x=None, sigma_y=None):
        '''
        '''
        if m_x is None:
            m_x = self.m_x
        if m_y is None:
            m_y = self.m_y
        m = m_x * m_y

        d_xmin, d_xmax = self.domain[0]
        d_ymin, d_ymax = self.domain[1]

        means_x = np.around(np.linspace(d_xmin, d_xmax, m_x), decimals=2)
        means_y = np.around(np.linspace(d_ymin, d_ymax, m_y), decimals=2)
        means_X, means_Y = np.meshgrid(means_x, means_y, sparse=False, indexing='ij')
        means = np.dstack((means_X, means_Y)).reshape((m, 2))

        if sigma_x is None:
            sigma_x = np.around(means_x[1] - means_x[0], decimals=2)
        if sigma_y is None:
            sigma_y = np.around(means_y[1] - means_y[0], decimals=2)
        cov = np.eye(2)
        cov[0, 0] *= sigma_x
        cov[1, 1] *= sigma_y

        self.m = m
        self.m_x = m_x
        self.m_y = m_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.means = means
        self.cov = cov

    def multivariate_normal_pdf(self, x, mean=None, cov=None):
        ''' 2d Gaussian v(x; mean, cov) evaluated at x
            x ((M, 2)-array) : posicion
            mean ((2,)-array) : center of the gaussian
            cov ((2, 2)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == 2, ''
        if mean is None:
            mean = np.zeros(2)
        if cov is None:
            cov = np.eye(2)
        assert mean.shape == (2,), ''
        assert cov.shape == (2, 2), ''

        rv = stats.multivariate_normal(mean, cov, allow_singular=False)
        return rv.pdf(x)

    def vectorized_multivariate_normal_pdf(self, x, mean=None, cov=None):
        ''' 2d Gaussian v(x; mean, cov) evaluated at x
            x ((M, 2)-array) : position
            mean ((m, 2)-array) : center of the gaussian
            cov ((2, 2)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == 2, ''
        if mean is None:
            mean = np.zeros(2)[None, :]
        if cov is None:
            cov = np.eye(2)
        assert mean.ndim == 2, ''
        assert cov.ndim == 2, ''
        assert mean.shape[1] == 2, ''
        assert cov.shape[0] == cov.shape[1] == 2, ''
        M = x.shape[0]
        m = mean.shape[0]
        norm_factor = 2 * np.pi * np.sqrt(np.linalg.det(cov))
        x = x[:, None, :]
        mean = mean[None, :, :]
        x_centered = (x - mean).reshape((M*m, 2))
        exp_term = - 0.5 * np.matmul(x_centered, cov)
        exp_term = np.sum(exp_term * x_centered, axis=1).reshape((M, m))
        pdf = np.exp(exp_term) / norm_factor
        return pdf

    def gradient_multivariate_normal_pdf(self, x, mean=None, cov=None):
        ''' 2d Gaussian v(x; mean, cov) evaluated at x
            x ((M, 2)-array) : posicion
            mean ((2,)-array) : center of the gaussian
            cov ((2, 2)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == 2, ''
        if mean is None:
            mean = np.zeros(2)
        if cov is None:
            cov = np.eye(2)
        assert mean.shape == (2,), ''
        assert cov.shape == (2, 2), ''

        M = x.shape[0]
        rv = stats.multivariate_normal(mean, cov, allow_singular=False)
        inv_cov = np.linalg.inv(cov)
        exp_grad_x = - (
            + 2 * (x[:, 0] - mean[0]) * inv_cov[0, 0]
            + (x[:, 1] - mean[1]) * (inv_cov[1, 0] + inv_cov[0, 1])
        ) / 2
        exp_grad_y = - (
            + (x[:, 0] - mean[0]) * (inv_cov[1, 0] + inv_cov[0, 1])
            + 2 * (x[:, 1] - mean[1]) * inv_cov[0, 0]
        ) / 2
        grad_x = (rv.pdf(x) * exp_grad_x).reshape((M, 1))
        grad_y = (rv.pdf(x) * exp_grad_y).reshape((M, 1))
        grad = np.hstack((grad_x, grad_y))
        return grad

    def vectorized_gradient_multivariate_normal_pdf(self, x, mean=None, cov=None):
        ''' 2d Gaussian v(x; mean, cov) evaluated at x
            x ((M, 2)-array) : posicion
            mean ((m, 2)-array) : center of the gaussian
            cov ((2, 2)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == 2, ''
        if mean is None:
            mean = np.zeros(2)[None, :]
        if cov is None:
            cov = np.eye(2)
        assert mean.ndim == 2, ''
        assert cov.ndim == 2, ''
        assert mean.shape[1] == 2, ''
        assert cov.shape[0] == cov.shape[1] == 2, ''
        M = x.shape[0]
        m = mean.shape[0]

        pdf = self.vectorized_multivariate_normal_pdf(x, mean, cov)
        x = x[:, None, :]
        mean = mean[None, :, :]
        inv_cov = np.linalg.inv(cov)

        exp_grad_x = - (
            + 2 * (x[:, :, 0] - mean[:, :, 0]) * inv_cov[0, 0]
            + (x[:, :, 1] - mean[:, :, 1]) * (inv_cov[1, 0] + inv_cov[0, 1])
        ) / 2
        exp_grad_y = - (
            + (x[:, :, 0] - mean[:, :, 0]) * (inv_cov[1, 0] + inv_cov[0, 1])
            + 2 * (x[:, :, 1] - mean[:, :, 1]) * inv_cov[0, 0]
        ) / 2
        grad_x = (pdf * exp_grad_x)
        grad_y = (pdf * exp_grad_y)

        grad = np.empty((M, m, 2))
        grad[:, :, 0] = grad_x
        grad[:, :, 1] = grad_y
        return grad

    def basis_value_f(self, x):
        '''This method computes the ansatz functions for the value function evaluated at x

        Args:
            x ((M, 2)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == 2, ''

        M = x.shape[0]
        m = self.m
        means = self.means
        cov = self.cov

        #basis_value_f = np.zeros((M, m))
        #for j in np.arange(m):
        #    basis_value_f[:, j] = self.multivariate_normal_pdf(x, means[j], covs[j])
        basis_value_f = self.vectorized_multivariate_normal_pdf(x, means, cov)
        return basis_value_f

    def basis_control(self, x):
        '''This method computes the control basis functions evaluated at x

        Args:
            x ((M, 2)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == 2, ''

        M = x.shape[0]
        m = self.m
        means = self.means
        cov = self.cov

        #basis_control = np.zeros((M, m, 2))
        #for j in np.arange(m):
        #    basis_control[:, j, :] = - np.sqrt(2) *  self.gradient_multivariate_normal_pdf(x, means[j], covs[j])
        basis_control = - np.sqrt(2) * self.vectorized_gradient_multivariate_normal_pdf(x, means, cov)
        return basis_control

    def write_ansatz_parameters(self, f):
        '''
        '''
        f.write('Value function parametrization: unif distr gaussian ansatz functions)\n')
        f.write('m_x: {:d}\n'.format(self.m_x))
        f.write('m_y: {:d}\n'.format(self.m_y))
        f.write('m: {:d}\n'.format(self.m))
        f.write('sigma_x: {:2.2f}\n'.format(self.sigma_x))
        f.write('sigma_y: {:2.2f}\n\n'.format(self.sigma_y))

    def plot_multivariate_normal_pdf(self, j):
        means = self.means
        cov = self.cov

        d_xmin, d_xmax = self.domain[0]
        d_ymin, d_ymax = self.domain[1]
        h = 0.2
        x = np.arange(d_xmin, d_xmax + h, h)
        y = np.arange(d_ymin, d_ymax + h, h)
        Nx = x.shape[0]
        Ny = y.shape[0]
        xx, yy = np.meshgrid(x, y, sparse=True, indexing='ij')
        X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')
        pos = np.dstack((X, Y)).reshape((Nx * Ny, 2))

        #Z = self.multivariate_normal_pdf(pos, means[j], cov).reshape((Nx, Ny))
        Z = self.vectorized_multivariate_normal_pdf(pos, means, cov).reshape((Nx, Ny, self.m))
        plt2d = Plot2d(self.dir_path, 'gaussian_surface')
        plt2d.surface(xx, yy, Z[:, :, j])

        plt2d = Plot2d(self.dir_path, 'gaussian_contour')
        plt2d.contour(X, Y, Z[:, :, j])

        #grad = self.gradient_multivariate_normal_pdf(pos, means[j], cov).reshape((Nx, Ny, 2))
        grad = self.vectorized_gradient_multivariate_normal_pdf(pos, means, cov).reshape((Nx, Ny, self.m, 2))
        U = grad[:, :, j, 0]
        V = grad[:, :, j, 1]
        plt2d = Plot2d(self.dir_path, 'grad_gaussian')
        plt2d.vector_field(X, Y, U, V)

    def plot_gaussian_ansatz_functions(self, omega=None):
        m = self.m
        m_x = self.m_x
        m_y = self.m_y

        d_xmin, d_xmax = self.domain[0]
        d_ymin, d_ymax = self.domain[1]
        h = 0.2
        x = np.arange(d_xmin, d_xmax + h, h)
        y = np.arange(d_ymin, d_ymax + h, h)
        Nx = x.shape[0]
        Ny = y.shape[0]
        xx, yy = np.meshgrid(x, y, sparse=True, indexing='ij')
        X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')
        pos = np.dstack((X, Y)).reshape((Nx * Ny, 2))

        basis_value_f = self.basis_value_f(pos)
        basis_control = self.basis_control(pos)
        theta = np.ones(m)
        value_f = np.dot(basis_value_f, theta)
        value_f = value_f.reshape((Nx, Ny))
        control = np.empty((Nx * Ny, 2))
        control[:, 0] = np.dot(basis_control[:, :, 0], theta)
        control[:, 1] = np.dot(basis_control[:, :, 1], theta)
        control = control.reshape((Nx, Ny, 2))
        plt2d = Plot2d(self.dir_path, 'basis_value_f_surface')
        plt2d.surface(xx, yy, value_f)

        plt2d = Plot2d(self.dir_path, 'basis_value_f_contour')
        plt2d.contour(X, Y, value_f)

        U = control[:, :, 0]
        V = control[:, :, 1]
        plt2d = Plot2d(self.dir_path, 'basis_control')
        plt2d.vector_field(X, Y, U, V)
