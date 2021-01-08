from mds.utils import make_dir_path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import os

class GaussianAnsatz:
    '''
    '''

    def __init__(self, n, domain=None, m=None):
        '''
        '''
        if domain is None:
            domain = np.full((n, 2), [-3, 3])

        # dimension and domain
        self.n = n
        self.domain = domain

        # number of ansatz, means and cov
        self.m = m
        self.means = None
        self.cov = None

        # uniform distributed along the axis
        self.m_x = None
        self.sigma_x = None

        # directory path
        self.dir_path = None

    def set_dir_path(self, example_dir_path):
        '''
        '''
        assert self.m is not None, ''

        # get dir path
        dir_path = os.path.join(
            example_dir_path,
            'gaussian-ansatz',
            'm_{}'.format(self.m)
        )

        # create dir path if not exists
        make_dir_path(dir_path)
        self.dir_path = dir_path

    def set_given_ansatz_functions(self, means, cov):
        '''
        '''
        assert means.shape[1] == self.n, ''
        assert cov.shape == (self.n, self.n), ''
        self.m = means.shape[0]
        self.means = means
        self.cov = cov

    def set_unif_dist_ansatz_functions(self, m_x, sigma_x):
        '''
        '''
        self.m = m_x ** self.n

        mgrid_input = []
        for i in range(self.n):
            h_x = (self.domain[i, 1] - self.domain[i, 0]) / (m_x - 1)
            mgrid_input.append(
                slice(self.domain[i, 0], self.domain[i, 1] + h_x, h_x)
            )
        self.means = np.mgrid[mgrid_input]
        self.means = np.moveaxis(self.means, 0, -1).reshape((self.m, self.n))

        self.cov = np.eye(self.n)
        self.cov *= sigma_x

    def mv_normal_pdf(self, x, mean=None, cov=None):
        ''' Multivariate normal probability density function (nd Gaussian)
        v(x; mean, cov) evaluated at x
            x ((N, n)-array) : posicion
            mean ((n,)-array) : center of the gaussian
            cov ((n, n)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''
        if mean is None:
            mean = np.zeros(self.n)
        if cov is None:
            cov = np.eye(self.n)
        assert mean.shape == (self.n,), ''
        assert cov.shape == (self.n, self.n), ''

        N = x.shape[0]
        mvn_pdf = np.empty(N)

        rv = stats.multivariate_normal(mean, cov, allow_singular=False)
        mvn_pdf[:] = rv.pdf(x)

        return mvn_pdf

    def grad_mv_normal_pdf(self, x, mean=None, cov=None):
        ''' Gradient of the Multivariate normal probability density function (nd Gaussian)
        \nabla v(x; mean, cov) evaluated at x
            x ((N, n)-array) : posicion
            mean ((n,)-array) : center of the gaussian
            cov ((n, n)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''
        if mean is None:
            mean = np.zeros(self.n)
        if cov is None:
            cov = np.eye(self.n)
        assert mean.shape == (self.n,), ''
        assert cov.shape == (self.n, self.n), ''

        N = x.shape[0]
        mvn_pdf = np.empty(N)

        rv = stats.multivariate_normal(mean, cov, allow_singular=False)
        mvn_pdf[:] = rv.pdf(x)

        grad_exp_term = np.zeros((N, self.n))
        inv_cov = np.linalg.inv(cov)
        for i in range(self.n):
            for j in range(self.n):
                grad_exp_term[:, i] += (x[:, i] - mean[i]) * (inv_cov[i, j] + inv_cov[j, i])
        grad_exp_term *= - 0.5

        grad_mvn_pdf = grad_exp_term * mvn_pdf[:, np.newaxis]

        return grad_mvn_pdf

    def vec_mv_normal_pdf(self, x, means=None, cov=None):
        ''' Multivariate normal pdf (nd Gaussian) v(x; means, cov) with means evaluated at x
            x ((N, n)-array) : position
            means ((m, n)-array) : center of the gaussian
            cov ((n, n)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''
        if means is None:
            means = np.zeros(self.n)[np.newaxis, :]
        if cov is None:
            cov = np.eye(self.n)
        assert means.ndim == 2, ''
        assert means.shape[1] == self.n, ''
        assert cov.ndim == 2, ''
        assert cov.shape == (self.n, self.n), ''

        N = x.shape[0]
        m = means.shape[0]

        norm_factor = np.sqrt(((2 * np.pi) ** self.n) * np.sqrt(np.linalg.det(cov)))
        inv_cov = np.linalg.inv(cov)
        x = x[:, np.newaxis, :]
        means = means[np.newaxis, :, :]

        x_centered = (x - means).reshape(N*m, self.n)
        exp_term = np.matmul(x_centered, inv_cov)
        exp_term = np.sum(exp_term * x_centered, axis=1).reshape(N, m)
        exp_term *= - 0.5
        mvn_pdf = np.exp(exp_term) / norm_factor

        return mvn_pdf

    def vec_grad_mv_normal_pdf(self, x, means=None, cov=None):
        ''' Gradient of the multivariate normal pdf (nd Gaussian) \nabla v(x; means, cov)
        with means evaluated at x
            x ((N, n)-array) : posicion
            means ((m, n)-array) : center of the gaussian
            cov ((n, n)-array) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''
        if means is None:
            means = np.zeros(self.n)[np.newaxis, :]
        if cov is None:
            cov = np.eye(self.n)
        assert means.ndim == 2, ''
        assert means.shape[1] == self.n, ''
        assert cov.ndim == 2, ''
        assert cov.shape == (self.n, self.n), ''

        N = x.shape[0]
        m = means.shape[0]

        mvn_pdf = self.vec_mv_normal_pdf(x, means, cov)
        x = x[:, np.newaxis, :]
        means = means[np.newaxis, :, :]
        inv_cov = np.linalg.inv(cov)

        grad_mvn_pdf = np.empty((N, m, self.n))
        grad_exp_term = np.zeros((N, m, self.n))
        for i in range(self.n):
            for j in range(self.n):
                grad_exp_term[:, :, i] += (x[:, :, i] - means[:, :, i]) * (inv_cov[i, j] + inv_cov[j, i])
        grad_exp_term *= - 0.5

        grad_mvn_pdf = grad_exp_term * mvn_pdf[:, :, np.newaxis]

        return grad_mvn_pdf

    def basis_value_f(self, x):
        '''This method computes the ansatz functions for the value function evaluated at x

        Args:
            x ((N, n)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        #N = x.shape[0]
        #basis_value_f = np.zeros((N, self.m))
        #for j in np.arange(self.m):
        #    basis_value_f[:, j] = self.mv_normal_pdf(x, self.means[j], self.cov)
        basis_value_f = self.vec_mv_normal_pdf(x, self.means, self.cov)
        return basis_value_f

    def basis_control(self, x):
        '''This method computes the control basis functions evaluated at x

        Args:
            x ((N, n)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        #N = x.shape[0]
        #basis_control = np.zeros((N, self.m, self.n))
        #for j in np.arange(self.m):
        #    basis_control[:, j, :] = - np.sqrt(2) \
        #                           *  self.grad_mv_normal_pdf(x, self.means[j], self.cov)
        basis_control = - np.sqrt(2) * self.vec_grad_mv_normal_pdf(x, self.means, self.cov)
        return basis_control

    def write_ansatz_parameters(self, f):
        '''
        '''
        f.write('Value function parametrization: unif distr gaussian ansatz functions)\n')
        f.write('m: {:d}\n'.format(self.m))
        f.write('m_x: {:d}\n'.format(self.m_x))
        f.write('sigma_x: {:2.2f}\n'.format(self.sigma_x))

    def plot_1d_multivariate_normal_pdf(self, j):
        from mds.plots_1d import Plot1d
        assert self.n == 1, ''

        h = 0.01
        grid_input = [slice(self.domain[0, 0], self.domain[0, 1] + h, h)]
        pos = np.mgrid[grid_input]
        pos = np.moveaxis(pos, 0, -1)
        x = pos[:, 0]
        Nx = x.shape[0]

        mvn_pdf_j = self.mv_normal_pdf(pos, self.means[j], self.cov)
        grad_mvn_pdf_j = self.grad_mv_normal_pdf(pos, self.means[j], self.cov)
        mvn_pdf = self.vec_mv_normal_pdf(pos, self.means, self.cov)
        grad_mvn_pdf = - np.sqrt(2) * self.vec_grad_mv_normal_pdf(pos, self.means, self.cov)

        plt1d = Plot1d(self.dir_path, 'gaussian' + '_j{:d}'.format(j))
        #plt1d.one_line_plot(x, mvn_pdf_j)
        plt1d.one_line_plot(x, mvn_pdf[:, j])

        plt1d = Plot1d(self.dir_path, 'grad_gaussian' + '_j{:d}'.format(j))
        #plt1d.one_line_plot(x, grad_mvn_pdf_j)
        plt1d.one_line_plot(x, grad_mvn_pdf[:, j, 0])

    def plot_2d_multivariate_normal_pdf(self, j):
        from mds.plots_2d import Plot2d
        assert self.n == 2, ''

        h = 0.1
        grid_input = [
            slice(self.domain[0, 0], self.domain[0, 1] + h, h),
            slice(self.domain[1, 0], self.domain[1, 1] + h, h),
        ]
        xx, yy = np.ogrid[grid_input]
        X, Y = np.mgrid[grid_input]

        Nx = X.shape[0]
        Ny = Y.shape[1]
        pos = np.mgrid[grid_input]
        pos = np.moveaxis(pos, 0, -1).reshape(Nx * Ny, 2)

        mvn_pdf_j = self.mv_normal_pdf(pos, self.means[j], self.cov).reshape(Nx, Ny)
        grad_mvn_pdf_j = self.grad_mv_normal_pdf(pos, self.means[j], self.cov).reshape(Nx, Ny, 2)
        mvn_pdf = self.vec_mv_normal_pdf(pos, self.means, self.cov).reshape(Nx, Ny, self.m)
        grad_mvn_pdf = - np.sqrt(2) * self.vec_grad_mv_normal_pdf(pos, self.means, self.cov).reshape(Nx, Ny, self.m, 2)

        plt2d = Plot2d(self.dir_path, 'gaussian_surface' + '_j{:d}'.format(j))
        #plt2d.surface(xx, yy, mvn_pdf_j)
        plt2d.surface(xx, yy, mvn_pdf[:, :, j])

        plt2d = Plot2d(self.dir_path, 'gaussian_contour' + '_j{:d}'.format(j))
        #plt2d.contour(X, Y, mvn_pdf_j)
        plt2d.contour(X, Y, mvn_pdf[:, :, j])

        plt2d = Plot2d(self.dir_path, 'grad_gaussian' + '_j{:d}'.format(j))
        #plt2d.vector_field(X, Y, grad_mvn_pdf_j[:, :, 0], grad_mvn_pdf_j[:, :, 1], scale=1)
        plt2d.vector_field(X, Y, grad_mvn_pdf[:, :, j, 0], grad_mvn_pdf[:, :, j, 1], scale=1)
