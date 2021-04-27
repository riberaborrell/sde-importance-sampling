from mds.langevin_nd_sde import LangevinSDE
from mds.utils import make_dir_path, get_gaussian_ansatz_dir_path

import numpy as np
import scipy.stats as stats

import os

class GaussianAnsatz():
    '''
    '''
    def __init__(self, n):
        '''
        '''
        # dimension of the state space
        self.n = n

        # number of ansatz, means and cov
        self.m = None
        self.means = None
        self.cov = None
        self.sigma_i = None
        self.theta = None

        # distributed
        self.distributed = None

        # uniform distributed along the axis
        self.m_i = None

        # meta distributed
        self.sigma_i_meta = None
        self.k = None
        self.N_meta = None

        self.theta_type = None

        # value function constanct
        self.K_value_f = None

        # directory path
        self.dir_path = None

    def set_dir_path(self, sde):
        '''
        '''
        # get dir path
        dir_path = get_gaussian_ansatz_dir_path(
            sde.settings_dir_path,
            self.distributed,
            self.theta_type,
            self.m_i,
            self.sigma_i,
            self.sigma_i_meta,
            self.k,
            self.N_meta,
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

    def set_unif_dist_ansatz_functions(self, sde, m_i, sigma_i):
        '''
        '''
        self.m_i = m_i
        self.m = m_i ** self.n

        mgrid_input = []
        for i in range(self.n):
            slice_i = slice(sde.domain[i, 0], sde.domain[i, 1], complex(0, m_i))
            mgrid_input.append(slice_i)
        self.means = np.mgrid[mgrid_input]
        self.means = np.moveaxis(self.means, 0, -1).reshape((self.m, self.n))

        self.sigma_i = sigma_i
        self.cov = np.eye(self.n)
        self.cov *= sigma_i

        self.distributed = 'uniform'

    def set_meta_dist_ansatz_functions(self, sde, dt_meta, sigma_i_meta, k, N_meta):
        '''
        '''
        meta = sde.get_metadynamics_sampling(dt_meta, sigma_i_meta, k, N_meta)
        assert N_meta == meta.ms.shape[0], ''

        self.set_given_ansatz_functions(meta.means, meta.cov)
        self.sigma_i_meta = sigma_i_meta
        self.k = k
        self.N_meta = N_meta
        self.distributed = 'meta'

    def set_meta_ansatz_functions(self, sde, dt_meta, sigma_i_meta, k, N_meta):
        '''
        '''
        meta = sde.get_metadynamics_sampling(dt_meta, sigma_i_meta, k, N_meta)
        meta_total_m = int(np.sum(meta.ms))
        assert N_meta == meta.ms.shape[0], ''

        # get the centers used for each trajectory
        means = np.empty((meta_total_m, self.n))
        theta = np.empty(meta_total_m)
        flatten_idx = 0
        for i in np.arange(N_meta):
            means[flatten_idx:flatten_idx+meta.ms[i]] = meta.means[i, :meta.ms[i]]
            theta[flatten_idx:flatten_idx+meta.ms[i]] = meta.thetas[i, :meta.ms[i]]
            flatten_idx += meta.ms[i]

        self.set_given_ansatz_functions(means, meta.cov)
        self.sigma_i_meta = sigma_i_meta
        self.k = k
        self.N_meta = N_meta
        self.theta = theta / N_meta
        self.distributed = 'meta'
        self.theta_type = 'meta'

    def set_theta_null(self):
        self.theta = np.zeros(self.m)
        self.theta_type = 'null'

    def set_theta_from_hjb_solution(self, sde, h):
        '''
        '''
        # get hjb solver
        hjb_sol = sde.get_hjb_solver(h)

        # flatten domain and free energy
        Nh = hjb_sol.Nh
        x = hjb_sol.domain_h.reshape(Nh, self.n)
        F = hjb_sol.F.reshape(Nh,)

        # compute the optimal theta given a basis of ansatz functions
        v = self.basis_value_f(x)
        self.theta, _, _, _ = np.linalg.lstsq(v, F, rcond=None)
        self.h = h
        self.theta_type = 'optimal'

    def set_theta_from_metadynamics(self, sde, dt_meta, sigma_i_meta, k, N_meta):
        '''
        '''
        if self.distributed == 'meta':
            assert self.sigma_i_meta ==sigma_i_meta, ''
            assert self.k == k, ''
            assert self.N_meta == N_meta, ''

        # discretize domain
        sde.discretize_domain()

        # flatten domain_h
        x = sde.domain_h.reshape(sde.Nh, self.n)

        meta = sde.get_metadynamics_sampling(dt_meta, sigma_i_meta, k, N_meta)
        assert meta.ms.shape[0] == N_meta, ''

        thetas = np.empty((N_meta, self.m))

        for i in np.arange(N_meta):
            # get means and thetas for each trajectory
            idx_i = slice(np.sum(meta.ms[:i]), np.sum(meta.ms[:i]) + meta.ms[i])
            meta_means_i = meta.means[idx_i]
            meta_thetas_i = meta.thetas[idx_i]

            # create ansatz functions from meta
            meta_ansatz = GaussianAnsatz(n=self.n)
            meta_ansatz.set_given_ansatz_functions(
                means=meta_means_i,
                cov=meta.cov,
            )

            # meta value function evaluated at the grid
            meta_ansatz.set_value_function_constant_corner(meta_thetas_i)
            value_f_meta = meta_ansatz.value_function(x, meta_thetas_i)

            # ansatz functions evaluated at the grid
            v = self.basis_value_f(x)

            # solve theta V = \Phi
            thetas[i], _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)

        self.theta = np.mean(thetas, axis=0)
        self.theta_type = 'meta'
        self.sigma_i_meta = sigma_i_meta
        self.k = k
        self.N_meta = N_meta

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

        norm_factor = np.sqrt(((2 * np.pi) ** self.n) * np.linalg.det(cov))
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

        basis_value_f = self.vec_mv_normal_pdf(x, self.means, self.cov)
        return basis_value_f

    def basis_control(self, x):
        '''This method computes the control basis functions evaluated at x

        Args:
            x ((N, n)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        basis_control = - np.sqrt(2) * self.vec_grad_mv_normal_pdf(x, self.means, self.cov)
        return basis_control

    def set_value_function_constant_target_set(self):

        # discretize domain
        self.discretize_domain()

        # get idx in the target set
        idx_ts = self.get_idx_target_set()

        # impose value function in the target set to be null
        self.K_value_f = - np.mean(value_f[idx_ts])

    def set_value_function_constant_boarder(self):
        pass
        #self.K_value_f = 

    def set_value_function_constant_corner(self, theta=None):
        if theta is None:
            theta = self.theta

        # define target set corner (1, ..., 1)
        x = np.ones((1, self.n))

        # evaluate value function at x
        basis_value_f_at_x = self.basis_value_f(x)
        value_f_at_x =  np.dot(basis_value_f_at_x, theta)

        self.K_value_f = - value_f_at_x

    def value_function(self, x, theta=None):
        '''This method computes the value function evaluated at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters

        Return:
        '''
        if theta is None:
            theta = self.theta

        # value function without constant K
        basis_value_f = self.basis_value_f(x)
        value_f =  np.dot(basis_value_f, theta)

        return value_f + self.K_value_f



    def control(self, x, theta=None):
        '''This method computes the control evaluated at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters
        '''
        N = x.shape[0]
        if theta is None:
            theta = self.theta

        basis_control = self.basis_control(x)
        control = np.tensordot(basis_control, theta, axes=([1], [0]))
        return control


    def write_ansatz_parameters(self, f):
        '''
        '''
        f.write('Value function parametrized by Gaussian ansatz functions\n')
        f.write('distributed: {}\n'.format(self.distributed))

        if self.distributed == 'uniform':
            f.write('m_i: {:d}\n'.format(self.m_i))
            f.write('sigma_i: {:2.2f}\n'.format(self.sigma_i))

        elif self.distributed == 'meta':
            f.write('sigma_i_meta: {:2.2f}\n'.format(self.sigma_i_meta))
            f.write('k: {:d}\n'.format(self.k))
            f.write('N_meta: {:2.2f}\n'.format(self.N_meta))

        f.write('m: {:d}\n\n'.format(self.m))

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
