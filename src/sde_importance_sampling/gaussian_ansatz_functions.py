from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.utils_path import make_dir_path, get_gaussian_ansatz_dir_path

import numpy as np

import os

class GaussianAnsatz():
    '''
    '''
    def __init__(self, n, beta, normalized=True):
        '''
        '''
        # dimension of the state space
        self.n = n

        # inverse of temperature
        self.beta = beta

        # number of ansatz, means and cov
        self.m = None
        self.means = None
        self.cov = None
        self.inv_cov = None
        self.det_cov = None
        self.sigma_i = None
        self.theta = None
        self.normalized = normalized

        # distributed
        self.distributed = None

        # uniform distributed along the axis
        self.m_i = None

        # meta distributed
        self.sigma_i_meta = None
        self.k = None
        self.N_meta = None
        self.seed_meta = None

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
            self.seed_meta,
        )

        # create dir path if not exists
        make_dir_path(dir_path)
        self.dir_path = dir_path

    def set_given_ansatz_functions(self, means, cov):
        '''
        '''
        # check means
        assert means.ndim == 2, ''
        assert means.shape[1] == self.n, ''
        self.m = means.shape[0]
        self.means = means

        # check covariance matrix
        assert cov.shape == (self.n, self.n), ''
        self.cov = cov

        # compute inverse
        self.inv_cov = np.linalg.inv(cov)

        # compute determinant
        self.det_cov = np.linalg.det(cov)

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
        self.means = np.moveaxis(self.means, 0, -1).reshape(self.m, self.n)

        self.sigma_i = sigma_i
        self.cov = np.eye(self.n)
        self.cov *= sigma_i

        self.distributed = 'uniform'

    def set_meta_dist_ansatz_functions(self, sde, dt_meta, sigma_i_meta, is_cumulative, k, N_meta):
        '''
        '''
        #TODO: update meta parameters
        meta = sde.get_metadynamics_sampling(dt_meta, sigma_i_meta, is_cumulative, k, N_meta)
        assert N_meta == meta.ms.shape[0], ''

        self.set_given_ansatz_functions(meta.means, meta.cov)
        self.sigma_i_meta = sigma_i_meta
        self.k = k
        self.N_meta = N_meta
        self.distributed = 'meta'

    def set_meta_ansatz_functions(self, sde, dt_meta, sigma_i_meta, k, N_meta):
        '''
        '''
        #TODO: update meta parameters
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

    def set_theta_random(self):
        bound = 0.1
        self.theta = np.random.uniform(-bound, bound, self.m)
        self.theta_type = 'random'

    def set_theta_hjb(self, sde, h):
        '''
        '''
        # get hjb solver
        hjb_sol = sde.get_hjb_solver(h)

        # flatten domain and value function
        Nh = hjb_sol.Nh
        x = hjb_sol.domain_h.reshape(Nh, self.n)
        value_f = hjb_sol.value_f.reshape(Nh,)

        # compute the optimal theta given a basis of ansatz functions
        v = self.basis_value_f(x)
        self.theta, _, _, _ = np.linalg.lstsq(v, value_f, rcond=None)
        self.h = h
        self.theta_type = 'optimal'

    def set_theta_metadynamics(self, meta, h):
        '''
        '''
        if self.distributed == 'meta':
            assert self.sigma_i_meta == meta.sigma_i, ''
            assert self.k == meta.k, ''
            assert self.N_meta == meta.N, ''

        # discretize domain
        meta.sample.discretize_domain(h)

        # flattened domain_h
        x = meta.sample.get_flat_domain_h()

        # set ansatz functions
        meta.set_ansatz()

        # meta value function evaluated at the grid
        meta.sample.ansatz.set_value_function_constant_corner()
        value_f_meta = meta.sample.ansatz.value_function(x)

        # ansatz functions evaluated at the grid
        v = self.basis_value_f(x)

        # solve theta v \theta = value_f_meta
        self.theta, _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)

        msg = 'uniformly distributed gaussian ansatz fitted (least squares) with metadynamics\n'
        print(msg)

        # save parameters
        self.theta_type = 'meta'
        self.sigma_i_meta = meta.sigma_i
        self.k = meta.k
        self.N_meta = meta.N
        self.seed_meta = meta.seed

    def mvn_pdf_basis(self, x):
        ''' Multivariate normal pdf (nd Gaussian) basis V(x; means, cov) with different m means
            but same covariance matrix evaluated at x
            x ((N, n)-array) : position

            returns (N, m)-array
        '''
        # assume shape of x array to be (N, n)
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''
        N = x.shape[0]

        # prepare position and means for broadcasting
        x = x[:, np.newaxis, :]
        means = self.means[np.newaxis, :, :]

        # compute exponential term
        x_centered = (x - means).reshape(N*self.m, self.n)
        exp_term = np.matmul(x_centered, self.inv_cov)
        exp_term = np.sum(exp_term * x_centered, axis=1).reshape(N, self.m)
        exp_term *= - 0.5

        # compute normalization factor if needed
        if self.normalized:

            # compute norm factor
            norm_factor = np.sqrt(((2 * np.pi) ** self.n) * self.det_cov)

            # normalize
            mvn_pdf_basis = np.exp(exp_term) / norm_factor
        else:
            mvn_pdf_basis = np.exp(exp_term)

        return mvn_pdf_basis

    def mvn_pdf_gradient_basis(self, x):
        ''' Multivariate normal pdf gradient (nd Gaussian gradients) basis \nabla V(x; means, cov)
        with different means but same covaianc matrix evaluated at x
            x ((N, n)-array) : posicion

            returns (N, m, n)-array
        '''
        # assume shape of x array to be (N, n)
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''
        N = x.shape[0]

        # get nd gaussian basis
        mvn_pdf_basis = self.mvn_pdf_basis(x)

        # prepare position and means for broadcasting
        x = x[:, np.newaxis, :]
        means = self.means[np.newaxis, :, :]

        # compute gradient of the exponential term
        exp_term_gradient = - 0.5 * np.matmul(x - means, self.inv_cov + self.inv_cov.T)

        # compute gaussian gradients basis
        mvn_pdf_gradient_basis = exp_term_gradient * mvn_pdf_basis[:, :, np.newaxis]

        return mvn_pdf_gradient_basis

    def basis_value_f(self, x):
        '''This method computes the ansatz functions for the value function evaluated at x

        Args:
            x ((N, n)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        basis_value_f = self.mvn_pdf_basis(x)
        return basis_value_f

    def basis_control(self, x):
        '''This method computes the control basis functions evaluated at x

        Args:
            x ((N, n)-ndarray) : positions
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        basis_control = - (np.sqrt(2) / self.beta) * self.mvn_pdf_gradient_basis(x)
        return basis_control

    def set_value_function_constant_to_zero(self):
        self.K_value_f = 0

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
        value_f_at_x = np.dot(basis_value_f_at_x, theta)

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

    def plot_1d_multivariate_normal_pdf(self, domain):
        from figures.myfigure import MyFigure
        import matplotlib.pyplot as plt

        assert self.n == 1, ''
        assert self.m == 1, ''

        h = 0.01
        grid_input = [slice(domain[0, 0], domain[0, 1] + h, h)]
        pos = np.mgrid[grid_input]
        pos = np.moveaxis(pos, 0, -1)
        x = pos[:, 0]
        Nx = x.shape[0]

        mvn_pdf = self.mvn_pdf_basis(pos)
        grad_mvn_pdf = - np.sqrt(2) * self.mvn_pdf_gradient_basis(pos)

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='gaussian-1d',
        )
        fig.plot(x, mvn_pdf[:, 0])

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='gaussian-grad-1d',
        )
        fig.plot(x, grad_mvn_pdf[:, 0, 0])

    def plot_2d_multivariate_normal_pdf(self, domain):
        from figures.myfigure import MyFigure
        import matplotlib.pyplot as plt

        assert self.n == 2, ''
        assert self.m == 1, ''

        h = 0.1
        grid_input = [
            slice(domain[0, 0], domain[0, 1] + h, h),
            slice(domain[1, 0], domain[1, 1] + h, h),
        ]
        xx, yy = np.ogrid[grid_input]
        X, Y = np.mgrid[grid_input]
        Nx = X.shape[0]
        Ny = Y.shape[1]
        pos = np.mgrid[grid_input]
        pos = np.moveaxis(pos, 0, -1).reshape(Nx * Ny, 2)

        mvn_pdf = self.mvn_pdf_basis(pos).reshape(Nx, Ny, self.m)
        grad_mvn_pdf = - np.sqrt(2) * self.mvn_pdf_gradient_basis(pos).reshape(Nx, Ny, self.m, 2)

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='gaussian-2d',
        )
        fig.contour(X, Y, mvn_pdf[:, :, 0])

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='gaussian-grad-2d',
        )
        fig.vector_field(X, Y, grad_mvn_pdf[:, :, 0, 0], grad_mvn_pdf[:, :, 0, 1], scale=1)

    def plot_nd_multivariate_normal_pdf(self, i=0, domain_i=[-3, 3], x_j=0.):
        from figures.myfigure import MyFigure
        import matplotlib.pyplot as plt

        assert self.m == 1, ''

        h = 0.1
        domain_i_h = np.arange(domain_i[0], domain_i[1] + h, h)
        Nh = domain_i_h.shape[0]

        x = x_j * np.ones((Nh, self.n))
        x[:, i] = domain_i_h

        mvn_pdf_i = self.mvn_pdf_basis(x)[:, 0]
        grad_mvn_pdf_i = - np.sqrt(2) * self.mvn_pdf_gradient_basis(x)[:, 0, i]

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='gaussian-nd-x{:d}'.format(i+1),
        )
        fig.plot(domain_i_h, mvn_pdf_i)

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='gaussian-grad-nd-x{:d}'.format(i+1),
        )
        fig.plot(domain_i_h, grad_mvn_pdf_i)
