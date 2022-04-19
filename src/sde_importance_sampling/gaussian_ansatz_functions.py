import os

import numpy as np
import torch

from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.utils_path import make_dir_path, get_gaussian_ansatz_dir_path

class GaussianAnsatz(object):
    ''' Gaussian ansatz

    Attributes
    ----------
    sde: object
        LangevinSDE object
    m: int
        number of ansatz functions
    means: array
        centers of the Gaussian functions
    means_tensor: tensor
        centers of the Gaussian functions
    cov: array
        covariance matrix
    inv_cov: array
        inverse of the covariance matrix
    det_cov: float
        determinant of the covariance matrix
    sigma_i: float
        factor of the scalar covariance matrix
    theta: array
        parameters of the Gaussian ansatz representation
    normalized: bool
        True if the Gaussian functions are normalized normal density functions
    is_cov_scalar_matrix: bool
        True if the covariance matrix is a scalar matrix
    distributed: str
        states how the centers of the Gaussian functions are distributed. "uniform" means
        that they are placed uniformly along the axis of the space. "meta" means that they are
        placed in the same place where the metadynamics algorithm did an update
    m_i: int
        number of ansatz functions placed in each coordinate
    sigma_i_meta: float
        factor of the scalar covariance matrix for the metadynamics algorithm
    delta_meta: float
        time interval between metadynamics updates
    K_meta: int
        number of trajectories used in the metadynamics algorithm
    seed_meta: int
        random seed used in the metadynamics algorithm
    theta_type: str
        ?

    K_value_f: float
        value function constanct

    dir_path: str
        directory path

    Methods
    -------
    __init__(sde, normalized=True)

    set_dir_path()

    set_cov_matrix(sigma_i=None, cov=None)

    set_given_ansatz_functions(means, sigma_i=None, cov=None)

    set_unif_dist_ansatz_functions(m_i, sigma_i)

    set_meta_ansatz_functions(sde, dt_meta, sigma_i_meta, k, K_meta)

    set_theta_null()

    set_theta_random()

    set_theta_hjb(h)

    set_theta_metadynamics(meta, h)

    mvn_pdf_basis_numpy(x)

    mvn_pdf_basis(x)

    mvn_pdf_gradient_basis(x)

    set_value_function_constant_to_zero()

    set_value_function_constant_target_set()

    set_value_function_constant_corner(theta=None)

    value_function(x, theta=None)

    control(x, theta=None)

    write_ansatz_parameters(f)

    plot_1d_multivariate_normal_pdf(domain)

    plot_2d_multivariate_normal_pdf(domain)

    plot_nd_multivariate_normal_pdf(i=0, domain_i=[-3, 3], x_j=0.)

    '''

    def __init__(self, sde, normalized=True):
        ''' init method

        Parameters
        ----------
        sde: object
            LangevinSDE object
        normalized: bool
            True if the Gaussian functions are normalized normal density functions
        '''

        # environment
        self.sde = sde

        # normalize flag
        self.normalized = normalized

        # metadynamics attributes
        self.sigma_i_meta = None
        self.delta_meta = None
        self.K_meta = None
        self.seed_meta = None


    def set_dir_path(self):
        ''' set directory path for the chosen gaussian ansatz representation
        '''
        # get dir path
        dir_path = get_gaussian_ansatz_dir_path(
            self.sde.settings_dir_path,
            self.distributed,
            self.theta_type,
            self.m_i,
            self.sigma_i,
            self.sigma_i_meta,
            self.delta_meta,
            self.K_meta,
            self.seed_meta,
        )

        # create dir path if not exists
        make_dir_path(dir_path)
        self.dir_path = dir_path

    def set_cov_matrix(self, sigma_i=None, cov=None):
        ''' sets the covariance matrix of the gaussian functions

        Parameters
        ----------
        sigma_i: float, optional
            value in the diagonal entries of the covariance matrix
        cov: (d, d)-array, optional
            covariance matrix
        '''

        # scalar covariance matrix case
        if sigma_i is not None:

            # covariance matrix
            self.sigma_i = sigma_i
            self.cov = sigma_i * np.eye(self.sde.d)
            self.is_cov_scalar_matrix = True

            # compute inverse
            self.inv_cov = np.eye(self.sde.d) / sigma_i

            # compute determinant
            self.det_cov = sigma_i**self.sde.d

        # general case
        if cov is not None:

            # check covariance matrix
            assert cov.shape == (self.sde.d, self.sde.d), ''

            # set covariance matrix
            self.cov = cov
            self.is_cov_scalar_matrix = False

            # compute inverse
            self.inv_cov = np.linalg.inv(cov)

            # compute determinant
            self.det_cov = np.linalg.det(cov)

    def set_given_ansatz_functions(self, means, sigma_i=None, cov=None):
        ''' sets means and covariance matrix for gaussian ansatz

        Parameters
        ----------
        means: (m, d)-array
            centers of the gaussian functions
        sigma_i: float
            value in the diagonal entries of the covariance matrix
        cov: (d, d)-array
            covariance matrix
        '''
        # check means
        assert means.ndim == 2, ''
        assert means.shape[1] == self.sde.d, ''

        # set means
        self.m = means.shape[0]
        self.means = means
        self.means_tensor = torch.tensor(means)

        # set covariance matrix
        self.set_cov_matrix(sigma_i, cov)


    def set_unif_dist_ansatz_functions(self, m_i, sigma_i):
        ''' sets gaussian ansatz uniformly distributed in the domain with scalar covariance
            matrix.

        Parameters
        ----------
        m_i: float
            number of gaussian ansatz along each direction
        sigma_i: float
            value in the diagonal entries of the covariance matrix
        '''

        # set number of gaussians
        self.m_i = m_i
        self.m = m_i ** self.sde.d

        # distribute centers of Gaussians uniformly
        mgrid_input = []
        for i in range(self.sde.d):
            slice_i = slice(self.sde.domain[i, 0], self.sde.domain[i, 1], complex(0, m_i))
            mgrid_input.append(slice_i)
        self.means = np.mgrid[mgrid_input]
        self.means = np.moveaxis(self.means, 0, -1).reshape(self.m, self.sde.d)
        self.means_tensor = torch.tensor(self.means)
        self.distributed = 'uniform'

        # set covariance matrix
        self.set_cov_matrix(sigma_i=sigma_i)


    #TODO: check method
    def set_meta_dist_ansatz_functions(self, sde, dt_meta, sigma_i_meta, is_cumulative, k, K_meta):
        '''
        '''
        meta = sde.get_metadynamics_sampling(dt_meta, sigma_i_meta, is_cumulative, k, K_meta)
        assert K_meta == meta.ms.shape[0], ''

        self.set_given_ansatz_functions(means=meta.means, sigma_i=meta.sigma_i)
        self.sigma_i_meta = sigma_i_meta
        self.k = k
        self.K_meta = K_meta
        self.distributed = 'meta'

    #TODO: check method
    def set_meta_ansatz_functions(self, sde, dt_meta, sigma_i_meta, k, K_meta):
        '''
        '''
        meta = sde.get_metadynamics_sampling(dt_meta, sigma_i_meta, k, K_meta)
        meta_total_m = int(np.sum(meta.ms))
        assert K_meta == meta.ms.shape[0], ''

        # get the centers used for each trajectory
        means = np.empty((meta_total_m, self.n))
        theta = np.empty(meta_total_m)
        flatten_idx = 0
        for i in np.arange(K_meta):
            means[flatten_idx:flatten_idx+meta.ms[i]] = meta.means[i, :meta.ms[i]]
            theta[flatten_idx:flatten_idx+meta.ms[i]] = meta.thetas[i, :meta.ms[i]]
            flatten_idx += meta.ms[i]

        self.set_given_ansatz_functions(means=means, sigma_i=meta.sigma_i)
        self.sigma_i_meta = sigma_i_meta
        self.k = k
        self.K_meta = K_meta
        self.theta = theta / K_meta
        self.distributed = 'meta'
        self.theta_type = 'meta'

    def set_theta_null(self):
        ''' set the parameters of the gaussian ansatz to zero
        '''
        self.theta = np.zeros(self.m)
        self.theta_type = 'null'

    def set_theta_random(self):
        ''' sample the parameters of the gaussian ansatz uniformly
        '''
        bound = 0.1
        self.theta = np.random.uniform(-bound, bound, self.m)
        self.theta_type = 'random'

    def set_theta_hjb(self, h):
        ''' compute the parameters of the gaussian ansatz functions such that the corresponding
            value function solve the hjb solution
        '''
        # get hjb solver
        hjb_sol = self.sde.get_hjb_solver(h)

        # flatten domain and value function
        Nh = hjb_sol.Nh
        x = hjb_sol.domain_h.reshape(Nh, self.sde.d)
        value_f = hjb_sol.value_f.reshape(Nh,)

        # compute the optimal theta given a basis of ansatz functions
        v = self.mvn_pdf_basis(x)
        self.theta, _, _, _ = np.linalg.lstsq(v, value_f, rcond=None)
        self.h = h
        self.theta_type = 'optimal'

    #TODO: revise!
    def set_theta_metadynamics(self, meta, distributed, h=None):
        '''
        '''

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
        v = self.mvn_pdf_basis(x)

        # solve theta v \theta = value_f_meta
        self.theta, _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)

        msg = 'uniformly distributed gaussian ansatz fitted (least squares) with metadynamics\n'
        print(msg)

        # save parameters
        self.theta_type = 'meta'
        self.sigma_i_meta = meta.sigma_i
        self.delta_meta = meta.delta
        self.K_meta = meta.K
        self.seed_meta = meta.seed

    def mvn_pdf_basis_numpy(self, x):
        ''' Multivariate normal pdf (nd Gaussian) basis V(x; means, cov) with different m means
            but same covariance matrix evaluated at x. Computations with numpy

        Parameters
        ----------
        x: (K, d)-array
            position

        Returns
        -------
        mvn_pdf_basis
            (K, m)-array
        '''
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.shape[1] == self.sde.d, ''
        K = x.shape[0]

        # prepare position and means for broadcasting
        x = x[:, np.newaxis, :]
        means = self.means[np.newaxis, :, :]

        # compute exponential term
        if self.is_cov_scalar_matrix:
            exp_term = - 0.5 * np.sum((x - means)**2, axis=2) / self.sigma_i
        else:
            x_centered = (x - means).reshape(K*self.m, self.sde.d)
            exp_term = np.matmul(x_centered, self.inv_cov)
            exp_term = np.sum(exp_term * x_centered, axis=1).reshape(K, self.m)
            exp_term *= - 0.5

        # compute normalization factor if needed
        if self.normalized:

            # compute norm factor
            norm_factor = np.sqrt(((2 * np.pi) ** self.sde.d) * self.det_cov)

            # normalize
            mvn_pdf_basis = np.exp(exp_term) / norm_factor
        else:
            mvn_pdf_basis = np.exp(exp_term)

        return mvn_pdf_basis

    def mvn_pdf_basis(self, x):
        ''' Multivariate normal pdf (nd Gaussian) basis v(x; means, cov) with different means
            but same covariance matrix evaluated at x. Computations with pytorch.

        Parameters
        ----------
        x: (K, d)-array
            position

        Returns
        -------
        mvn_pdf_basis
            (K, m)-array
        '''
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.shape[1] == self.sde.d, ''
        K = x.shape[0]

        # tensorize x
        x_tensor = torch.tensor(x)

        # compute log of the basis

        # scalar covariance matrix
        if self.is_cov_scalar_matrix:
            log_mvn_pdf_basis = - 0.5 * torch.sum(
                (x_tensor.view(K, 1, self.sde.d)
               - self.means_tensor.view(1, self.m, self.sde.d))**2,
                axis=2,
            ) / self.sigma_i

            # add normalization factor
            if self.normalized:
                log_mvn_pdf_basis -= torch.log(2 * torch.tensor(np.pi) * self.sigma_i) \
                                   * self.sde.d / 2

        # general covariance matrix
        else:
            #TODO! convert to pytorch

            # prepare position and means for broadcasting
            x = x[:, np.newaxis, :]
            means = self.means[np.newaxis, :, :]

            x_centered = (x - means).reshape(K * self.m, self.sde.d)
            log_mvn_pdf_basis = -0.5 * np.sum(
                np.matmul(x_centered, self.inv_cov) * x_centered,
                axis=1,
            ).reshape(K, self.m)

            # add normalization factor
            if self.normalized:
                log_mvn_pdf_basis -= torch.log((2 * torch.tensor(np.pi)) ** self.sde.d * self.det_cov) / 2

        return torch.exp(log_mvn_pdf_basis).numpy()

    def mvn_pdf_gradient_basis_numpy(self, x):
        ''' Multivariate normal pdf gradient (nd Gaussian gradients) basis \nabla V(x; means, cov)
        with different means but same covaianc matrix evaluated at x. Computations with numpy

        Parameters
        ----------
        x: (K, d)-array
            posicion

        Returns
        -------
        mvn_pdf_gradient
            (K, m, d)-array
        '''
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.shape[1] == self.sde.d, ''
        K = x.shape[0]

        # compute nd gaussian basis
        mvn_pdf_basis = self.mvn_pdf_basis(x)

        # compute gradient of the exponential term
        if self.is_cov_scalar_matrix:
            grad_exp_term = (x[:, np.newaxis, :] - self.means[np.newaxis, :, :]) / self.sigma_i
        else:
            grad_exp_term = 0.5 * np.matmul(
                x[:, np.newaxis, :] - self.means[np.newaxis, :, :],
                self.inv_cov + self.inv_cov.T,
            )

        # compute gaussian gradients basis
        return - grad_exp_ternm * mvn_pdf_basis[:, :, np.newaxis]

    def mvn_pdf_gradient_basis(self, x):
        ''' Multivariate normal pdf gradient (nd Gaussian gradients) basis \nabla V(x; means, cov)
        with different means but same covaianc matrix evaluated at x. Computations with torch

        Parameters
        ----------
        x: (K, d)-array
            posicion

        Returns
        -------
        mvn_pdf_gradient
            (K, m, d)-array
        '''
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.shape[1] == self.sde.d, ''
        K = x.shape[0]

        # get nd gaussian basis
        mvn_pdf_basis = self.mvn_pdf_basis(x)

        # tensorize x
        x_tensor = torch.tensor(x)

        # compute gradient of the exponential term

        # scalar covariance matrix
        if self.is_cov_scalar_matrix:
            grad_exp_term = (
                x_tensor.view(K, 1, self.sde.d) - self.means_tensor.view(1, self.m, self.sde.d)
            ) / self.sigma_i

        # general covariance matrix
        else:
            grad_exp_term = 0.5 * np.matmul(x - means, self.inv_cov + self.inv_cov.T)

        # compute gaussian gradients basis
        return (- grad_exp_term * mvn_pdf_basis[:, :, np.newaxis]).numpy()

    def set_value_function_constant_to_zero(self):
        ''' sets the value function constant to zero
        '''
        self.K_value_f = 0

    #TODO! revise method
    def set_value_function_constant_target_set(self):
        ''' sets the value function constant such that the evaluation of the value function at the
            target sets points on average is zero
        '''

        # discretize domain
        self.discretize_domain()

        # get idx in the target set
        idx_ts = self.get_idx_target_set()

        # impose value function in the target set to be null
        self.K_value_f = - np.mean(value_f[idx_ts])

    def set_value_function_constant_corner(self, theta=None):
        ''' sets the value function constant such that the evaluation of the value function at the
            target set corner (1, ..., 1) is zero
        '''

        if theta is None:
            theta = self.theta

        # define target set corner (1, ..., 1)
        x = np.ones((1, self.sde.d))

        # evaluate value function at the corner
        value_f =  np.dot(self.mvn_pdf_basis(x), - theta / self.sde.sigma)

        self.K_value_f = - value_f


    def value_function(self, x, theta=None):
        '''This method computes the value function evaluated at x

        Parameters
        ----------
        x: (K, d)-array
            position
        theta: (m,)-array
            parameters

        Returns
        -------
        array
            value function evaluated at the given points
        '''
        if theta is None:
            theta = self.theta

        # value function without constant K
        value_f =  np.dot(self.mvn_pdf_basis(x), - theta / self.sde.sigma)

        return value_f + self.K_value_f


    def control(self, x, theta=None):
        '''This method computes the control evaluated at x

        Parameters
        ----------
        x: (K, d)-array
            position
        theta: (m,)-array
            parameters

        Returns
        -------
        array
            control evaluated at the given points
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.sde.d, ''
        if theta is None:
            theta = self.theta

        control = np.tensordot(
            self.mvn_pdf_gradient_basis(x),
            theta,
            axes=([1], [0]),
        )
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
            f.write('delta_meta: {:2.2f}\n'.format(self.delta_meta))
            f.write('K_meta: {:2.2f}\n'.format(self.K_meta))

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
