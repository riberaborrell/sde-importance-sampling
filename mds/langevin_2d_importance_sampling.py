from mds.gaussian_2d_ansatz_functions import GaussianAnsatz
from mds.plots_2d import Plot2d
from mds.potentials_and_gradients_2d import get_potential_and_gradient
from mds.utils import get_example_data_path, get_gd_data_path, get_time_in_hms, make_dir_path
from mds.validation import is_2d_valid_interval, is_2d_valid_target_set, is_2d_valid_control

import numpy as np
import time
import os

class Sampling:
    '''
    '''

    def __init__(self, potential_name, alpha, beta,
                 target_set, domain=None, h=0.01, is_drifted=False):
        '''
        '''
        # get potential and gradient functions
        potential, gradient, _, _, _ = get_potential_and_gradient(potential_name, alpha)

        # validate domain and target set
        if domain is None:
            domain = np.array([[-3, 3], [-3, 3]])
        is_2d_valid_interval(domain)
        is_2d_valid_target_set(domain, target_set)

        #seed
        self.seed = None

        # sde parameters
        self.potential_name = potential_name
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta

        # sampling
        self.domain = domain
        self.xzero = None
        self.target_set = target_set
        self.N = None
        self.is_drifted = is_drifted
        self.is_optimal = None

        # domain discretization
        self.h = h
        self.domain_h = None
        self.Nx = None
        self.Ny = None
        self.Nh = None
        self.discretize_domain()

        # Euler-Marujama
        self.dt = None
        self.N_lim = None

        # ansatz functions (gaussians) and coefficients
        self.ansatz = None
        self.theta_type = None
        self.theta = None

        # variables

        # trajectories which arrived
        self.N_arrived = None
        self.been_in_target_set = None

        # first hitting time
        self.fht = None
        self.first_fht = None
        self.last_fht = None
        self.mean_fht = None
        self.var_fht = None
        self.re_fht = None

        # quantity of interest
        self.mean_I = None
        self.var_I = None
        self.re_I = None

        # reweighting
        self.M1_fht = None
        self.M2_fht = None
        self.k = None
        self.M1_k = None
        self.M2_k = None
        self.mean_M_k= None

        self.mean_I_u = None
        self.var_I_u = None
        self.re_I_u = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # reference solution
        self.ref_sol = None
        self.value_f_at_xzero = None

        # metadynamics
        self.meta_bias_pot = None

        # dir_path
        self.example_dir_path = None
        self.dir_path = None
        self.set_example_dir_path()

    def discretize_domain(self, h=None):
        ''' this method discretizes the rectangular domain uniformly with step-size h
        Args:
            h (float): step-size
        '''
        if h is None:
            h = self.h

        d_xmin, d_xmax = self.domain[0]
        d_ymin, d_ymax = self.domain[1]
        x = np.around(np.arange(d_xmin, d_xmax + h, h), decimals=3)
        y = np.around(np.arange(d_ymin, d_ymax + h, h), decimals=3)
        X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')
        self.domain_h = np.dstack((X, Y))
        self.Nx = x.shape[0]
        self.Ny = y.shape[0]
        self.Nh = self.Nx * self.Ny

    def set_example_dir_path(self):
        self.example_dir_path = get_example_data_path(self.potential_name, self.alpha,
                                                      self.beta, self.target_set)

    def set_dir_path(self, dir_path):
        self.dir_path = dir_path
        make_dir_path(self.dir_path)

    def set_gaussian_ansatz_functions(self, m_x, m_y, sigma_x=None, sigma_y=None):
        '''
        '''
        assert self.is_drifted, ''

        # set gaussian ansatz functions
        ansatz = GaussianAnsatz(
            domain=self.domain,
        )
        ansatz.set_unif_dist_ansatz_functions(m_x, m_y, sigma_x, sigma_y)

        # set ansatz dir path
        ansatz.set_dir_path(self.example_dir_path)
        self.ansatz = ansatz

    def set_bias_potential(self, theta, means, cov):
        ''' set the gaussian ansatz functions and the coefficients theta
        Args:
            theta ((m,)-array): parameters
            means ((m, 2)-array): mean of each gaussian
            cov ((m, 2, 2)-array) : covaraince matrix of each gaussian
        '''
        assert self.is_drifted, ''
        assert theta.shape[0] == means.shape[0], ''

        # set gaussian ansatz functions
        ansatz = GaussianAnsatz(domain=self.domain)
        ansatz.set_given_ansatz_functions(means, cov)

        self.ansatz = ansatz
        self.theta = theta

    def get_idx_discretized_domain(self, x):
        assert x.ndim == 2, ''
        assert x.shape[0] == self.N, ''
        assert x.shape[1] == 2, ''

        x1 = x[:, 0].reshape(self.N, 1)
        x2 = x[:, 1].reshape(self.N, 1)

        axis1_h = self.domain_h[:, 0, 0]
        axis2_h = self.domain_h[0, :, 1]

        idx_x1 = np.argmin(np.abs(axis1_h - x1), axis=1)
        idx_x2 = np.argmin(np.abs(axis2_h - x2), axis=1)
        return idx_x1, idx_x2

    def load_meta_bias_potential(self):
        if not self.meta_bias_pot:
            file_path = os.path.join(
                self.example_dir_path,
                'metadynamics',
                'bias_potential.npz',
            )
            self.meta_bias_pot = np.load(file_path)

    def load_reference_solution(self):
        if not self.ref_sol:
            h_ext = '_h{:.0e}'.format(self.h)
            file_name = 'reference_solution' + h_ext + '.npz'

            file_path = os.path.join(
                self.example_dir_path,
                'reference_solution',
                file_name,
            )
            self.ref_sol = np.load(file_path)

    def get_value_f_at_xzero(self):
        # load ref sol
        self.load_reference_solution()
        x = self.ref_sol['domain_h']
        F = self.ref_sol['F']

        # evaluate F at xzero
        idx_x = np.where(
            (x[:, :, 0] == self.xzero[0]) &
            (x[:, :, 1] == self.xzero[1])
        )
        assert idx_x[0].shape[0] == idx_x[1].shape[0] == 1, ''
        idx_x1 = idx_x[0][0]
        idx_x2 = idx_x[1][0]

        self.value_f_at_xzero = F[idx_x1, idx_x2]

    def set_theta_optimal(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.load_reference_solution()
        Nx, Ny, _ = self.ref_sol['domain_h'].shape
        x = self.ref_sol['domain_h'].reshape(Nx * Ny, 2)
        F = self.ref_sol['F'].reshape(Nx * Ny,)

        # compute the optimal theta given a basis of ansatz functions
        v = self.ansatz.basis_value_f(x)
        self.theta, _, _, _ = np.linalg.lstsq(v, F, rcond=None)
        self.theta_type = 'optimal'

        # set drifted sampling dir path
        dir_path = os.path.join(self.ansatz.dir_path, 'optimal-importance-sampling')
        self.set_dir_path(dir_path)

    def set_theta_null(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.theta = np.zeros(self.ansatz.m)
        self.theta_type = 'null'

        # set drifted sampling dir path
        dir_path = os.path.join(self.ansatz.dir_path, 'null-importance-sampling')
        self.set_dir_path(dir_path)

    def set_theta_from_metadynamics(self):
        '''
        '''
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        x = self.domain_h.reshape(self.Nh, 2)

        self.load_meta_bias_potential()
        meta_ms = self.meta_bias_pot['ms']
        meta_thetas = self.meta_bias_pot['thetas']
        meta_means = self.meta_bias_pot['means']
        meta_cov = self.meta_bias_pot['cov']

        meta_N = meta_ms.shape[0]

        thetas = np.empty((meta_N, self.ansatz.m))

        for i in np.arange(meta_N):

            # create ansatz functions from meta
            meta_ansatz = GaussianAnsatz(domain=self.domain)
            meta_ansatz.set_given_ansatz_functions(
                means=meta_means[i, :meta_ms[i]],
                cov=meta_cov,
            )

            # meta value function evaluated at the grid
            value_f_meta = self.value_function(x, meta_thetas[i, :meta_ms[i]], meta_ansatz)

            # ansatz functions evaluated at the grid
            v = self.ansatz.basis_value_f(x)

            # solve theta V = \Phi
            thetas[i], _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)

        self.theta = np.mean(thetas, axis=0)
        self.theta_type = 'meta'

        # set drifted sampling dir path
        dir_path = os.path.join(self.ansatz.dir_path, 'meta-importance-sampling')
        self.set_dir_path(dir_path)

    def set_theta_from_gd(self, gd_type, gd_theta_init, gd_lr):
        '''
        '''
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        # get gd dir path
        ansatz_dir_path = self.ansatz.dir_path
        gd_dir_path = get_gd_data_path(ansatz_dir_path, gd_type, gd_theta_init, gd_lr)

        # load gd
        file_path = os.path.join(gd_dir_path, 'gd.npz')
        gd = np.load(file_path)

        # get last theta
        last_epoch = gd['epochs'] - 1
        self.theta = gd['thetas'][last_epoch, :]
        self.theta_type = 'gd'

        # set drifted sampling dir path
        dir_path = os.path.join(gd_dir_path, 'gd-importance-sampling')
        self.set_dir_path(dir_path)

    def value_function(self, x, theta=None, ansatz=None):
        '''This method computes the value function evaluated at x

        Args:
            x ((N, 2)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions

        Return:
        '''
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz

        target_set_x, target_set_y = self.target_set
        target_set_x_min, target_set_x_max = target_set_x
        target_set_y_min, target_set_y_max = target_set_y

        # value function with constant K=0
        basis_value_f =  ansatz.basis_value_f(x)
        value_f =  np.dot(basis_value_f, theta)

        # compute constant

        # get idx for x in the target set
        is_in_target_set = (
            (x[:, 0] >= target_set_x_min) &
            (x[:, 0] <= target_set_x_max) &
            (x[:, 1] >= target_set_y_min) &
            (x[:, 1] <= target_set_y_max)
        )
        idx_ts = np.where(is_in_target_set == True)[0]

        # impose value function in the target set to be null
        K = - np.mean(value_f[idx_ts])

        # get idx for x to be (1, 1)
        idx_corner_ts = np.where(
            (x[:, 0] == 1) &
            (x[:, 1] == 1)
        )[0]

        # impose value function in (1, 1) to be null
        L = - value_f[idx_corner_ts]

        return value_f + L


    def control(self, x, theta=None, ansatz=None):
        '''This method computes the control evaluated at x

        Args:
            x ((N, 2)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions
        '''
        N = x.shape[0]
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz

        basis_control = ansatz.basis_control(x)
        control = np.empty((N, 2))
        control[:, 0] = np.dot(basis_control[:, :, 0], theta).reshape((N,))
        control[:, 1] = np.dot(basis_control[:, :, 1], theta).reshape((N,))

        return control

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x ((N, 2)-array) : position
            theta ((m,)-array): parameters
        '''
        return 2 * self.value_function(x, theta)

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u ((N, 2)-array) : control at x
        '''
        return - np.sqrt(2) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x ((N, 2)-array) : position/s
            theta ((m,)-array): parameters
        '''
        return self.potential(x) + self.bias_potential(x, theta)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x ((N, 2)-array) : position/s
            u ((N, 2)-array) : control at x
        '''
        assert x.shape == u.shape

        return self.gradient(x) + self.bias_gradient(u)

    def set_sampling_parameters(self, xzero, N, dt, N_lim, seed=None):
        '''
        '''
        # set random seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.xzero = xzero
        self.N = N

        # Euler-Marujama
        self.dt = dt
        self.N_lim = N_lim

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def initialize_fht(self):
        '''
        '''
        assert self.N is not None, ''

        self.been_in_target_set = np.repeat([False], self.N)
        self.fht = np.empty(self.N)

    def initialize_girsanov_martingale_terms(self):
        '''
        '''
        assert self.N is not None, ''

        self.M1_fht = np.empty(self.N)
        self.M2_fht = np.empty(self.N)
        self.M1_k = np.empty((self.N, 10))
        self.M2_k = np.empty((self.N, 10))

    def sde_update(self, x, gradient, dB):
        drift = - gradient * self.dt
        diffusion = np.dot(dB, np.sqrt(2 / self.beta) * np.eye(2))
        return x + drift + diffusion

    def get_idx_new_in_target_set(self, x):
        target_set_x, target_set_y = self.target_set
        target_set_x_min, target_set_x_max = target_set_x
        target_set_y_min, target_set_y_max = target_set_y

        # trajectories in the target set
        is_in_target_set = (
            (x[:, 0] >= target_set_x_min) &
            (x[:, 0] <= target_set_x_max) &
            (x[:, 1] >= target_set_y_min) &
            (x[:, 1] <= target_set_y_max)
        )

        # indices of trajectories new in the target set
        idx_new = np.where(
            (is_in_target_set == True) &
            (self.been_in_target_set == False)
        )[0]

        # update list of indices whose trajectories have been in the target set
        self.been_in_target_set[idx_new] = True

        return idx_new

    def sample_not_drifted(self):
        self.start_timer()
        self.initialize_fht()

        # initialize xtemp
        xtemp = np.full((self.N, 2), self.xzero)

        for n in np.arange(1, self.N_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, 2 * self.N).reshape(self.N, 2)

            # compute gradient
            gradient = self.gradient(xtemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # get indices from the trajectories which are new in target
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save first hitting time
            self.fht[idx_new] = n * self.dt

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        self.compute_fht_statistics()
        self.compute_I_statistics()
        self.stop_timer()

    def sample_drifted(self):
        self.start_timer()
        self.initialize_fht()
        self.initialize_girsanov_martingale_terms()

        # initialize xtemp
        xtemp = np.full((self.N, 2), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(self.N)
        M2temp = np.zeros(self.N)
        k = np.array([])

        for n in np.arange(1, self.N_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, 2 * self.N).reshape(self.N, 2)

            # control at Xtemp
            utemp = self.control(xtemp)

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # Girsanov Martingale terms
            M1temp -= np.sqrt(self.beta) * np.matmul(utemp, dB.T).diagonal()
            M2temp -= self.beta * 0.5 * (np.linalg.norm(utemp, axis=1) ** 2) * self.dt

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save first hitting time and Girsanov Martingale terms
            self.fht[idx_new] = n * self.dt
            self.M1_fht[idx_new] = M1temp[idx_new]
            self.M2_fht[idx_new] = M2temp[idx_new]

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()

    def sample_optimal_drifted(self):
        self.start_timer()
        self.initialize_fht()
        self.initialize_girsanov_martingale_terms()

        # initialize xtemp
        xtemp = np.full((self.N, 2), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(self.N)
        M2temp = np.zeros(self.N)
        k = np.array([])

        # load optimal control
        self.load_reference_solution()
        u_opt = self.ref_sol['u_opt']

        for n in np.arange(1, self.N_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, 2 * self.N).reshape(self.N, 2)

            # control at xtemp
            idx_grid_x1, idx_grid_x2 = self.get_idx_discretized_domain(xtemp)
            utemp = u_opt[idx_grid_x1, idx_grid_x2]

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # Girsanov Martingale terms
            M1temp -= np.sqrt(self.beta) * np.matmul(utemp, dB.T).diagonal()
            M2temp -= self.beta * 0.5 * (np.linalg.norm(utemp, axis=1) ** 2) * self.dt

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save first hitting time and Girsanov Martingale terms
            self.fht[idx_new] = n * self.dt
            self.M1_fht[idx_new] = M1temp[idx_new]
            self.M2_fht[idx_new] = M2temp[idx_new]

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()

    def sample_meta(self):
        self.initialize_fht()

        # initialize xtemp
        xtemp = np.empty((self.N_lim +1, self.N, 2))
        xtemp[0] = np.full((self.N, 2), self.xzero)

        for n in np.arange(self.N_lim):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, 2 * self.N).reshape(self.N, 2)

            if not self.is_drifted:
                # compute gradient
                gradient = self.gradient(xtemp[n])

            else:
                # control at xtemp
                utemp = self.control(xtemp[n])

                # compute gradient
                gradient = self.tilted_gradient(xtemp[n], utemp)

            # sde update
            xtemp[n+1] = self.sde_update(xtemp[n], gradient, dB)

            # get indices from the trajectories which are new in target
            idx_new = self.get_idx_new_in_target_set(xtemp[n+1])

            # check if the half of the trajectories have arrived to the target set
            if np.sum(self.been_in_target_set) >= self.N / 2:
                return True, xtemp[:n+1]

        return False, xtemp

    def sample_loss(self):
        self.initialize_fht()

        # number of ansatz functions
        m = self.ansatz.m

        # initialize statistics 
        J = np.zeros(self.N)
        grad_J = np.zeros((self.N, m))

        # initialize xtemp
        xtemp = np.full((self.N, 2), self.xzero)

        # initialize ipa variables
        cost_temp = np.zeros(self.N)
        grad_phi_temp = np.zeros((self.N, m))
        grad_S_temp = np.zeros((self.N, m))

        for n in np.arange(1, self.N_lim+1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, 2 * self.N).reshape(self.N, 2)

            # control
            btemp = self.ansatz.basis_control(xtemp)
            utemp = self.control(xtemp)
            lower_bound = -10 * np.ones(2)
            upper_bound = 10 * np.ones(2)
            if not is_2d_valid_control(utemp, lower_bound, upper_bound):
                return False, None, None, None

            # ipa statistics 
            normed_utemp = np.linalg.norm(utemp, axis=1)
            cost_temp += 0.5 * (normed_utemp ** 2) * self.dt
            grad_phi_temp += np.sum(utemp[:, None, :] * btemp, axis=2) * self.dt
            grad_S_temp -= np.sqrt(self.beta) * np.sum(dB[:, None, :] * btemp, axis=2)

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save ipa statistics
            J[idx_new] = n * self.dt + cost_temp[idx_new]
            grad_J[idx_new, :] = grad_phi_temp[idx_new, :] \
                               - (n * self.dt + cost_temp[idx_new])[:, None] \
                               * grad_S_temp[idx_new, :]

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        # compute averages
        mean_J = np.mean(J)
        mean_grad_J = np.mean(grad_J, axis=0)

        return True, mean_J, mean_grad_J, n


    def compute_mean_variance_and_rel_error(self, x):
        '''This method computes the mean, the variance and the relative
           error of the given ndarray x
        Args:
            x (ndarray) :
        '''
        # check that dim of x is not 0
        if x.shape[0] == 0:
            return np.nan, np.nan, np.nan

        mean = np.mean(x)
        var = np.var(x)
        if mean != 0:
            re = np.sqrt(var) / mean
        else:
            re = np.nan

        return mean, var, re

    def compute_fht_statistics(self):
        been_in_target_set = self.been_in_target_set
        fht = self.fht

        # count trajectories which have arrived
        idx_arrived = np.where(been_in_target_set == True)
        self.N_arrived = fht[idx_arrived].shape[0]
        if self.N_arrived != self.N:
            return

        # replace trajectories which have not arrived
        idx_not_arrived = np.where(been_in_target_set == False)
        fht[idx_not_arrived] = self.N_lim
        self.fht = fht

        # first and last fht
        self.first_fht = np.min(fht)
        self.last_fht = np.max(fht)

        # compute mean and variance of fht
        self.mean_fht, \
        self.var_fht, \
        self.re_fht = self.compute_mean_variance_and_rel_error(fht)

    def compute_I_statistics(self):
        # compute mean and variance of I
        I = np.exp(- self.fht)
        self.mean_I, \
        self.var_I, \
        self.re_I = self.compute_mean_variance_and_rel_error(I)

    def compute_I_u_statistics(self):
        # compute mean of M_fht
        M_fht = np.exp(self.M1_fht + self.M2_fht)

        # compute mean and variance of I_u
        I_u = np.exp(- self.fht) * M_fht
        self.mean_I_u, \
        self.var_I_u, \
        self.re_I_u = self.compute_mean_variance_and_rel_error(I_u)

    def save_not_drifted(self):
        # file name
        N_ext = '_N{:.0e}'.format(self.N)
        file_name = 'mc_sampling' + N_ext + '.npz'
        np.savez(
            os.path.join(self.dir_path, file_name),
            N=self.N,
            mean_I=self.mean_I,
            var_I=self.var_I,
            re_I=self.re_I,
        )

    def load_not_drifted(self, N):
        # file name
        N_ext = '_N{:.0e}'.format(N)
        file_name = 'mc_sampling' + N_ext + '.npz'
        file_path = os.path.join(self.example_dir_path, 'not-drifted-sampling', file_name)
        data = np.load(file_path, allow_pickle=True)
        self.N = data['N']
        self.mean_I = data['mean_I']
        self.var_I = data['var_I']
        self.re_I = data['re_I']

    def write_sde_parameters(self, f):
        '''
        '''
        f.write('SDE parameters\n')
        f.write('potential: {}\n'.format(self.potential_name))
        f.write('alpha: {}\n'.format(self.alpha))
        f.write('beta: {:2.1f}\n'.format(self.beta))
        f.write('drifted process: {}\n\n'.format(self.is_drifted))

    def write_euler_maruyama_parameters(self, f):
        f.write('Euler-Maruyama discretization parameters\n')
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('maximal time steps: {:,d}\n\n'.format(self.N_lim))

    def write_sampling_parameters(self, f):
        f.write('Sampling parameters\n')
        if self.seed:
            f.write('seed: {:2.1f}'.format(self.seed))
        f.write('xzero: ({:2.1f}, {:2.1f})\n'.format(self.xzero[0], self.xzero[1]))
        f.write('target set: [[{:2.1f}, {:2.1f}], [{:2.1f}, {:2.1f}]]\n'.format(
            self.target_set[0, 0],
            self.target_set[0, 1],
            self.target_set[1, 0],
            self.target_set[1, 1],
        ))
        f.write('sampled trajectories: {:,d}\n\n'.format(self.N))

    def write_report(self):
        '''
        '''
        # set file path
        if self.is_drifted:
            theta_stamp = 'theta-{}_'.format(self.theta_type)
        else:
            theta_stamp = ''

        trajectories_ext = 'N{:.0e}'.format(self.N)
        file_name = 'report_2d_' + theta_stamp + trajectories_ext + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        self.write_sde_parameters(f)
        self.write_euler_maruyama_parameters(f)
        self.write_sampling_parameters(f)

        if self.is_drifted and not self.is_optimal:
            self.ansatz.write_ansatz_parameters(f)

        f.write('Statistics\n\n')

        f.write('trajectories which arrived: {:2.2f} %\n'
                ''.format(100 * self.N_arrived / self.N))
        f.write('used time steps: {:,d}\n\n'.format(int(self.last_fht / self.dt)))
        if self.N_arrived == 0:
            f.close()
            return

        f.write('First hitting time\n')
        f.write('first fht = {:2.3f}\n'.format(self.first_fht))
        f.write('last fht = {:2.3f}\n'.format(self.last_fht))
        f.write('E[fht] = {:2.3f}\n'.format(self.mean_fht))
        f.write('Var[fht] = {:2.3f}\n'.format(self.var_fht))
        f.write('RE[fht] = {:2.3f}\n\n'.format(self.re_fht))

        if not self.is_drifted:
            f.write('Quantity of interest\n')
            f.write('E[exp(-beta * fht)] = {:2.3e}\n'.format(self.mean_I))
            f.write('Var[exp(-beta * fht)] = {:2.3e}\n'.format(self.var_I))
            f.write('RE[exp(-beta * fht)] = {:2.3e}\n\n'.format(self.re_I))

        else:
            #f.write('Girsanov Martingale\n')
            #for i, n in enumerate(self.k):
            #    f.write('E[M_k] = {:2.3e}, k = {:d}\n'.format(self.mean_M_k[i], int(n)))

            f.write('\nReweighted Quantity of interest\n')
            f.write('E[exp(-beta * fht) * M_fht] = {:2.3e}\n'
                    ''.format(self.mean_I_u))
            f.write('Var[exp(-beta * fht) * M_fht] = {:2.3e}\n'
                    ''.format(self.var_I_u))
            f.write('RE[exp(-beta * fht) * M_fht] = {:2.3e}\n\n'
                    ''.format(self.re_I_u))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        f.close()

    def plot_appr_psi_surface(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'appr_psi_surface'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        Vbias = self.bias_potential(x).reshape(self.Nx, self.Ny)
        appr_F = Vbias / 2
        appr_Psi = np.exp(- self.beta * appr_F)
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$\Psi(x_1, x_2)$')
        plt2d.surface(X, Y, appr_Psi)

    def plot_appr_psi_contour(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'appr_psi_contour'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        Vbias = self.bias_potential(x).reshape(self.Nx, self.Ny)
        appr_F = Vbias / 2
        appr_Psi = np.exp(- self.beta * appr_F)
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$\Psi(x_1, x_2)$')
        plt2d.contour(X, Y, appr_Psi)

    def plot_appr_free_energy_surface(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'appr_free_energy_surface'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        Vbias = self.bias_potential(x).reshape(self.Nx, self.Ny)
        appr_F = Vbias / 2
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$F(x_1, x_2)$')
        plt2d.surface(X, Y, appr_F)

    def plot_appr_free_energy_contour(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'appr_free_energy_contour'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        Vbias = self.bias_potential(x).reshape(self.Nx, self.Ny)
        appr_F = Vbias / 2
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$F(x_1, x_2)$')
        plt2d.contour(X, Y, appr_F)

    def plot_control(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'control'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        u = self.control(x).reshape(self.Nx, self.Ny, 2)
        u_x = u[:, :, 0]
        u_y = u[:, :, 1]
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$u(x_1, x_2)$')
        plt2d.vector_field(X, Y, u_x, u_y, scale=8)

    def plot_tilted_potential_surface(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'tilted_potential_surface'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        V = self.potential(x).reshape(self.Nx, self.Ny)
        if self.is_drifted:
            Vbias = self.bias_potential(x).reshape(self.Nx, self.Ny)
        else:
            Vbias = np.zeros((self.Nx, self.Ny))
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$\tilde{V}(x_1, x_2)$')
        plt2d.set_xlim(-2, 2)
        plt2d.set_ylim(-2, 2)
        plt2d.set_zlim(0, 10)
        plt2d.surface(X, Y, V + Vbias)

    def plot_tilted_potential_contour(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'tilted_potential_contour'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        V = self.potential(x).reshape(self.Nx, self.Ny)
        if self.is_drifted:
            Vbias = self.bias_potential(x).reshape(self.Nx, self.Ny)
        else:
            Vbias = np.zeros((self.Nx, self.Ny))
        levels = np.logspace(-2, 1, 20, endpoint=True)
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$\tilde{V}(x_2, x_1)$')
        plt2d.set_xlim(-2, 2)
        plt2d.set_ylim(-2, 2)
        plt2d.set_zlim(0, 10)
        plt2d.contour(X, Y, V + Vbias, levels=levels)

    def plot_tilted_drift(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'tilted_drift'
        if dir_path is None:
            dir_path = self.dir_path
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape(self.Nh, 2)
        dV = self.gradient(x).reshape(self.Nx, self.Ny, 2)
        if self.is_drifted:
            u = self.control(x).reshape(self.Nx, self.Ny, 2)
            dVbias = self.bias_gradient(u)
        else:
            dVbias = np.zeros((self.Nx, self.Ny, 2))
        U = - dV[:, :, 0] - dVbias[:, :, 0]
        V = - dV[:, :, 1] - dVbias[:, :, 1]
        plt2d = Plot2d(dir_path, file_name)
        #plt2d.set_title(r'$-\nabla \tilde{V}(x_1, x_2)$')
        plt2d.set_xlim(-1.5, 1.5)
        plt2d.set_ylim(-1.5, 1.5)
        plt2d.vector_field(X, Y, U, V, scale=50)
