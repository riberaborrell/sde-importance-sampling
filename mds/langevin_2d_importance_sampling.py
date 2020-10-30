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

        # dir_path
        self.example_dir_path = get_example_data_path(potential_name, alpha,
                                                      beta, target_set)
        self.dir_path = None

        #seed
        self.seed = None

        # sde parameters
        self.potential_name = potential_name
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.is_drifted = is_drifted

        # sampling
        self.domain = domain
        self.xzero = None
        self.target_set = target_set
        self.M = None

        # domain discretization
        self.h = h
        self.domain_h = None
        self.Nx = None
        self.Ny = None
        self.N = None
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
        self.M_arrived = None
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


    def discretize_domain(self, h=None):
        ''' this method discretizes the rectangular domain uniformly with step-size h
        Args:
            h (float): step-size
        '''
        if h is None:
            h = self.h

        d_xmin, d_xmax = self.domain[0]
        d_ymin, d_ymax = self.domain[1]

        x = np.arange(d_xmin, d_xmax + h, h)
        y = np.arange(d_ymin, d_ymax + h, h)
        X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')
        self.domain_h = np.dstack((X, Y))
        self.Nx = x.shape[0]
        self.Ny = y.shape[0]
        self.N = x.shape[0] * y.shape[0]

    def set_not_drifted_dir_path(self):
        self.dir_path = os.path.join(self.example_dir_path, 'not-drifted-sampling')
        make_dir_path(self.dir_path)

    def set_drifted_dir_path(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.dir_path = os.path.join(self.ansatz.dir_path, 'drifted-sampling')
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

    def load_meta_bias_potential(self):
        if not self.meta_bias_pot:
            file_path = os.path.join(
                self.example_dir_path,
                'metadynamics',
                'bias_potential.npz',
            )
            self.meta_bias_pot = np.load(file_path)

    def set_bias_potential_from_metadynamics(self):
        ''' set the gaussian ansatz functions and the coefficients from metadynamics
        '''
        self.load_meta_bias_potential()
        meta_theta = self.meta_bias_pot['omegas'] / 2
        meta_mus = self.meta_bias_pot['mus']
        meta_sigmas = self.meta_bias_pot['sigmas']

        assert meta_theta.shape == meta_mus.shape == meta_sigmas.shape, ''

        self.set_bias_potential(meta_theta, meta_mus, meta_sigmas)

    def load_reference_solution(self):
        if not self.ref_sol:
            file_path = os.path.join(
                self.example_dir_path,
                'reference_solution',
                'reference_solution.npz',
            )
            self.ref_sol = np.load(file_path)

    def get_value_f_at_xzero(self):
        xzero = self.xzero

        # load ref sol
        self.load_reference_solution()
        x = self.ref_sol['domain_h']
        F = self.ref_sol['F']

        # evaluate F at xzero
        idx_x = np.where(
            (x[:, :, 0] == xzero[0]) &
            (x[:, :, 1] == xzero[1])
        )
        assert idx_x[0].shape[0] == idx_x[1].shape[0] == 1, ''
        idx_x1 = idx_x[0][0]
        idx_x2 = idx_x[1][0]

        self.value_f_at_xzero = F[idx_x1, idx_x2]

    def set_theta_optimal(self):
        assert self.ansatz is not None, ''
        self.load_reference_solution()

        h = self.ref_sol['h']
        self.discretize_domain(h)

        Nx = self.Nx
        Ny = self.Ny
        N = self.N
        domain_h = self.ref_sol['domain_h'].reshape((N, 2))
        F = self.ref_sol['F'].reshape((N, 1))

        # compute the optimal theta given a basis of ansatz functions
        v = self.ansatz.basis_value_f(domain_h)
        self.theta, _, _, _ = np.linalg.lstsq(v, F, rcond=None)
        self.theta_type = 'optimal'

    def set_theta_null(self):
        assert self.ansatz is not None, ''
        m = self.ansatz.m
        self.theta = np.zeros(m)
        self.theta_type = 'null'

    def set_theta_from_metadynamics(self):
        '''
        '''
        x = self.domain_h

        self.load_meta_bias_potential()
        meta_theta = self.meta_bias_pot['omegas'] / 2
        meta_mus = self.meta_bias_pot['mus']
        meta_sigmas = self.meta_bias_pot['sigmas']
        assert meta_theta.shape == meta_mus.shape == meta_sigmas.shape, ''

        # create ansatz functions from meta
        meta_ansatz = GaussianAnsatz(domain=self.domain)
        meta_ansatz.set_given_ansatz_functions(meta_mus, meta_sigmas)

        # meta value function evaluated at the grid
        value_f_meta = self.value_function(x, meta_theta, meta_ansatz)

        # ansatz functions evaluated at the grid
        v = self.ansatz.basis_value_f(x)

        # solve theta V = \Phi
        self.theta, _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)
        self.theta_type = 'meta'

    def set_theta_from_gd(self, gd_type, gd_theta_init, gd_lr):
        '''
        '''
        assert self.ansatz, ''

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
        self.dir_path = os.path.join(gd_dir_path, 'drifted-sampling')
        make_dir_path(self.dir_path)

    def value_function(self, x, theta=None, ansatz=None):
        '''This method computes the value function evaluated at x

        Args:
            x ((M,2)-array) : position
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

        # x in the target set
        is_in_target_set = (
            (x[:, 0] >= target_set_x_min) &
            (x[:, 0] <= target_set_x_max) &
            (x[:, 1] >= target_set_y_min) &
            (x[:, 1] <= target_set_y_max)
        )
        idx_ts = np.where(is_in_target_set == True)[0]

        # impose value function in the target set is null
        K = - np.mean(value_f[idx_ts])

        return value_f + K


    def control(self, x, theta=None, ansatz=None):
        '''This method computes the control evaluated at x

        Args:
            x ((M,2)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions
        '''
        M = x.shape[0]
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz

        basis_control = ansatz.basis_control(x)
        control = np.empty((M, 2))
        control[:, 0] = np.dot(basis_control[:, :, 0], theta).reshape((M,))
        control[:, 1] = np.dot(basis_control[:, :, 1], theta).reshape((M,))

        return control

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x ((M,2)-array) : position
            theta ((m,)-array): parameters
        '''
        return 2 * self.value_function(x, theta)

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u ((M,2)-array) : control at x
        '''
        return - np.sqrt(2) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x ((M,2)-array) : position/s
            theta ((m,)-array): parameters
        '''
        return self.potential(x) + self.bias_potential(x, theta)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x ((M,2)-array) : position/s
            u ((M,2)-array) : control at x
        '''
        assert x.shape == u.shape

        return self.gradient(x) + self.bias_gradient(u)

    def set_sampling_parameters(self, xzero, M, dt, N_lim, seed=None):
        '''
        '''
        # set random seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.xzero = xzero
        self.M = M

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
        assert self.M is not None, ''
        M = self.M
        self.been_in_target_set = np.repeat([False], M)
        self.fht = np.empty(M)

    def initialize_girsanov_martingale_terms(self):
        '''
        '''
        assert self.M is not None, ''
        M = self.M

        self.M1_fht = np.empty(M)
        self.M2_fht = np.empty(M)
        self.M1_k = np.empty((M, 10))
        self.M2_k = np.empty((M, 10))

    def sde_update(self, x, gradient, dB):
        beta = self.beta
        dt = self.dt

        drift = - gradient * dt
        diffusion = np.dot(dB, np.sqrt(2 / beta) * np.eye(2))
        return x + drift + diffusion

    def get_idx_new_in_target_set(self, x, been_in_target_set):
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
            (been_in_target_set == False)
        )[0]

        # update list of indices whose trajectories have been in the target set
        been_in_target_set[idx_new] = True

        return idx_new

    def sample_not_drifted(self):
        self.start_timer()
        self.initialize_fht()

        dt = self.dt
        N_lim = self.N_lim
        xzero = self.xzero
        M = self.M
        been_in_target_set = self.been_in_target_set

        # initialize Xtemp
        xtemp = np.ones((M, 2))
        xtemp[:, 0] *= xzero[0]
        xtemp[:, 1] *= xzero[1]

        for n in np.arange(1, N_lim +1):
            # Brownian increment
            dB = (np.sqrt(dt) * np.random.normal(0, 1, 2 * M)).reshape((M, 2))

            # compute gradient
            gradient = self.gradient(xtemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # get indices from the trajectories which are new in target
            idx_new = self.get_idx_new_in_target_set(xtemp, been_in_target_set)

            # save first hitting time
            self.fht[idx_new] = n * dt

            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break

        self.been_in_target_set = been_in_target_set
        self.compute_fht_statistics()
        self.compute_I_statistics()
        self.stop_timer()

    def sample_drifted(self):
        self.start_timer()
        self.initialize_fht()
        self.initialize_girsanov_martingale_terms()

        beta = self.beta
        dt = self.dt
        N_lim = self.N_lim
        xzero = self.xzero
        M = self.M
        been_in_target_set = self.been_in_target_set

        # initialize xtemp
        xtemp = np.ones((M, 2))
        xtemp[:, 0] *= xzero[0]
        xtemp[:, 1] *= xzero[1]

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(M)
        M2temp = np.zeros(M)
        k = np.array([])

        for n in np.arange(1, N_lim +1):
            # Brownian increment
            dB = (np.sqrt(dt) * np.random.normal(0, 1, 2 * M)).reshape((M, 2))

            # control at Xtemp
            utemp = self.control(xtemp)

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # Girsanov Martingale terms
            M1temp -= np.sqrt(beta) * np.matmul(utemp, dB.T).diagonal()
            M2temp -= beta * 0.5 * (np.linalg.norm(utemp, axis=1) ** 2) * dt

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp, been_in_target_set)

            # save first hitting time and Girsanov Martingale terms
            self.fht[idx_new] = n * dt
            self.M1_fht[idx_new] = M1temp[idx_new]
            self.M2_fht[idx_new] = M2temp[idx_new]

            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break

        self.been_in_target_set = been_in_target_set
        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()

    def sample_loss(self):
        self.initialize_fht()

        alpha = self.alpha
        beta = self.beta
        dt = self.dt
        N_lim = self.N_lim
        xzero = self.xzero
        M = self.M
        m = self.ansatz.m
        been_in_target_set = self.been_in_target_set

        # initialize statistics 
        J = np.zeros(M)
        grad_J = np.zeros((M, m))

        # initialize xtemp
        xtemp = np.ones((M, 2))
        xtemp[:, 0] *= xzero[0]
        xtemp[:, 1] *= xzero[1]

        # initialize ipa variables
        cost_temp = np.zeros(M)
        grad_phi_temp = np.zeros((M, m))
        grad_S_temp = np.zeros((M, m))

        for n in np.arange(1, N_lim+1):
            # Brownian increment
            dB = (np.sqrt(dt) * np.random.normal(0, 1, 2 * M)).reshape((M, 2))

            # control
            btemp = self.ansatz.basis_control(xtemp)
            utemp = self.control(xtemp)
            lower_bound = -10 * np.ones(2)
            upper_bound = 10 * np.ones(2)
            if not is_2d_valid_control(utemp, lower_bound, upper_bound):
                return False, None, None

            # ipa statistics 
            normed_utemp = np.linalg.norm(utemp, axis=1)
            normed_btemp = np.linalg.norm(btemp, axis=2)
            normed_dB = np.linalg.norm(dB, axis=1)
            cost_temp += 0.5 * (normed_utemp ** 2) * dt
            grad_phi_temp += (normed_utemp * normed_btemp.T * dt).T
            grad_S_temp -= (np.sqrt(beta) * normed_btemp.T * normed_dB).T

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp, been_in_target_set)

            # save ipa statistics
            J[idx_new] = n * dt + cost_temp[idx_new]
            grad_J[idx_new, :] = grad_phi_temp[idx_new, :] \
                               - ((n * dt + cost_temp[idx_new]) \
                               * grad_S_temp[idx_new, :].T).T

            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break

        # compute averages
        mean_J = np.mean(J)
        mean_grad_J = np.mean(grad_J, axis=0)

        return True, mean_J, mean_grad_J


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
        self.M_arrived = fht[idx_arrived].shape[0]
        if self.M_arrived != self.M:
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
        beta = self.beta
        fht = self.fht

        # compute mean and variance of I
        I = np.exp(-beta * fht)
        self.mean_I, \
        self.var_I, \
        self.re_I = self.compute_mean_variance_and_rel_error(I)

    def compute_I_u_statistics(self):
        beta = self.beta
        fht = self.fht
        M1_fht = self.M1_fht
        M2_fht = self.M2_fht

        # compute mean of M_fht
        M_fht = np.exp(M1_fht + M2_fht)

        # compute mean and variance of I_u
        I_u = np.exp(-beta * fht) * M_fht
        self.mean_I_u, \
        self.var_I_u, \
        self.re_I_u = self.compute_mean_variance_and_rel_error(I_u)

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
        f.write('xzero: ({:2.1f}, {:2.1f})\n'.format(self.xzero[0], self.xzero[1]))
        f.write('target set: [[{:2.1f}, {:2.1f}], [{:2.1f}, {:2.1f}]]\n'.format(
            self.target_set[0, 0],
            self.target_set[0, 1],
            self.target_set[1, 0],
            self.target_set[1, 1],
        ))
        f.write('sampled trajectories: {:,d}\n\n'.format(self.M))

    def write_report(self):
        '''
        '''
        # set file path
        if self.is_drifted:
            theta_stamp = 'theta-{}_'.format(self.theta_type)
        else:
            theta_stamp = ''

        trajectories_stamp = 'M{:.0e}'.format(self.M)
        file_name = 'report_2d_' + theta_stamp + trajectories_stamp + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        self.write_sde_parameters(f)
        self.write_euler_maruyama_parameters(f)
        self.write_sampling_parameters(f)

        if self.is_drifted:
            self.ansatz.write_ansatz_parameters(f)

        f.write('Statistics\n\n')

        f.write('trajectories which arrived: {:2.2f} %\n'
                ''.format(100 * self.M_arrived / self.M))
        f.write('used time steps: {:,d}\n\n'.format(int(self.last_fht / self.dt)))
        if self.M_arrived == 0:
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

    def plot_appr_mgf(self, file_name, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path

        beta = self.beta
        h = self.h
        Nx = self.Nx
        Ny = self.Ny
        N = self.N
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape((N, 2))
        Vbias = self.bias_potential(x).reshape((Nx, Ny))
        appr_F = Vbias / 2
        appr_Psi = np.exp(- beta * appr_F)

        # surface plot
        plt2d = Plot2d(dir_path, file_name + '_surface')
        plt2d.set_title(r'$\Psi(x, y)$')
        plt2d.surface(X, Y, appr_Psi)

        # contour plot
        plt2d = Plot2d(dir_path, file_name + '_contour')
        plt2d.set_title(r'$\Psi(x, y)$')
        plt2d.contour(X, Y, appr_Psi)

    def plot_appr_free_energy(self, file_name, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path

        h = self.h
        Nx = self.Nx
        Ny = self.Ny
        N = self.N
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape((N, 2))
        Vbias = self.bias_potential(x).reshape((Nx, Ny))
        appr_F = Vbias / 2

        # surface plot
        plt2d = Plot2d(dir_path, file_name + '_surface')
        plt2d.set_title(r'$F(x, y)$')
        plt2d.surface(X, Y, appr_F)

        # contour plot
        plt2d = Plot2d(dir_path, file_name + '_contour')
        plt2d.set_title(r'$F(x, y)$')
        plt2d.contour(X, Y, appr_F)


    def plot_control(self, file_name, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path

        h = self.h
        Nx = self.Nx
        Ny = self.Ny
        N = self.N
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape((N, 2))
        u = self.control(x).reshape((Nx, Ny, 2))
        u_x = u[:, :, 0]
        u_y = u[:, :, 1]

        # show every k arrow
        k = int(X.shape[0] / 20)
        X = X[::k, ::k]
        Y = Y[::k, ::k]
        U = u_x[::k, ::k]
        V = u_y[::k, ::k]

        #gradient plot
        plt2d = Plot2d(dir_path, file_name)
        plt2d.set_title(r'$u(x, y)$')
        plt2d.vector_field(X, Y, U, V)

    def plot_tilted_potential(self, file_name, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path
        h = self.h
        Nx = self.Nx
        Ny = self.Ny
        N = self.N
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape((N, 2))
        V = self.potential(x).reshape((Nx, Ny))
        if self.is_drifted:
            Vbias = self.bias_potential(x).reshape((Nx, Ny))
        else:
            Vbias = np.zeros((Nx, Ny))

        # surface plot
        vmin = 0
        vmax = 10
        plt2d = Plot2d(dir_path, file_name + '_surface')
        plt2d.set_title(r'$\tilde{V}(x, y)$')
        plt2d.surface(X, Y, V + Vbias, vmin, vmax)

        # contour plot
        levels = np.logspace(-2, 1, 20, endpoint=True)
        plt2d = Plot2d(dir_path, file_name + '_contour')
        plt2d.set_title(r'$\tilde{V}(x, y)$')
        plt2d.contour(X, Y, V + Vbias, vmin, vmax, levels)

    def plot_tilted_drift(self, file_name, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path
        h = self.h
        Nx = self.Nx
        Ny = self.Ny
        N = self.N
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        x = self.domain_h.reshape((N, 2))
        dV = self.gradient(x).reshape((Nx, Ny, 2))
        if self.is_drifted:
            u = self.control(x).reshape((Nx, Ny, 2))
            dVbias = self.bias_gradient(u)
        else:
            dVbias = np.zeros((Nx, Ny, 2))
        U = - dV[:, :, 0] - dVbias[:, :, 0]
        V = - dV[:, :, 1] - dVbias[:, :, 1]

        # show every k arrow
        k = int(X.shape[0] / 20)
        X = X[::k, ::k]
        Y = Y[::k, ::k]
        U = U[::k, ::k]
        V = V[::k, ::k]

        #gradient plot
        plt2d = Plot2d(dir_path, file_name)
        plt2d.set_title(r'$-\nabla \tilde{V}(x, y)$')
        plt2d.vector_field(X, Y, U, V)
