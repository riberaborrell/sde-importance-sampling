from mds.gaussian_1d_ansatz_functions import GaussianAnsatz
from mds.potentials_and_gradients_1d import get_potential_and_gradient
from mds.plots_1d import Plot1d
from mds.utils import get_example_data_path, get_gd_data_path, get_time_in_hms, make_dir_path
from mds.validation import is_1d_valid_interval, is_1d_valid_target_set, is_1d_valid_control

import numpy as np
import time
import os

class Sampling:
    '''
    '''

    def __init__(self, potential_name, alpha, beta,
                 target_set, domain=None, h=0.001, is_drifted=False):
        '''
        '''
        # get potential and gradient functions
        potential, gradient, _, _, _ = get_potential_and_gradient(potential_name, alpha)

        # validate domain and target set
        if domain is None:
            domain = np.array([-3, 3])
        is_1d_valid_interval(domain)
        is_1d_valid_target_set(domain, target_set)

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

        # dir_path
        self.example_dir_path = None
        self.dir_path = None
        self.set_example_dir_path()

    def discretize_domain(self, h=None):
        ''' this method discretizes the domain interval uniformly with step-size h
        Args:
            h (float): step-size
        '''
        if h is None:
            h = self.h

        d_min, d_max = self.domain
        self.domain_h = np.around(np.arange(d_min, d_max + h, h), decimals=3)
        self.N = self.domain_h.shape[0]

    def set_example_dir_path(self):
        self.example_dir_path = get_example_data_path(self.potential_name, self.alpha,
                                                      self.beta, self.target_set)

    def set_not_drifted_dir_path(self):
        self.dir_path = os.path.join(self.example_dir_path, 'not-drifted-sampling')
        make_dir_path(self.dir_path)

    def set_drifted_dir_path(self, dir_path):
        self.dir_path = dir_path
        make_dir_path(dir_path)

    def set_gaussian_ansatz_functions(self, m, sigma=None):
        '''
        '''
        assert self.is_drifted, ''

        # set gaussian ansatz functions
        ansatz = GaussianAnsatz(
            domain=self.domain,
        )
        ansatz.set_unif_dist_ansatz_functions(m, sigma)

        # set ansatz dir path
        ansatz.set_dir_path(self.example_dir_path)
        self.ansatz = ansatz

    def set_bias_potential(self, theta, mus, sigmas):
        ''' set the gaussian ansatz functions and the coefficients theta
        Args:
            theta ((m,)-array): parameters
            mus ((m,)-array): mean of each gaussian
            sigmas ((m,)-array) : standard deviation of each gaussian
        '''
        assert self.is_drifted, ''
        assert theta.shape == mus.shape == sigmas.shape, ''

        # set gaussian ansatz functions
        ansatz = GaussianAnsatz(domain=self.domain)
        ansatz.set_given_ansatz_functions(mus, sigmas)

        self.ansatz = ansatz
        self.theta = theta

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
            file_path = os.path.join(
                self.example_dir_path,
                'reference_solution',
                'reference_solution.npz',
            )
            self.ref_sol = np.load(file_path)

    def get_value_f_at_xzero(self):
        # load ref sol
        self.load_reference_solution()
        x = self.ref_sol['domain_h']
        F = self.ref_sol['F']

        # evaluate F at xzero
        idx = np.where(x == self.xzero)[0][0]
        self.value_f_at_xzero = F[idx]

    def set_theta_optimal(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.load_reference_solution()
        x = self.ref_sol['domain_h']
        F = self.ref_sol['F']

        # compute the optimal theta given a basis of ansatz functions
        v = self.ansatz.basis_value_f(x)
        self.theta, _, _, _ = np.linalg.lstsq(v, F, rcond=None)
        self.theta_type = 'optimal'

        # set drifted sampling dir path
        dir_path = os.path.join(self.ansatz.dir_path, 'optimal-importance-sampling')
        self.set_drifted_dir_path(dir_path)

    def set_theta_null(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.theta = np.zeros(self.ansatz.m)
        self.theta_type = 'null'

        # set drifted sampling dir path
        dir_path = os.path.join(self.ansatz.dir_path, 'null-importance-sampling')
        self.set_drifted_dir_path(dir_path)

    def set_theta_from_metadynamics(self):
        '''
        '''
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        x = self.domain_h

        self.load_meta_bias_potential()
        meta_theta = self.meta_bias_pot['theta']
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

        # set drifted sampling dir path
        dir_path = os.path.join(self.ansatz.dir_path, 'meta-importance-sampling')
        self.set_drifted_dir_path(dir_path)

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
        self.set_drifted_dir_path(dir_path)

    def value_function(self, x, theta=None, ansatz=None):
        '''This method computes the value function evaluated at x

        Args:
            x ((M,)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions

        Return:
        '''
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz
        assert theta.shape == ansatz.mus.shape == ansatz.sigmas.shape, ''

        # value function with constant K=0
        value_f =  np.dot(ansatz.basis_value_f(x), theta)

        # compute K
        target_set_min, target_set_max = self.target_set
        idx_ts = np.where((x >= target_set_min) & (x <= target_set_max))[0]
        K = - np.mean(value_f[idx_ts])
        #K = - value_f[idx_ts[0]]

        return value_f + K

    def control(self, x, theta=None, ansatz=None):
        '''This method computes the control evaluated at x

        Args:
            x ((M,)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions
        '''
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz
        assert theta.shape == ansatz.mus.shape == ansatz.sigmas.shape, ''

        return np.dot(ansatz.basis_control(x), theta)

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x ((M,)-array) : position
            theta ((m,)-array): parameters
        '''
        return 2 * self.value_function(x, theta)

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u ((M,)-array) : control at x
        '''
        return - np.sqrt(2) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x ((M,)-array) : position
            theta ((m,)-array): parameters
        '''
        return self.potential(x) + self.bias_potential(x, theta)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x ((M,)-array) : position
            u ((M,)-array) : control at x
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
        diffusion = np.sqrt(2 / beta) * dB
        return x + drift + diffusion

    def get_idx_new_in_target_set(self, x):
        target_set_min, target_set_max = self.target_set

        # trajectories in the target set
        is_in_target_set = ((x >= target_set_min) & (x <= target_set_max))

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
        xtemp = np.full(self.M, self.xzero)

        for n in np.arange(1, self.N_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, self.M)

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
        xtemp = np.full(self.M, self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(self.M)
        M2temp = np.zeros(self.M)
        k = np.array([])

        for n in np.arange(1, self.N_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, self.M)

            # control at Xtemp
            utemp = self.control(xtemp)

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # Girsanov Martingale terms
            M1temp -= np.sqrt(self.beta) * utemp * dB
            M2temp -= self.beta * 0.5 * (utemp ** 2) * self.dt

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save first hitting time and Girsanov Martingale terms
            self.fht[idx_new] = n * self.dt
            self.M1_fht[idx_new] = M1temp[idx_new]
            self.M2_fht[idx_new] = M2temp[idx_new]

            if n % 1000 == 0:
                k = np.append(k, n)
                self.M1_k[:, k.shape[0]-1] = M1temp
                self.M2_k[:, k.shape[0]-1] = M2temp

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                # save Girsanov Martingale at the time when 
                # the last trajectory arrive
                k = np.append(k, n)
                self.M1_k[:, k.shape[0]-1] = M1temp
                self.M2_k[:, k.shape[0]-1] = M2temp
                self.k = k
                break

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()

    def sample_meta(self):
        self.initialize_fht()

        # initialize xtemp
        xtemp = np.empty((self.N_lim + 1, self.M))
        xtemp[0, :] = np.full(self.M, self.xzero)

        for n in np.arange(self.N_lim):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, self.M)

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
            if np.sum(self.been_in_target_set) >= self.M / 3:
                return True, xtemp[:n+1]

        return False, xtemp


    def sample_loss(self):
        self.initialize_fht()

        # number of ansatz functions
        m = self.ansatz.m

        # initialize loss and its gradient 
        J = np.zeros(self.M)
        grad_J = np.zeros((self.M, m))

        # initialize xtemp
        xtemp = np.full(self.M, self.xzero)

        # initialize temp variables
        cost_temp = np.zeros(self.M)
        grad_phi_temp = np.zeros((self.M, m))
        grad_S_temp = np.zeros((self.M, m))

        for n in np.arange(1, self.N_lim+1):
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.normal(0, 1, self.M)

            # control
            btemp = self.ansatz.basis_control(xtemp)
            utemp = self.control(xtemp)
            if not is_1d_valid_control(utemp, -self.alpha * 10, self.alpha * 10):
                return False, None, None, None

            # ipa statistics 
            cost_temp += 0.5 * (utemp ** 2) * self.dt
            grad_phi_temp += (utemp * btemp.T * self.dt).T
            grad_S_temp -= (np.sqrt(self.beta) * btemp.T * dB).T

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save first hitting time
            self.fht[idx_new] = n * self.dt

            # save ipa statistics
            J[idx_new] = n * self.dt + cost_temp[idx_new]
            grad_J[idx_new, :] = grad_phi_temp[idx_new, :] \
                               - ((n * self.dt + cost_temp[idx_new]) \
                               * grad_S_temp[idx_new, :].T).T

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
        self.M_arrived = fht[idx_arrived].shape[0]
        if self.M_arrived == 0:
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

        #TODO: implement method to stabilize the exponential mean
        #k = np.max(-self.beta * fht)
        #a = np.mean(np.exp(-self.beta * fht -k))
        #print('{:2.3e}'.format(np.exp(k) * a))

    def compute_I_u_statistics(self):
        beta = self.beta
        fht = self.fht
        been_in_target_set = self.been_in_target_set
        M1_fht = self.M1_fht
        M2_fht = self.M2_fht
        M1_k = self.M1_k
        M2_k = self.M2_k

        # compute mean of M_k
        M1_k = M1_k[:, :self.k.shape[0]]
        M2_k = M2_k[:, :self.k.shape[0]]
        M_k = np.exp(M1_k + M2_k)
        self.mean_M_k = np.mean(M_k, axis=0)

        # compute mean of M_fht
        idx_not_arrived = np.where(been_in_target_set == False)
        M1_fht[idx_not_arrived] = M1_k[idx_not_arrived, -1]
        M2_fht[idx_not_arrived] = M2_k[idx_not_arrived, -1]
        M_fht = np.exp(M1_fht + M2_fht)

        #TODO: implement method to stabilize the exponential mean
        #print('{:2.3e}'.format(np.mean(self.M1_fht + self.M2_fht)))
        #k = np.max(self.M1_fht + self.M2_fht)
        #a = np.mean(np.exp(self.M1_fht + self.M2_fht -k))
        #l = np.min(self.M1_fht + self.M2_fht)
        #b = np.mean(np.exp(self.M1_fht + self.M2_fht -l))
        #print('{:2.3e}'.format(np.exp(k) * a))
        #print('{:2.3e}'.format(np.exp(l) * b))

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
        f.write('xzero: {:2.1f}\n'.format(self.xzero))
        f.write('target set: [{:2.1f}, {:2.1f}]\n'
                ''.format(self.target_set[0], self.target_set[1]))
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
        file_name = 'report_' + theta_stamp + trajectories_stamp + '.txt'
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
            f.write('Girsanov Martingale\n')
            for i, n in enumerate(self.k):
                f.write('E[M_k] = {:2.3e}, k = {:d}\n'.format(self.mean_M_k[i], int(n)))

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

    def plot_appr_psi(self, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path
        file_name = 'appr_mgf' + ext

        Vbias = self.bias_potential(self.domain_h)
        appr_F = Vbias / 2
        appr_Psi = np.exp(- self.beta * appr_F)

        self.load_reference_solution()
        Psi = self.ref_sol['Psi']

        plt1d = Plot1d(dir_path, file_name)
        plt1d.set_ylim(bottom=0, top=self.alpha * 2)
        plt1d.mgf(self.domain_h, Psi, appr_Psi)

    def plot_appr_free_energy(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'appr_free_energy'
        if dir_path is None:
            dir_path = self.dir_path

        Vbias = self.bias_potential(self.domain_h)
        appr_F = Vbias / 2

        self.load_reference_solution()
        F = self.ref_sol['F']

        plt1d = Plot1d(dir_path, file_name)
        plt1d.set_ylim(bottom=0, top=self.alpha * 3)
        plt1d.free_energy(self.domain_h, F, appr_F)

    def plot_control(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'control'
        if dir_path is None:
            dir_path = self.dir_path

        u = self.control(self.domain_h)

        self.load_reference_solution()
        u_opt = self.ref_sol['u_opt']

        plt1d = Plot1d(dir_path, file_name)
        plt1d.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        plt1d.control(self.domain_h, u_opt, u)

    def plot_potential_and_tilted_potential(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'potential_and_tilted_potential'
        if dir_path is None:
            dir_path = self.dir_path

        V = self.potential(self.domain_h)

        if self.is_drifted:
            Vb = self.bias_potential(self.domain_h)
        else:
            Vb = np.zeros(self.domain_h.shape[0])

        self.load_reference_solution()
        F = self.ref_sol['F']
        Vb_opt = 2 * F

        plt1d = Plot1d(dir_path, file_name)
        plt1d.set_ylim(bottom=0, top=self.alpha * 10)
        plt1d.potential_and_tilted_potential(self.domain_h, V, Vb, Vb_opt)

    def plot_tilted_potential(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'tilted_potential'
        if dir_path is None:
            dir_path = self.dir_path

        V = self.potential(self.domain_h)

        if self.is_drifted:
            Vb = self.bias_potential(self.domain_h)
        else:
            Vb = np.zeros(self.domain_h.shape[0])

        self.load_reference_solution()
        F = self.ref_sol['F']
        Vb_opt = 2 * F

        plt1d = Plot1d(dir_path, file_name)
        plt1d.set_ylim(bottom=0, top=self.alpha * 10)
        plt1d.tilted_potential(self.domain_h, V, Vb, Vb_opt)

    def plot_tilted_drift(self, file_name=None, dir_path=None):
        if file_name is None:
            file_name = 'tilted_drift'
        if dir_path is None:
            dir_path = self.dir_path

        dV = self.gradient(self.domain_h)

        if self.is_drifted:
            U = self.control(self.domain_h)
            dVb = self.bias_gradient(U)
        else:
            dVb = np.zeros(self.domain_h.shape[0])

        self.load_reference_solution()
        u_opt = self.ref_sol['u_opt']
        dVb_opt = - np.sqrt(2) * u_opt

        plt1d = Plot1d(dir_path, file_name)
        plt1d.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        plt1d.drift_and_tilted_drift(self.domain_h, dV, dVb, dVb_opt)
