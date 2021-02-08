from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.potentials_and_gradients_nd import get_potential_and_gradient
from mds.utils import get_example_dir_path, \
                      get_hjb_solution_dir_path, \
                      get_metadynamics_dir_path, \
                      get_gd_dir_path, \
                      get_time_in_hms, \
                      make_dir_path

import numpy as np
import time
import os

class Sampling:
    '''
    '''

    def __init__(self, sde, is_controlled=None, is_optimal=None):
        '''
        '''

        #sde
        self.sde = sde

        # sampling
        self.is_controlled = is_controlled
        self.is_optimal = is_optimal
        self.seed = None
        self.xzero = None
        self.N = None
        self.xtemp = None
        self.save_trajectory = False
        self.traj = None

        # Euler-Marujama
        self.dt = None
        self.k_lim = None

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

        # first hitting time step (fhts)
        self.fhts = None
        self.mean_fhts = None
        self.var_fhts = None
        self.re_fhts = None

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
        self.hjb_sol = None
        self.value_f_at_xzero = None

        # metadynamics
        self.meta_bias_pot = None

        # dir_path
        self.dir_path = None

    def set_dir_path(self, dir_path):
        self.dir_path = dir_path
        make_dir_path(self.dir_path)

    def set_gaussian_ansatz_uniformly(self, m_i, sigma_i):
        '''
        '''
        assert self.is_drifted, ''

        # initialize Gaussian ansatz
        self.ansatz = GaussianAnsatz(self.n, self.domain)

        # set gaussian ansatz functions
        self.ansatz.set_unif_dist_ansatz_functions(m_i, sigma_i)

        # set ansatz dir path
        self.ansatz.set_dir_path(self.example_dir_path)

    def set_gaussian_ansatz_from_meta(self, sigma_i, k, N_meta):
        '''
        '''
        assert self.is_drifted, ''

        self.load_meta_bias_potential(sigma_i, k, N_meta)
        meta_ms = self.meta_bias_pot['ms']
        meta_total_m = int(np.sum(meta_ms))
        meta_means = self.meta_bias_pot['means']
        meta_cov = self.meta_bias_pot['cov']
        meta_thetas = self.meta_bias_pot['thetas']
        assert N_meta == meta_ms.shape[0], ''

        # get the centers used for each trajectory
        means = np.empty((meta_total_m, self.n))
        theta = np.empty(meta_total_m)
        flatten_idx = 0
        for i in np.arange(N_meta):
            means[flatten_idx:flatten_idx+meta_ms[i]] = meta_means[i, :meta_ms[i]]
            theta[flatten_idx:flatten_idx+meta_ms[i]] = meta_thetas[i, :meta_ms[i]]
            flatten_idx += meta_ms[i]

        # initialize Gaussian ansatz
        self.ansatz = GaussianAnsatz(self.n, self.domain)
        self.ansatz.set_given_ansatz_functions(means, meta_cov)
        self.ansatz.are_meta_distributed = True
        self.ansatz.sigma_i = sigma_i
        self.ansatz.k = k
        self.ansatz.N_meta = N_meta
        self.theta = theta
        self.theta_type = 'meta'

        # set ansatz dir path
        self.ansatz.set_dir_path(self.example_dir_path)

    #TODO: generalize for arbitrary n
    def set_bias_potential(self, theta, means, covs):
        ''' set the gaussian ansatz functions and the coefficients theta
        Args:
            theta ((m,)-array): parameters
            means ((m, 2)-array): mean of each gaussian
            covs ((m, 2, 2)-array) : covaraince matrix of each gaussian
        '''
        assert self.is_drifted, ''
        assert theta.shape[0] == means.shape[0] == covs.shape[0], ''

        # set gaussian ansatz functions
        ansatz = GaussianAnsatz(domain=self.domain)
        ansatz.set_given_ansatz_functions(mus, sigmas)

        self.ansatz = ansatz
        self.theta = theta

    def get_idx_discretized_domain(self, x):
        assert x.ndim == 2, ''
        assert x.shape == (self.N, self.n), ''

        # get index of xzero
        idx = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], self.hjb_sol['Nx'][i])
            idx[i] = tuple(np.argmin(np.abs(axis_i - x[:, i].reshape(self.N, 1)), axis=1))

        idx = tuple(idx)
        return idx

    def load_meta_bias_potential(self, sigma_i, k, meta_N):
        if not self.meta_bias_pot:
            meta_dir_path = get_metadynamics_dir_path(self.example_dir_path, sigma_i, k, meta_N)
            file_path = os.path.join(meta_dir_path, 'bias_potential.npz')
            self.meta_bias_pot = np.load(file_path)

    def load_hjb_solution(self, h=None):
        if self.hjb_sol:
            return True

        if h is None:
            h = self.h

        try:
            hjb_dir_path = get_hjb_solution_dir_path(self.example_dir_path, h)
            self.hjb_sol = np.load(
                os.path.join(hjb_dir_path, 'hjb-solution.npz'),
                allow_pickle=True,
            )
            return True
        except:
            print('no hjb-solution found with h={:.0e}'.format(h))
            return False

    def get_value_f_at_xzero(self, h=None):
        # load ref sol
        succ = self.load_hjb_solution(h)
        if not succ:
            return

        domain_h = self.hjb_sol['domain_h']
        Nx = domain_h.shape[1:]
        F = self.hjb_sol['F']

        # get index of xzero
        idx_xzero = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], Nx[i])
            idx_xzero[i] = np.argmin(np.abs(axis_i - self.xzero[i]))

        idx_xzero = tuple(idx_xzero)

        # evaluate F at xzero
        self.value_f_at_xzero = F[idx_xzero]

    def set_theta_optimal(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.load_hjb_solution()
        Nx = self.hjb_sol['domain_h'].shape[:-1]
        Nh = self.hjb_sol['Nh']
        x = self.hjb_sol['domain_h'].reshape(Nh, self.n)
        F = self.hjb_sol['F'].reshape(Nh,)

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

    def set_theta_from_metadynamics(self, sigma_i, k, meta_N):
        '''
        '''
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        if not self.ansatz.are_meta_distributed:
            # discretize domain
            self.discretize_domain()
            x = self.domain_h.reshape(self.Nh, self.n)

            self.load_meta_bias_potential(sigma_i, k, meta_N)
            meta_ms = self.meta_bias_pot['ms']
            meta_N = meta_ms.shape[0]
            meta_total_m = int(np.sum(meta_ms))
            meta_means = self.meta_bias_pot['means']
            meta_cov = self.meta_bias_pot['cov']
            meta_thetas = self.meta_bias_pot['thetas']

            thetas = np.empty((meta_N, self.ansatz.m))

            for i in np.arange(meta_N):
                # create ansatz functions from meta
                meta_ansatz = GaussianAnsatz(self.n, self.domain)
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

    #TODO: generalize for arbitrary n
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

    #TODO: generalize for arbitrary n
    def value_function(self, x, theta=None, ansatz=None):
        '''This method computes the value function evaluated at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions

        Return:
        '''
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz

        # value function with constant K=0
        basis_value_f =  ansatz.basis_value_f(x)
        value_f =  np.dot(basis_value_f, theta)

        # assume they are in the target set
        N = x.shape[0]
        is_in_target_set = np.repeat([True], N)

        for i in range(self.n):
            is_not_in_target_set_i_axis_idx = np.where(
                (x[:, i] < self.target_set[i, 0]) |
                (x[:, i] > self.target_set[i, 1])
            )[0]
            # if they are NOT in the target set change flag
            is_in_target_set[is_not_in_target_set_i_axis_idx] = False

           # break loop for the dimensions if all positions are switched to False
            if is_in_target_set.all() == False:
                break

        # compute constant
        idx_ts = np.where(is_in_target_set == True)[0]

        # impose value function in the target set to be null
        K = - np.mean(value_f[idx_ts])

        # get idx for x to be in (1, ..., 1)
        #idx_corner_ts = np.where(
        #    (x[:, 0] == 1) &
        #    (x[:, 1] == 1)
        #)[0]

        # impose value function in (1, 1) to be null
        #L = - value_f[idx_corner_ts]

        return value_f + K


    def control(self, x, theta=None, ansatz=None):
        '''This method computes the control evaluated at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters
            ansatz (object): ansatz functions
        '''
        N = x.shape[0]
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz

        basis_control = ansatz.basis_control(x)
        control = np.tensordot(basis_control, theta, axes=([1], [0]))
        #control = np.empty((N, self.n))
        #for i in range(self.n):
        #    control[:, i] = np.dot(basis_control[:, :, i], theta).reshape((N,))

        return control

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters
        '''
        return 2 * self.value_function(x, theta)

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u ((N, n)-array) : control at x
        '''
        return - np.sqrt(2) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x ((N, n)-array) : position/s
            theta ((m,)-array): parameters
        '''
        return self.potential(x) + self.bias_potential(x, theta)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x ((N, n)-array) : position/s
            u ((N, n)-array) : control at x
        '''
        assert x.shape == u.shape

        return self.gradient(x) + self.bias_gradient(u)

    def set_sampling_parameters(self, xzero, N, dt, k_lim, seed=None):
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
        self.k_lim = k_lim

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

    def sde_update(self, x, gradient, dB):
        drift = - gradient * self.dt
        diffusion = np.dot(dB, np.sqrt(2 / self.sde.beta) * np.eye(self.sde.n))
        return x + drift + diffusion

    def get_idx_new_in_target_set(self, x):
        # assume trajectories are in the target set
        is_in_target_set = np.repeat([True], self.N)
        for i in range(self.sde.n):
            is_not_in_target_set_i_axis_idx = np.where(
                (x[:, i] < self.sde.target_set[i, 0]) |
                (x[:, i] > self.sde.target_set[i, 1])
            )[0]
            # if they are NOT in the target set change flag
            is_in_target_set[is_not_in_target_set_i_axis_idx] = False

            # break if none of them is in the target set
            if is_in_target_set.any() == False:
                break

        # indices of trajectories new in the target set
        idx_new = np.where(
            (is_in_target_set == True) &
            (self.been_in_target_set == False)
        )[0]

        # update list of indices whose trajectories have been in the target set
        self.been_in_target_set[idx_new] = True

        return idx_new

    def sample_not_controlled(self):
        self.start_timer()
        self.initialize_fht()

        # initialize xtemp
        xtemp = np.full((self.N, self.sde.n), self.xzero)

        if self.save_trajectory:
            self.traj = np.empty((self.k_lim + 1, self.sde.n))
            self.traj[0] = xtemp[0, :]

        for k in np.arange(1, self.k_lim + 1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.sde.n).reshape(self.N, self.sde.n)

            # compute gradient
            gradient = self.sde.gradient(xtemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            if self.save_trajectory:
                self.traj[k] = xtemp[0, :]

            # get indices from the trajectories which are new in target
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save first hitting time
            self.fht[idx_new] = k * self.dt

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
        xtemp = np.full((self.N, self.n), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(self.N)
        M2temp = np.zeros(self.N)

        for k in np.arange(1, self.k_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

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
            self.fht[idx_new] = k * self.dt
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
        xtemp = np.full((self.N, self.n), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(self.N)
        M2temp = np.zeros(self.N)

        # load optimal control
        self.load_hjb_solution()
        u_opt = self.hjb_sol['u_opt']

        for k in np.arange(1, self.k_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # control at xtemp
            idx = self.get_idx_discretized_domain(xtemp)
            utemp = u_opt[idx]

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
            self.fht[idx_new] = k * self.dt
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
        xtemp = np.empty((self.k_lim + 1, self.N, self.n))
        xtemp[0] = np.full((self.N, self.n), self.xzero)

        for k in np.arange(1, self.k_lim + 1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            if not self.is_drifted:
                # compute gradient
                gradient = self.gradient(xtemp[k - 1])

            else:
                # control at xtemp
                utemp = self.control(xtemp[k - 1])

                # compute gradient
                gradient = self.tilted_gradient(xtemp[k - 1], utemp)

            # sde update
            xtemp[k] = self.sde_update(xtemp[k - 1], gradient, dB)

            # get indices from the trajectories which are new in target
            idx_new = self.get_idx_new_in_target_set(xtemp[k])

            # check if the half of the trajectories have arrived to the target set
            if np.sum(self.been_in_target_set) >= self.N / 2:
                return True, xtemp[:k]

        return False, xtemp

    def sample_loss(self):
        self.initialize_fht()

        # number of ansatz functions
        m = self.ansatz.m

        # initialize statistics 
        J = np.zeros(self.N)
        grad_J = np.zeros((self.N, m))

        # initialize xtemp
        xtemp = np.full((self.N, self.n), self.xzero)

        # initialize ipa variables
        cost_temp = np.zeros(self.N)
        grad_phi_temp = np.zeros((self.N, m))
        grad_S_temp = np.zeros((self.N, m))

        for k in np.arange(1, self.k_lim+1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # control
            btemp = self.ansatz.basis_control(xtemp)
            utemp = self.control(xtemp)

            # ipa statistics 
            normed_utemp = np.linalg.norm(utemp, axis=1)
            cost_temp += 0.5 * (normed_utemp ** 2) * self.dt
            grad_phi_temp += np.sum(utemp[:, np.newaxis, :] * btemp, axis=2) * self.dt
            grad_S_temp -= np.sqrt(self.beta) * np.sum(dB[:, np.newaxis, :] * btemp, axis=2)

            # compute gradient
            gradient = self.tilted_gradient(xtemp, utemp)

            # sde update
            xtemp = self.sde_update(xtemp, gradient, dB)

            # get indices from the trajectories which are new in target set
            idx_new = self.get_idx_new_in_target_set(xtemp)

            # save ipa statistics
            J[idx_new] = k * self.dt + cost_temp[idx_new]
            grad_J[idx_new, :] = grad_phi_temp[idx_new, :] \
                               - (k * self.dt + cost_temp[idx_new])[:, np.newaxis] \
                               * grad_S_temp[idx_new, :]

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        # compute averages
        mean_J = np.mean(J)
        mean_grad_J = np.mean(grad_J, axis=0)

        return True, mean_J, mean_grad_J, k


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

        # cut saved trajectory
        if self.save_trajectory:
            k_last = int(fht[0] / self.dt)
            self.traj = self.traj[:k_last+1]

        # count trajectories which have arrived
        idx_arrived = np.where(been_in_target_set == True)
        self.N_arrived = fht[idx_arrived].shape[0]
        if self.N_arrived != self.N:
            return

        # replace trajectories which have not arrived
        idx_not_arrived = np.where(been_in_target_set == False)
        fht[idx_not_arrived] = self.k_lim
        self.fht = fht

        # first and last fht
        self.first_fht = np.min(fht)
        self.last_fht = np.max(fht)

        # compute mean and variance of fht
        self.mean_fht, \
        self.var_fht, \
        self.re_fht = self.compute_mean_variance_and_rel_error(fht)

        # compute mean and variance of fhts
        self.mean_fhts, \
        self.var_fhts, \
        self.re_fhts = self.compute_mean_variance_and_rel_error(fht / self.dt)

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

    def save_not_controlled(self):
        np.savez(
            os.path.join(self.dir_path, 'mc-sampling.npz'),
            mean_I=self.mean_I,
            var_I=self.var_I,
            re_I=self.re_I,
            traj=self.traj,
        )

    def load_not_controlled(self, N):
        dir_path = os.path.join(
            sample.example_dir_path,
            'mc-sampling',
            'N_{:.0e}'.format(self.N),
        )
        file_path = os.path.join(dir_path, 'mc-sampling.npz')
        data = np.load(file_path, allow_pickle=True)
        self.mean_I = data['mean_I']
        self.var_I = data['var_I']
        self.re_I = data['re_I']
        self.N = N

    def write_euler_maruyama_parameters(self, f):
        f.write('Euler-Maruyama discretization parameters\n')
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('maximal time steps: {:,d}\n\n'.format(self.k_lim))

    def write_sampling_parameters(self, f):
        f.write('Sampling parameters\n')
        f.write('controlled process: {}\n'.format(self.is_controlled))

        if self.seed:
            f.write('seed: {:2.1f}'.format(self.seed))

        initial_posicion = 'xzero: ('
        for i in range(self.sde.n):
            if i == 0:
                initial_posicion += '{:2.1f}'.format(self.xzero[i])
            else:
                initial_posicion += ', {:2.1f}'.format(self.xzero[i])
        initial_posicion += ')\n'
        f.write(initial_posicion)

        f.write('sampled trajectories: {:,d}\n\n'.format(self.N))

    def write_report(self):
        '''
        '''
        # set file path

        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, "w")

        self.sde.write_setting(f)
        self.write_euler_maruyama_parameters(f)
        self.write_sampling_parameters(f)

        if self.is_controlled and not self.is_optimal:
            self.ansatz.write_ansatz_parameters(f)

        f.write('Statistics\n\n')

        f.write('trajectories which arrived: {:2.2f} %\n'
                ''.format(100 * self.N_arrived / self.N))
        f.write('used time steps: {:,d}\n\n'.format(int(self.last_fht / self.dt)))
        if self.N_arrived == 0:
            f.close()
            return

        f.write('First hitting time (fht)\n')
        f.write('first fht = {:2.3f}\n'.format(self.first_fht))
        f.write('last fht = {:2.3f}\n'.format(self.last_fht))
        f.write('E[fht] = {:2.3f}\n'.format(self.mean_fht))
        f.write('Var[fht] = {:2.3f}\n'.format(self.var_fht))
        f.write('RE[fht] = {:2.3f}\n\n'.format(self.re_fht))

        f.write('First hitting time step (fhts)\n')
        f.write('E[fhts] = {:2.3f}\n'.format(self.mean_fhts))
        f.write('Var[fhts] = {:2.3f}\n'.format(self.var_fhts))
        f.write('RE[fhts] = {:2.3f}\n\n'.format(self.re_fhts))

        if not self.is_controlled:
            f.write('Quantity of interest\n')
            f.write('E[exp(- fht)] = {:2.3e}\n'.format(self.mean_I))
            f.write('Var[exp(- fht)] = {:2.3e}\n'.format(self.var_I))
            f.write('RE[exp(- fht)] = {:2.3e}\n\n'.format(self.re_I))

        else:
            f.write('\nReweighted Quantity of interest\n')
            f.write('E[exp(- fht) * M_fht] = {:2.3e}\n'.format(self.mean_I_u))
            f.write('Var[exp(- fht) * M_fht] = {:2.3e}\n'.format(self.var_I_u))
            f.write('RE[exp(- fht) * M_fht] = {:2.3e}\n\n'.format(self.re_I_u))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        f.close()

    def plot_trajectory(self):
        from mds.plots_1d import Plot1d

        traj_fhs = self.traj.shape[0]
        traj_fht = traj_fhs * self.dt
        x = np.linspace(0, traj_fht, traj_fhs)
        ys = np.moveaxis(self.traj, 0, -1)
        labels = [r'$x_{}$'.format(i+1) for i in np.arange(self.sde.n)]

        for i in np.arange(self.sde.n):
            file_name = 'trajectory_x{:d}'.format(i+1)
            plt1d = Plot1d(self.dir_path, file_name)
            plt1d.xlabel = 't'
            plt1d.ylabel = r'$x_{:d}$'.format(i+1)
            plt1d.one_line_plot(x, self.traj[:, i])

        file_name = 'trajectory'
        plt1d = Plot1d(self.dir_path, file_name)
        plt1d.xlabel = 't'
        plt1d.multiple_lines_plot(x, ys, labels=labels)
