from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_sde import LangevinSDE
from mds.utils import get_time_in_hms, make_dir_path
from mds.plots import Plot

import numpy as np
import time
import os

class Sampling(LangevinSDE):
    '''
    '''

    def __init__(self, n, potential_name, alpha, beta, target_set=None,
                 domain=None, h=None, is_controlled=None, is_optimal=None):
        '''
        '''

        super().__init__(n, potential_name, alpha, beta,
                         target_set, domain, h)

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

        # grid evalutations
        self.grid_potential = None
        self.grid_bias_potential = None
        self.grid_controlled_potential = None
        self.grid_value_function = None

        self.grid_gradient = None
        self.grid_control = None
        self.grid_controlled_drift = None

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

        # dir_path
        self.dir_path = None

    def set_dir_path(self, dir_path):
        self.dir_path = dir_path
        make_dir_path(self.dir_path)

    def set_not_controlled_dir_path(self):
        assert self.N is not None, ''

        self.dir_path = os.path.join(
            self.example_dir_path,
            'mc-sampling',
            'N_{:.0e}'.format(self.N),
        )
        make_dir_path(self.dir_path)

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters
        '''
        return 2 * self.ansatz.value_function(x, theta)

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
        diffusion = np.dot(dB, np.sqrt(2 / self.beta) * np.eye(self.n))
        return x + drift + diffusion

    def get_idx_new_in_target_set(self, x):
        # assume trajectories are in the target set
        is_in_target_set = np.repeat([True], self.N)
        for i in range(self.n):
            is_not_in_target_set_i_axis_idx = np.where(
                (x[:, i] < self.target_set[i, 0]) |
                (x[:, i] > self.target_set[i, 1])
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
        xtemp = np.full((self.N, self.n), self.xzero)

        if self.save_trajectory:
            self.traj = np.empty((self.k_lim + 1, self.n))
            self.traj[0] = xtemp[0, :]

        for k in np.arange(1, self.k_lim + 1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # compute gradient
            gradient = self.gradient(xtemp)

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

    def sample_controlled(self):
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
            utemp = self.ansatz.control(xtemp)

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

    def sample_optimal_controlled(self, h):
        self.start_timer()
        self.initialize_fht()
        self.initialize_girsanov_martingale_terms()

        # initialize xtemp
        xtemp = np.full((self.N, self.n), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1temp = np.zeros(self.N)
        M2temp = np.zeros(self.N)

        # load optimal control
        sol = self.get_hjb_solver(h)
        sol.load_hjb_solution()
        u_opt = sol.u_opt
        self.Nx = sol.Nx

        for k in np.arange(1, self.k_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # control at xtemp
            idx = self.get_index_vectorized(xtemp)
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

            if not self.is_controlled:
                # compute gradient
                gradient = self.gradient(xtemp[k - 1])

            else:
                # control at xtemp
                utemp = self.ansatz.control(xtemp[k - 1])

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
            utemp = self.ansatz.control(xtemp)

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

    def save_not_controlled_statistics(self):
        np.savez(
            os.path.join(self.dir_path, 'mc-sampling.npz'),
            seed=self.seed,
            xzero=self.xzero,
            dt=self.dt,
            k_lim=self.k_lim,
            N_arrived=self.N_arrived,
            first_fht=self.first_fht,
            last_fht=self.last_fht,
            mean_fht=self.mean_fht,
            var_fht=self.var_fht,
            re_fht=self.re_fht,
            mean_I=self.mean_I,
            var_I=self.var_I,
            re_I=self.re_I,
            traj=self.traj,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_not_controlled_statistics(self):
        try:
            mcs = np.load(
                os.path.join(self.dir_path, 'mc-sampling.npz'),
                allow_pickle=True,
            )
            self.seed = mcs['seed']
            self.xzero = mcs['xzero']
            self.dt = mcs['dt']
            self.k_lim = mcs['k_lim']
            self.N_arrived = mcs['N_arrived']
            self.first_fht = mcs['first_fht']
            self.last_fht = mcs['last_fht']
            self.mean_fht = mcs['mean_fht']
            self.var_fht = mcs['var_fht']
            self.re_fht = mcs['re_fht']
            self.mean_I = mcs['mean_I']
            self.var_I = mcs['var_I']
            self.re_I = mcs['re_I']
            self.traj = mcs['traj']
            self.t_initial = mcs['t_initial']
            self.t_final = mcs['t_final']
            return True
        except:
            print('no mc-sampling found with N={:.0e}'.format(self.N))
            return False

    def write_euler_maruyama_parameters(self, f):
        f.write('Euler-Maruyama discretization parameters\n')
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('maximal time steps: {:,d}\n\n'.format(self.k_lim))

    def write_sampling_parameters(self, f):
        f.write('Sampling parameters\n')
        f.write('controlled process: {}\n'.format(self.is_controlled))

        initial_posicion = 'xzero: ('
        for i in range(self.n):
            if i == 0:
                initial_posicion += '{:2.1f}'.format(self.xzero[i])
            else:
                initial_posicion += ', {:2.1f}'.format(self.xzero[i])
        initial_posicion += ')\n'
        f.write(initial_posicion)

        f.write('sampled trajectories: {:,d}\n'.format(self.N))

        if self.seed:
            f.write('seed: {:2.1f}'.format(self.seed))
        else:
            f.write('seed: -\n\n')

    def write_report(self):
        '''
        '''
        # set path

        file_path = os.path.join(self.dir_path, 'report.txt')

        # write file
        f = open(file_path, "w")

        self.write_setting(f)
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
        f.write('E[fhts] = {:2.3f}\n'.format(self.mean_fht / self.dt))
        f.write('Var[fhts] = {:2.3f}\n'.format(self.var_fht / (self.dt **2)))
        f.write('RE[fhts] = {:2.3f}\n\n'.format(self.re_fht))

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

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def get_grid_value_function_and_control(self):
        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.n)

        # potential and gradient
        self.grid_potential = self.potential(x).reshape(self.Nx)
        self.grid_gradient = self.gradient(x).reshape(self.domain_h.shape)

        if not self.is_controlled:
            # bias potential, value function and control
            self.grid_bias_potential = np.zeros(self.Nx)
            self.grid_value_function = np.zeros(self.Nx)
            self.grid_control = np.zeros(self.domain_h.shape)

        else:
            # set value f constant
            self.ansatz.set_value_function_constant_corner(self.h)

            # bias potential, value function and control
            self.ansatz.set_value_function_constant_corner()
            self.grid_bias_potential = self.bias_potential(x).reshape(self.Nx)
            self.grid_value_function = self.ansatz.value_function(x)
            self.grid_control = self.ansatz.control(x).reshape(self.domain_h.shape)

        # controlled potential and drift
        self.grid_controlled_potential = self.grid_potential + self.grid_bias_potential
        self.grid_controlled_drift = - self.grid_gradient + np.sqrt(2) * self.grid_control

    def plot_trajectory(self):
        traj_fhs = self.traj.shape[0]
        traj_fht = traj_fhs * self.dt
        x = np.linspace(0, traj_fht, traj_fhs)
        ys = np.moveaxis(self.traj, 0, -1)
        labels = [r'$x_{}$'.format(i+1) for i in np.arange(self.n)]

        for i in np.arange(self.n):
            file_name = 'trajectory_x{:d}'.format(i+1)
            plt = Plot(self.dir_path, file_name)
            plt.xlabel = 't'
            plt.ylabel = r'$x_{:d}$'.format(i+1)
            plt.one_line_plot(x, self.traj[:, i])

        file_name = 'trajectory'
        plt = Plot(self.dir_path, file_name)
        plt.xlabel = 't'
        plt.multiple_lines_plot(x, ys, labels=labels)
