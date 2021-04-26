from mds.langevin_nd_sde import LangevinSDE
from mds.utils import get_not_controlled_dir_path, \
                      get_controlled_dir_path, \
                      get_time_in_hms
from mds.plots import Plot

import numpy as np
import torch

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
        self.xt = None
        self.save_trajectory = False
        self.traj = None

        # Euler-Marujama
        self.dt = None
        self.k_lim = None

        # ansatz functions (gaussians) and coefficients
        self.ansatz = None

        # function approximation by a neural network
        self.nn_func_appr = None

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

        # loss and its gradient
        self.loss = None
        self.tilted_loss = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # dir_path
        self.dir_path = None

    def set_not_controlled_dir_path(self):
        assert self.dt is not None, ''
        assert self.N is not None, ''

        self.dir_path = get_not_controlled_dir_path(
            self.settings_dir_path,
            self.dt,
            self.N,
        )

    def set_controlled_dir_path(self, parent_dir_path):
        assert self.dt is not None, ''
        assert self.N is not None, ''

        self.dir_path = get_controlled_dir_path(
            parent_dir_path,
            self.dt,
            self.N,
        )

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

    def set_sampling_parameters(self, dt, k_lim, xzero, N, seed=None):
        '''
        '''
        # set random seed
        if seed:
            np.random.seed(seed)

        # Euler-Marujama
        self.dt = dt
        self.k_lim = k_lim

        # sampling
        self.xzero = xzero
        self.N = N

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def initialize_fht(self):
        '''
        '''
        assert self.N is not None, ''

        # boolean array telling us if a trajectory have been in the target set
        self.been_in_target_set = np.repeat([False], self.N).reshape(self.N, 1)

        # first hitting time of each trajectory
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

        # boolean array telling us if a trajectory is in the target set
        is_in_target_set = (
            (x >= self.target_set[:, 0]) &
            (x <= self.target_set[:, 1])
        ).all(axis=1).reshape(self.N, 1)

        # indices of trajectories new in the target set
        idx = np.where(
            (is_in_target_set == True) &
            (self.been_in_target_set == False)
        )[0]

        # update list of indices whose trajectories have been in the target set
        self.been_in_target_set[idx] = True

        return idx

    def sample_not_controlled(self):
        self.start_timer()
        self.initialize_fht()

        # initialize xt
        xt = np.full((self.N, self.n), self.xzero)

        if self.save_trajectory:
            self.traj = np.empty((self.k_lim + 1, self.n))
            self.traj[0] = xt[0, :]

        for k in np.arange(1, self.k_lim + 1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # compute gradient
            gradient = self.gradient(xt)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            if self.save_trajectory:
                self.traj[k] = xt[0, :]

            # get indices from the trajectories which are new in target
            idx = self.get_idx_new_in_target_set(xt)

            # save first hitting time
            if idx.shape[0] != 0:
                self.fht[idx] = k * self.dt

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

        # initialize xt
        xt = np.full((self.N, self.n), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1_t = np.zeros(self.N)
        M2_t = np.zeros(self.N)

        for k in np.arange(1, self.k_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # control at xt
            ut = self.ansatz.control(xt)

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            # Girsanov Martingale terms
            M1_t -= np.sqrt(self.beta) \
                  * np.matmul(
                      ut[:, np.newaxis, :],
                      dB[:, :, np.newaxis],
                  ).squeeze()
            M2_t -= self.beta * 0.5 * (np.linalg.norm(ut, axis=1) ** 2) * self.dt

            # get indices from the trajectories which are new in target set
            idx = self.get_idx_new_in_target_set(xt)

            # save first hitting time and Girsanov Martingale terms
            if idx.shape[0] != 0:
                self.fht[idx] = k * self.dt
                self.M1_fht[idx] = M1_t[idx]
                self.M2_fht[idx] = M2_t[idx]

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

        # initialize xt
        xt = np.full((self.N, self.n), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1_t = np.zeros(self.N)
        M2_t = np.zeros(self.N)

        # load optimal control
        sol = self.get_hjb_solver(h)
        sol.load_hjb_solution()
        u_opt = sol.u_opt
        self.Nx = sol.Nx

        for k in np.arange(1, self.k_lim +1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # control at xt
            idx_xt = self.get_index_vectorized(xt)
            ut = u_opt[idx_xt]

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            # Girsanov Martingale terms
            M1_t -= np.sqrt(self.beta) \
                  * np.matmul(
                      ut[:, np.newaxis, :],
                      dB[:, :, np.newaxis],
                  ).squeeze()
            M2_t -= self.beta * 0.5 * (np.linalg.norm(ut, axis=1) ** 2) * self.dt

            # get indices from the trajectories which are new in target set
            idx = self.get_idx_new_in_target_set(xt)

            # save first hitting time and Girsanov Martingale terms
            if idx.shape[0] != 0:
                self.fht[idx] = k * self.dt
                self.M1_fht[idx] = M1_t[idx]
                self.M2_fht[idx] = M2_t[idx]

            # check if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()

    def sample_meta(self):
        self.initialize_fht()

        # initialize xt
        x = np.empty((self.k_lim + 1, self.N, self.n))
        x[0] = np.full((self.N, self.n), self.xzero)

        for k in np.arange(1, self.k_lim + 1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            if not self.is_controlled:
                # compute gradient
                gradient = self.gradient(x[k - 1])

            else:
                # control at xt
                ut = self.ansatz.control(x[k - 1])

                # compute gradient
                gradient = self.tilted_gradient(x[k - 1], ut)

            # sde update
            x[k] = self.sde_update(x[k - 1], gradient, dB)

            # update been in target set
            _ = self.get_idx_new_in_target_set(x[k])

            # check if the half of the trajectories have arrived to the target set
            if np.sum(self.been_in_target_set) >= self.N / 2:
                return True, x[:k]

        return False, x

    def sample_loss_ansatz(self):
        self.initialize_fht()

        # number of ansatz functions
        m = self.ansatz.m

        # initialize statistics 
        J = np.zeros(self.N)
        grad_J = np.zeros((self.N, m))

        # initialize xt
        xt = np.full((self.N, self.n), self.xzero)

        # initialize ipa variables
        cost_temp = np.zeros(self.N)
        grad_phi_temp = np.zeros((self.N, m))
        grad_S_temp = np.zeros((self.N, m))

        for k in np.arange(1, self.k_lim+1):
            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)

            # control
            btemp = self.ansatz.basis_control(xt)
            ut = self.ansatz.control(xt)

            # ipa statistics 
            normed_ut = np.linalg.norm(ut, axis=1)
            cost_temp += (1 + 0.5 * (normed_ut ** 2)) * self.dt
            grad_phi_temp += np.sum(ut[:, np.newaxis, :] * btemp, axis=2) * self.dt
            grad_S_temp -= np.sqrt(self.beta) * np.sum(dB[:, np.newaxis, :] * btemp, axis=2)

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            # get indices from the trajectories which are new in target set
            idx = self.get_idx_new_in_target_set(xt)

            # save ipa statistics
            if idx.shape[0] != 0:
                J[idx] = cost_temp[idx]
                grad_J[idx, :] = grad_phi_temp[idx, :] \
                               - (cost_temp[idx])[:, np.newaxis] \
                               * grad_S_temp[idx, :]

            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        # compute averages
        mean_J = np.mean(J)
        mean_grad_J = np.mean(grad_J, axis=0)

        return True, mean_J, mean_grad_J, k


    def sample_loss_nn(self, device):
        self.initialize_fht()
        self.initialize_girsanov_martingale_terms()

        ## nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # preallocate loss and tilted loss
        loss = np.zeros(self.N)
        tilted_loss = torch.zeros(self.N)
        a_tensor = torch.zeros(self.N).to(device)
        b_tensor = torch.zeros(self.N).to(device)
        c_tensor = torch.zeros(self.N).to(device)

        # initialize trajectory
        xt = np.full((self.N, self.n), self.xzero)

        # initialize Girsanov Martingale terms, M_t = e^(M1_t + M2_t)
        M1_t = np.zeros(self.N)
        M2_t = np.zeros(self.N)

        for k in np.arange(1, self.k_lim + 1):

            # Brownian increment
            dB = np.sqrt(self.dt) \
               * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)
            dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32).to(device)

            # control
            xt_tensor = torch.tensor(xt, dtype=torch.float)
            ut_tensor = model.forward(xt_tensor)
            ut_tensor_det = ut_tensor.detach()
            ut = ut_tensor_det.numpy()

            # sde update
            controlled_gradient = self.gradient(xt) - np.sqrt(2) * ut
            xt = self.sde_update(xt, controlled_gradient, dB)

            # update statistics
            a_tensor = a_tensor \
                     + torch.matmul(
                         torch.unsqueeze(ut_tensor_det, 1),
                         torch.unsqueeze(ut_tensor, 2),
                     ).reshape(self.N,) * self.dt

            ut_norm_det = torch.linalg.norm(ut_tensor_det, axis=1)
            b_tensor = b_tensor + ((1 + 0.5 * (ut_norm_det ** 2)) * self.dt).reshape(self.N,)

            c_tensor = c_tensor \
                     - np.sqrt(self.beta) * torch.matmul(
                         torch.unsqueeze(ut_tensor, 1),
                         torch.unsqueeze(dB_tensor, 2),
                     ).reshape(self.N,)

            # Girsanov Martingale terms
            M1_t -= np.sqrt(self.beta) \
                  * np.matmul(
                      ut[:, np.newaxis, :],
                      dB[:, :, np.newaxis],
                  ).squeeze()
            M2_t -= self.beta * 0.5 * (np.linalg.norm(ut, axis=1) ** 2) * self.dt

            # get indices of trajectories which are new in the target set
            idx = self.get_idx_new_in_target_set(xt)

            if idx.shape[0] != 0:

                # get tensor indices if there are new trajectories 
                idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)

                # save loss and tilted loss for the arrived trajectorries
                loss[idx] = b_tensor.numpy()[idx]
                tilted_loss[idx_tensor] = a_tensor.index_select(0, idx_tensor) \
                                        - b_tensor.index_select(0, idx_tensor) \
                                        * c_tensor.index_select(0, idx_tensor)

                # save first hitting time and Girsanov Martingale terms
                self.fht[idx] = k * self.dt
                self.M1_fht[idx] = M1_t[idx]
                self.M2_fht[idx] = M2_t[idx]

            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
               break

        self.loss = np.mean(loss)
        self.tilted_loss = torch.mean(tilted_loss)

        self.compute_I_u_statistics()

        return True, k


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
        # cut saved trajectory
        if self.save_trajectory:
            k_last = int(self.fht[0] / self.dt)
            self.traj = self.traj[:k_last+1]

        # count trajectories which have arrived
        idx_arrived = np.where(self.been_in_target_set == True)[0]
        self.N_arrived = self.fht[idx_arrived].shape[0]
        if self.N_arrived != self.N:
            return

        # replace trajectories which have not arrived
        idx_not_arrived = np.where(self.been_in_target_set == False)[0]
        self.fht[idx_not_arrived] = self.k_lim

        # first and last fht
        self.first_fht = np.min(self.fht)
        self.last_fht = np.max(self.fht)

        # compute mean and variance of fht
        self.mean_fht, \
        self.var_fht, \
        self.re_fht = self.compute_mean_variance_and_rel_error(self.fht)

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
            msg = 'no mc-sampling found with dt={:.4f} and N={:.0e}' \
                  ''.format(self.dt, self.N)
            print(msg)
            return False

    def save_controlled_statistics(self):
        np.savez(
            os.path.join(self.dir_path, 'is.npz'),
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
            mean_I_u=self.mean_I_u,
            var_I_u=self.var_I_u,
            re_I_u=self.re_I_u,
            traj=self.traj,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_controlled_statistics(self):
        try:
            ims = np.load(
                os.path.join(self.dir_path, 'is.npz'),
                allow_pickle=True,
            )
            self.seed = ims['seed']
            self.xzero = ims['xzero']
            self.dt = ims['dt']
            self.k_lim = ims['k_lim']
            self.N_arrived = ims['N_arrived']
            self.first_fht = ims['first_fht']
            self.last_fht = ims['last_fht']
            self.mean_fht = ims['mean_fht']
            self.var_fht = ims['var_fht']
            self.re_fht = ims['re_fht']
            self.mean_I_u = ims['mean_I_u']
            self.var_I_u = ims['var_I_u']
            self.re_I_u = ims['re_I_u']
            self.traj = ims['traj']
            self.t_initial = ims['t_initial']
            self.t_final = ims['t_final']
            return True
        except:
            msg = 'no importance-sampling found with dt={:.4f} and N={:.0e}' \
                  ''.format(self.dt, self.N)
            print(msg)
            return False

    def write_euler_maruyama_parameters(self, f):
        f.write('\nEuler-Maruyama discretization parameters\n')
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('maximal time steps: {:,d}\n'.format(self.k_lim))

    def write_sampling_parameters(self, f):
        f.write('\nSampling parameters\n')
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
            f.write('seed: {:2.1f}\n'.format(self.seed))
        else:
            f.write('seed: -\n')

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

        f.write('\nStatistics\n')

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

    def get_grid_value_function(self):
        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.n)

        # potential
        self.grid_potential = self.potential(x).reshape(self.Nx)

        if not self.is_controlled:
            # bias potential and value function
            self.grid_bias_potential = np.zeros(self.Nx)
            self.grid_value_function = np.zeros(self.Nx)

        elif self.is_controlled and self.ansatz is not None:
            # set value f constant
            self.ansatz.set_value_function_constant_corner()

            # bias potential and value function
            self.grid_bias_potential = self.bias_potential(x).reshape(self.Nx)
            self.grid_value_function = self.ansatz.value_function(x)

        if self.grid_bias_potential is not None:
            # controlled potential
            self.grid_controlled_potential = self.grid_potential + self.grid_bias_potential

    def get_grid_control(self):
        # flattened domain_h
        x = self.domain_h.reshape(self.Nh, self.n)

        # gradient
        self.grid_gradient = self.gradient(x).reshape(self.domain_h.shape)

        # null control
        if not self.is_controlled:
            self.grid_control = np.zeros(self.domain_h.shape)

        # gaussian ansatz control
        elif self.is_controlled and self.ansatz is not None:
            self.grid_control = self.ansatz.control(x).reshape(self.domain_h.shape)

        # two layer nn control
        elif self.is_controlled and self.nn_func_appr is not None:
            inputs = torch.tensor(x, dtype=torch.float)
            control_flattened = self.nn_func_appr.model(inputs).detach().numpy()
            self.grid_control = control_flattened.reshape(self.domain_h.shape)

        # controlled drift
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
