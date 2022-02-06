from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.utils_path import get_not_controlled_dir_path, \
                                               get_controlled_dir_path, \
                                               get_time_in_hms

import matplotlib.pyplot as plt
import numpy as np
import torch

import time
import os

class Sampling(LangevinSDE):
    '''
    '''

    def __init__(self, problem_name, potential_name, n, alpha, beta, domain=None,
                 target_set=None, h=None, is_controlled=None, T=None, nu=None, is_optimal=None):
        '''
        '''

        super().__init__(problem_name, potential_name, n, alpha, beta,
                         domain, target_set, T, nu)

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
        self.k = None

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

        # deterministic and stochastic integrals
        self.stoch_int_t = None
        self.stoch_int_fht = None
        self.det_int_t = None
        self.det_int_fht = None

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
        #self.M1_fht = None
        #self.M2_fht = None
        #self.M_fht = None
        #self.k = None
        #self.M1_k = None
        #self.M2_k = None
        #self.mean_M_k= None

        self.mean_I_u = None
        self.var_I_u = None
        self.re_I_u = None

        # loss and its gradient
        self.loss = None
        self.var_loss = None
        self.grad_loss = None
        self.ipa_loss = None

        # control l2 error wrt the hjb solution
        self.do_u_l2_error = False
        self.u_l2_error = None

        # computational time
        self.ct_initial = None
        self.ct_final = None
        self.ct = None

        # dir_path
        self.dir_path = None

    def set_not_controlled_dir_path(self):
        assert self.dt is not None, ''
        assert self.N is not None, ''

        self.dir_path = get_not_controlled_dir_path(
            self.settings_dir_path,
            self.dt,
            self.N,
            self.seed,
        )

    def set_controlled_dir_path(self, parent_dir_path):
        assert self.dt is not None, ''
        assert self.N is not None, ''

        self.dir_path = get_controlled_dir_path(
            parent_dir_path,
            self.dt,
            self.N,
            self.seed,
        )

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x ((N, n)-array) : position
            theta ((m,)-array): parameters
        '''
        return 2 * self.ansatz.value_function(x, theta) / self.beta

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

    def brownian_increment(self, tensor=False):
        '''
        '''
        dB = np.sqrt(self.dt) \
           * np.random.normal(0, 1, self.N * self.n).reshape(self.N, self.n)
        if tensor:
            dB = torch.tensor(dB, dtype=torch.float32)

        return dB

    def set_sampling_parameters(self, dt, k_lim, xzero, N, seed=None):
        '''
        '''
        # set random seed
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

        # Euler-Marujama
        self.dt = dt
        self.k_lim = k_lim

        # sampling
        self.xzero = xzero
        self.N = N

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def preallocate_fht(self):
        '''
        '''
        assert self.N is not None, ''

        # boolean array telling us if a trajectory have been in the target set
        self.been_in_target_set = np.repeat([False], self.N).reshape(self.N, 1)

        # first hitting time of each trajectory
        self.fht = np.empty(self.N)

    def preallocate_girsanov_martingale_terms(self):
        ''' preallocates Girsanov Martingale terms, M_fht = e^(M1_fht + M2_fht)
        '''
        assert self.N is not None, ''

        self.M1_fht = np.empty(self.N)
        self.M2_fht = np.empty(self.N)

    def preallocate_integrals(self):
        '''
        '''
        assert self.N is not None, ''

        self.stoch_int_fht = np.empty(self.N)
        self.det_int_fht = np.empty(self.N)

    def preallocate_l2_error(self):
        '''
        '''
        assert self.N is not None, ''

        self.u_l2_error_fht = np.empty(self.N)


    def initial_position(self, tensor=False):
        ''' returns same initial posicion for all trajectories
        '''
        x_init = np.full((self.N, self.n), self.xzero, dtype=np.float64)
        if tensor:
            x_init = torch.tensor(x_init, dtype=torch.float32)

        return x_init


    def initialize_running_integrals(self):
        '''
        '''
        assert self.N is not None, ''

        self.stoch_int_t = np.zeros(self.N)
        self.det_int_t = np.zeros(self.N)

    def initialize_running_l2_error(self):
        '''
        '''
        assert self.N is not None, ''

        self.u_l2_error_t = np.zeros(self.N)

    def update_integrals(self, ut, dB):
        '''
        '''
        # stochastic integral
        self.stoch_int_t += np.matmul(
            ut[:, np.newaxis, :],
            dB[:, :, np.newaxis],
        ).squeeze()

        # deterministic integral
        self.det_int_t += (np.linalg.norm(ut, axis=1) ** 2) * self.dt

    def update_running_l2_error(self, xt, ut):

        # hjb control
        idx_xt = self.get_index_vectorized(xt)
        ut_hjb = self.u_hjb[idx_xt]

        # update u l2 running error
        self.u_l2_error_t += (np.linalg.norm(ut - ut_hjb, axis=1) ** 2) * self.dt

    def update_running_l2_error_det(self, k, xt, ut):

        # hjb control
        idx_xt = self.sol_hjb.get_space_index_vectorized(xt)
        ut_hjb = self.sol_hjb.u_opt_i[k, idx_xt]
        ut_hjb = np.moveaxis(ut_hjb, 0, -1)

        # update u l2 running error
        self.u_l2_error_t += (np.linalg.norm(ut - ut_hjb, axis=1) ** 2) * self.dt

    def sde_update(self, x, gradient, dB, tensor=False):
        drift = - gradient * self.dt

        if not tensor:
            diffusion = np.dot(dB, np.sqrt(2 / self.beta) * np.eye(self.n))
        else:
            diffusion = torch.mm(dB, np.sqrt(2 / self.beta) * torch.eye(self.n))

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

    def sample_not_controlled_det(self):
        self.start_timer()

        # initialize xt
        xt = self.initial_position()

        for k in np.arange(1, self.k_lim + 1):

            # compute gradient
            gradient = self.gradient(xt)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dB)

        # save work functional
        self.work = self.g(xt)

        self.stop_timer()

    def sample_not_controlled(self):
        self.start_timer()
        self.preallocate_fht()

        # initialize xt
        xt = self.initial_position()

        if self.save_trajectory:

            # preallocate array for the trajectory and save initial position
            self.traj = np.empty((self.k_lim + 1, self.n))
            self.traj[0] = xt[0, :]

        for k in np.arange(0, self.k_lim):

            # compute gradient
            gradient = self.gradient(xt)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            if self.save_trajectory:

                # save trajectory at time k
                self.traj[k] = xt[0, :]

            # get indices from the trajectories which are new in target
            idx = self.get_idx_new_in_target_set(xt)

            # save first hitting time
            if idx.shape[0] != 0:
                self.fht[idx] = k * self.dt

            # break if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        self.stop_timer()

    def sample_controlled(self):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        if self.do_u_l2_error:

            # load hjb solution
            sol_hjb = self.get_hjb_solver()
            self.u_hjb = sol_hjb.u_opt
            self.h = sol_hjb.h

            # preallocate l2 error
            self.preallocate_l2_error()


        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        for k in np.arange(1, self.k_lim +1):

            # control at xt
            if self.ansatz is not None:
                ut = self.ansatz.control(xt)
            elif self.nn_func_appr is not None:
                xt_tensor = torch.tensor(xt, dtype=torch.float)
                ut_tensor = self.nn_func_appr.model.forward(xt_tensor)
                ut_tensor_det = ut_tensor.detach()
                ut = ut_tensor_det.numpy()

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dB)

            # update l2 running error
            if self.do_u_l2_error:
                self.update_running_l2_error(xt, ut)

            # get indices from the trajectories which are new in target set
            idx = self.get_idx_new_in_target_set(xt)

            # save first hitting time and Girsanov Martingale terms
            if idx.shape[0] != 0:
                self.fht[idx] = k * self.dt
                self.stoch_int_fht[idx] = self.stoch_int_t[idx]
                self.det_int_fht[idx] = self.det_int_t[idx]
                if self.do_u_l2_error:
                    self.u_l2_error_fht[idx] = self.u_l2_error_t[idx]


            # break if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)
        self.stop_timer()

    def sample_optimal_controlled_det(self, h, dt):
        self.start_timer()

        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # load hjb solver and get the "optimal" control
        sol_hjb = self.get_hjb_solver_det(h, dt)

        for k in np.arange(1, self.k_lim +1):

            # control at xt
            idx_xt = sol_hjb.get_space_index_vectorized(xt)
            ut = sol_hjb.u_opt_i[k, idx_xt]
            ut = np.moveaxis(ut, 0, -1)

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dB)

        # save work functional
        self.work = self.g(xt)

        self.compute_I_u_statistics_det()
        self.stop_timer()

    def sample_optimal_controlled(self, h):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        # load hjb solver and get the "optimal" control
        sol_hjb = self.get_hjb_solver(h)
        u_opt = sol_hjb.u_opt
        self.h = sol_hjb.h

        # initialize xt
        xt = self.initial_position()

        # initialize control
        idx_xt = self.get_index_vectorized(xt)
        ut = u_opt[idx_xt]

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        for k in np.arange(0, self.k_lim):

            # get Brownian increment
            dB = self.brownian_increment()

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dB)

            # get indices from the trajectories which are new in target set
            idx = self.get_idx_new_in_target_set(xt)

            if idx.shape[0] != 0:

                # save first hitting time and integrals
                self.fht[idx] = k * self.dt
                self.stoch_int_fht[idx] = self.stoch_int_t[idx]
                self.det_int_fht[idx] = self.det_int_t[idx]

            # break if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

            # control at xt
            idx_xt = self.get_index_vectorized(xt)
            ut = u_opt[idx_xt]

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()


    def sample_meta(self):
        self.preallocate_fht()

        # preallocate trajectory
        x = np.empty((self.k_lim + 1, self.N, self.n))

        # initialize xt
        x[0] = self.initial_position()

        for k in np.arange(1, self.k_lim + 1):

            if not self.is_controlled:
                # compute gradient
                gradient = self.gradient(x[k - 1])

            else:
                # control at xt
                if self.ansatz is not None:
                    ut = self.ansatz.control(x[k - 1])
                elif self.nn_func_appr is not None:
                    xt_tensor = torch.tensor(x[k - 1], dtype=torch.float)
                    ut_tensor = self.nn_func_appr.model.forward(xt_tensor)
                    ut_tensor_det = ut_tensor.detach()
                    ut = ut_tensor_det.numpy()

                # compute gradient
                gradient = self.tilted_gradient(x[k - 1], ut)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            x[k] = self.sde_update(x[k - 1], gradient, dB)

            # update been in target set
            _ = self.get_idx_new_in_target_set(x[k])

            # check if the half of the trajectories have arrived to the target set
            if np.sum(self.been_in_target_set) >= self.N / 2:
                return True, x[:k]

        return False, x

    def sample_loss_ipa_ansatz(self):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        if self.do_u_l2_error:

            # load hjb solution
            sol_hjb = self.get_hjb_solver()
            self.u_hjb = sol_hjb.u_opt
            self.h = sol_hjb.h

            # preallocate l2 error
            self.preallocate_l2_error()

        # number of ansatz functions
        m = self.ansatz.m

        # preallocate loss and its gradient for the trajectories
        loss_traj = np.empty(self.N)
        grad_loss_traj = np.empty((self.N, m))

        # initialize running gradient of phi and running gradient of S
        grad_phi_t = np.zeros((self.N, m))
        grad_S_t = np.zeros((self.N, m))

        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        for k in np.arange(1, self.k_lim+1):

            # control
            ut = self.ansatz.control(xt)

            # the gradient of the control wrt the parameters are the basis
            # of the gaussian ansatz gradient
            grad_ut = - (np.sqrt(2) / self.beta) * self.ansatz.mvn_pdf_gradient_basis(xt)

            # get Brownian increment
            dB = self.brownian_increment()

            # update running gradient of phi and running gradient of S
            grad_phi_t += self. beta * np.sum(ut[:, np.newaxis, :] * grad_ut, axis=2) * self.dt
            grad_S_t -= np.sqrt(self.beta) * np.sum(dB[:, np.newaxis, :] * grad_ut, axis=2)

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dB)

            # update l2 running error
            if self.do_u_l2_error:
                self.update_running_l2_error(xt, ut)

            # get indices from the trajectories which are new in target set
            idx = self.get_idx_new_in_target_set(xt)

            # save ipa statistics
            if idx.shape[0] != 0:

                # save first hitting time and integrals
                self.fht[idx] = k * self.dt
                self.stoch_int_fht[idx] = self.stoch_int_t[idx]
                self.det_int_fht[idx] = self.det_int_t[idx]

                # save loss and grad loss for the arrived trajectories
                loss_traj[idx] = self.fht[idx] + 0.5 * self.beta * self.det_int_fht[idx]
                grad_loss_traj[idx, :] = grad_phi_t[idx, :] \
                                       - loss_traj[idx][:, np.newaxis] \
                                       * grad_S_t[idx, :]

                # u l2 error
                if self.do_u_l2_error:
                    self.u_l2_error_fht[idx] = self.u_l2_error_t[idx]

            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

        # compute averages
        self.loss = np.mean(loss_traj)
        self.var_loss = np.var(loss_traj)
        self.grad_loss = np.mean(grad_loss_traj, axis=0)

        # compute statistics
        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)

        self.stop_timer()

        return True


    def sample_loss_ipa_nn(self, device):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        if self.do_u_l2_error:

            # load hjb solution
            sol_hjb = self.get_hjb_solver()
            self.u_hjb = sol_hjb.u_opt

            # preallocate l2 error
            self.preallocate_l2_error()

        # nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # initialize phi and S
        phi_t = torch.zeros(self.N)
        phi_fht = torch.empty(self.N)
        S_t = torch.zeros(self.N)
        S_fht = torch.empty(self.N)

        # initialize trajectory
        xt = self.initial_position()
        xt_tensor = torch.tensor(xt, dtype=torch.float32)

        # initialize control
        ut_tensor = model.forward(xt_tensor)
        ut = ut_tensor.detach().numpy()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        for k in np.arange(0, self.k_lim):

            # get Brownian increment and tensorize it
            dB = self.brownian_increment()
            dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32)

            # update running phi
            ut_norm_tensor = torch.linalg.norm(ut_tensor, axis=1)
            phi_t = phi_t + ((1 + 0.5 * self.beta * (ut_norm_tensor ** 2)) * self.dt).reshape(self.N,)

            # update running discretized action
            S_t = S_t \
                - np.sqrt(self.beta) * torch.matmul(
                    torch.unsqueeze(ut_tensor, 1),
                    torch.unsqueeze(dB_tensor, 2),
                ).reshape(self.N,)

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dB)

            # update l2 running error
            if self.do_u_l2_error:
                self.update_running_l2_error(xt, ut)

            # get indices of trajectories which are new in the target set
            idx = self.get_idx_new_in_target_set(xt)

            if idx.shape[0] != 0:

                # save first hitting time and integrals
                self.fht[idx] = k * self.dt
                self.stoch_int_fht[idx] = self.stoch_int_t[idx]
                self.det_int_fht[idx] = self.det_int_t[idx]

                # get tensor indices if there are new trajectories 
                idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)

                # save phi and S loss for the arrived trajectorries
                phi_fht[idx_tensor] = phi_t.index_select(0, idx_tensor)
                S_fht[idx_tensor] = S_t.index_select(0, idx_tensor)

                # u l2 error
                if self.do_u_l2_error:
                    self.u_l2_error_fht[idx] = self.u_l2_error_t[idx]

            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
               break

            # sde update
            controlled_gradient = self.gradient(xt) - np.sqrt(2) * ut
            xt = self.sde_update(xt, controlled_gradient, dB)
            xt_tensor = torch.tensor(xt, dtype=torch.float32)

            # control
            ut_tensor = model.forward(xt_tensor)
            ut = ut_tensor.detach().numpy()


        # compute loss
        a = torch.mean(phi_fht)
        self.loss = a.detach().numpy()
        self.var_loss = torch.var(phi_fht)

        # compute ipa loss
        b = - torch.mean(phi_fht.detach() * S_fht)
        self.ipa_loss = a + b

        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)

        self.stop_timer()

        return True

    def sample_loss_ipa_nn_det(self, device):
        self.start_timer()

        if self.do_u_l2_error:

            # load hjb solver 
            self.sol_hjb = self.get_hjb_solver_det(dt=self.dt)

            # preallocate l2 error
            self.preallocate_l2_error()

        # nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # initialize phi and S
        phi_t = torch.zeros(self.N)
        S_t = torch.zeros(self.N)

        # initialize trajectory
        xt = self.initial_position()
        xt_tensor = torch.tensor(xt, dtype=torch.float32)

        # initialize control
        ut_tensor = model.forward(k, xt_tensor)
        ut = ut_tensor.detach().numpy()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        for k in np.arange(0, self.k_lim):

            # get Brownian increment and tensorize it
            dB = self.brownian_increment()
            dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32)

            # update stochastic and deterministic running integrals 
            self.update_integrals(ut, dB)

            # update running phi
            ut_norm_tensor = torch.linalg.norm(ut_tensor, axis=1)
            phi_t = phi_t + (0.5 * self.beta * (ut_norm_tensor ** 2) * self.dt).reshape(self.N,)

            # update running discretized action
            S_t = S_t \
                - np.sqrt(self.beta) * torch.matmul(
                torch.unsqueeze(ut_tensor, 1),
                torch.unsqueeze(dB_tensor, 2),
            ).reshape(self.N,)

            # update l2 running error
            if self.do_u_l2_error:
                self.update_running_l2_error_det(k, xt, ut)

            # sde update
            #controlled_gradient = self.gradient(xt) - np.sqrt(2) * ut
            #xt = self.sde_update(xt, controlled_gradient, dB_tensor)
            #xt_tensor = torch.tensor(xt, dtype=torch.float32)
            controlled_gradient = self.gradient(xt_tensor, tensor=True) - np.sqrt(2) * ut_tensor
            xt_tensor = self.sde_update(xt_tensor, controlled_gradient, dB_tensor, tensor=True)
            xt = xt_tensor.detach().numpy()

            # control
            ut_tensor = model.forward(k, xt_tensor)
            ut = ut_tensor.detach().numpy()


        # save work functional
        self.work = self.g(xt)

        # update phi
        phi_t = phi_t + self.g(xt_tensor, tensor=True)

        # compute loss
        a = torch.mean(phi_t)
        self.loss = a.detach().numpy()

        # compute ipa loss
        b = - torch.mean(phi_t.detach() * S_t)
        self.ipa_loss = a + b

        self.compute_I_u_statistics_det()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)

        self.stop_timer()

        return True

    def sample_loss_ipa2_nn(self, device):
        self.preallocate_fht()
        self.preallocate_integrals()

        if self.do_u_l2_error:

            # load hjb solution
            sol_hjb = self.get_hjb_solver()
            self.u_hjb = sol_hjb.u_opt
            self.h = sol_hjb.h

            # preallocate l2 error
            self.preallocate_l2_error()


        # nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # initialize loss and ipa loss for the trajectories
        loss_traj = np.zeros(self.N)
        ipa_loss_traj = torch.zeros(self.N)
        a_tensor = torch.zeros(self.N).to(device)
        b_tensor = torch.zeros(self.N).to(device)
        c_tensor = torch.zeros(self.N).to(device)

        # initialize trajectory
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        for k in np.arange(1, self.k_lim + 1):

            # get Brownian increment and tensorize it
            dB = self.brownian_increment()
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

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dB)

            # update l2 running error
            if self.do_u_l2_error:
                self.update_running_l2_error(xt, ut)

            # get indices of trajectories which are new in the target set
            idx = self.get_idx_new_in_target_set(xt)

            if idx.shape[0] != 0:

                # save first hitting time and integrals
                self.fht[idx] = k * self.dt
                self.stoch_int_fht[idx] = self.stoch_int_t[idx]
                self.det_int_fht[idx] = self.det_int_t[idx]

                # get tensor indices if there are new trajectories 
                idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)

                # save loss and ipa loss for the arrived trajectorries
                loss_traj[idx] = b_tensor.numpy()[idx]
                ipa_loss_traj[idx_tensor] = a_tensor.index_select(0, idx_tensor) \
                                          - b_tensor.index_select(0, idx_tensor) \
                                          * c_tensor.index_select(0, idx_tensor)

                # u l2 error
                if self.do_u_l2_error:
                    self.u_l2_error_fht[idx] = self.u_l2_error_t[idx]

            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
               break

        self.loss = np.mean(loss_traj)
        self.ipa_loss = torch.mean(ipa_loss_traj)

        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)

        return True

    def sample_loss_re_nn(self, device):
        self.preallocate_fht()
        #self.preallocate_integrals()

        ## nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # initialize loss and ipa loss for the trajectories
        #loss_traj = np.zeros(self.N)
        re_loss_traj = torch.zeros(self.N)
        #phi_det = torch.zeros(self.N)
        phi = torch.zeros(self.N)

        # initialize trajectory
        xt_tensor = self.initial_position(tensor=True)

        # control
        ut_tensor = model.forward(xt_tensor)

        # initialize deterministic and stochastic integrals at time t
        #self.initialize_running_integrals()

        for k in np.arange(1, self.k_lim + 1):

            # get Brownian increment
            dB_tensor = self.brownian_increment(tensor=True)
            #dB = dB_tensor.detach().numpy()


            # sde update
            gradient_tensor = self.gradient(xt_tensor, tensor=True)
            controlled_gradient = gradient_tensor - np.sqrt(2) * ut_tensor
            xt_tensor = self.sde_update(xt_tensor, controlled_gradient, dB_tensor, tensor=True)
            xt = xt_tensor.detach().numpy()

            # control
            ut_tensor = model.forward(xt_tensor)
            #ut_tensor_det = ut_tensor.detach()
            #ut = ut_tensor_det.numpy()

            # update statistics
            #ut_norm_det = torch.linalg.norm(ut_tensor_det, axis=1)
            #phi_det = phi_det + ((1 + 0.5 * (ut_norm_det ** 2)) * self.dt).reshape(self.N,)

            #ut_norm = torch.linalg.norm(ut_tensor, axis=1)
            #phi = phi + ((1 + 0.5 * (ut_norm ** 2)) * self.dt).reshape(self.N,)
            phi = phi + ((1 + 0.5 * torch.sum(ut_tensor ** 2, dim=1)) * self.dt).reshape(self.N)

            # stochastic and deterministic integrals 
            #self.update_integrals(ut, dB)

            # get indices of trajectories which are new in the target set
            idx = self.get_idx_new_in_target_set(xt)

            if idx.shape[0] != 0:

                # save first hitting time and Girsanov Martingale terms
                self.fht[idx] = k * self.dt
                #self.stoch_int_fht[idx] = self.stoch_int_t[idx]
                #self.det_int_fht[idx] = self.det_int_t[idx]

                # get tensor indices if there are new trajectories 
                idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)

                # save loss and ipa loss for the arrived trajectorries
                #loss_traj[idx] = phi_det.numpy()[idx]
                re_loss_traj[idx_tensor] = phi.index_select(0, idx_tensor)


            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
               break

        #self.loss = np.mean(loss_traj)
        self.re_loss = torch.mean(re_loss_traj)

        #self.compute_I_u_statistics()

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
        ''' compute mean, variance and relative error of the sampled quantity of interest
        '''
        if self.problem_name == 'langevin_stop-t':
            I = np.exp(- self.fht)
        elif self.problem_name == 'langevin_det-t':
            I = np.exp(- self.work)
        self.mean_I, \
        self.var_I, \
        self.re_I = self.compute_mean_variance_and_rel_error(I)

    def compute_I_u_statistics(self):
        #TODO: compute mean of M_fht
        M_fht = np.exp(
            - np.sqrt(self.beta) * self.stoch_int_fht
            - (self.beta / 2) * self.det_int_fht
        )

        # compute mean, variance and relative error of I_u
        I_u = np.exp(
            - self.fht
            - np.sqrt(self.beta) * self.stoch_int_fht
            - (self.beta / 2) * self.det_int_fht
        )
        self.mean_I_u, \
        self.var_I_u, \
        self.re_I_u = self.compute_mean_variance_and_rel_error(I_u)

    def compute_I_u_statistics_det(self):
        #TODO: compute mean of M_fht
        M_fht = np.exp(
            - np.sqrt(self.beta) * self.stoch_int_t
            - (self.beta / 2) * self.det_int_t
        )

        # compute mean, variance and relative error of I_u
        I_u = np.exp(
            - self.work
            - np.sqrt(self.beta) * self.stoch_int_t
            - (self.beta / 2) * self.det_int_t
        )
        self.mean_I_u, \
        self.var_I_u, \
        self.re_I_u = self.compute_mean_variance_and_rel_error(I_u)

    def compute_loss(self):
        pass

    def save(self):

        # set file name
        if not self.is_controlled:
            file_name = 'mc-sampling.npz'
        elif self.is_controlled:
            file_name = 'is.npz'
        else:
            return

        # set file path
        file_path = os.path.join(self.dir_path, file_name)

        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # create dictionary
        files_dict = {}

        # add sampling attributes
        files_dict['seed'] = self.seed
        files_dict['xzero'] = self.xzero
        files_dict['N'] = self.N

        # Euler-Marujama
        files_dict['dt'] = self.dt
        files_dict['k_lim'] =self.k_lim

        # fht
        if self.problem_name == 'langevin_stop-t':
            files_dict['been_in_target_set'] = self.been_in_target_set
            files_dict['fht'] = self.fht
            self.k = int(np.max(self.fht) / self.dt)
            files_dict['k'] = self.k
            files_dict['N_arrived'] = self.N_arrived
            files_dict['first_fht'] = self.first_fht
            files_dict['last_fht'] = self.last_fht
            files_dict['mean_fht'] = self.mean_fht
            files_dict['var_fht'] = self.var_fht
            files_dict['re_fht'] = self.re_fht

        # quantity of interest
        files_dict['mean_I'] = self.mean_I
        files_dict['var_I'] = self.var_I
        files_dict['re_I'] = self.re_I

        # reweighted quantity of interest
        if self.is_controlled:
            files_dict['mean_I_u'] = self.mean_I_u
            files_dict['var_I_u'] = self.var_I_u
            files_dict['re_I_u'] = self.re_I_u

        # u l2 error
        if self.do_u_l2_error:
            files_dict['u_l2_error'] = self.u_l2_error

        # save trajectory
        if self.save_trajectory:
            files_dict['traj'] = self.traj

        # computational time
        files_dict['ct'] = self.ct

        # save npz file
        np.savez(file_path, **files_dict)


    def load(self):

        # set file name
        if not self.is_controlled:
            file_name = 'mc-sampling.npz'
        elif self.is_controlled:
            file_name = 'is.npz'
        else:
            return

        # set file path
        file_path = os.path.join(self.dir_path, file_name)

        # load data
        try:
            data = np.load(file_path, allow_pickle=True)
            for npz_file_name in data.files:
                if data[npz_file_name].ndim == 0:
                    setattr(self, npz_file_name, data[npz_file_name][()])
                else:
                    setattr(self, npz_file_name, data[npz_file_name])
            return True
        except:
            if not self.is_controlled:
                msg = 'no mc-sampling found with dt={:.4f} and N={:.0e}' \
                      ''.format(self.dt, self.N)
            else:
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

        if self.seed is not None:
            f.write('seed: {:2.1f}\n'.format(self.seed))
        else:
            f.write('seed: -\n')

    def write_report(self):
        '''
        '''

        # set file name
        if not hasattr(self, 'n_batch_samples'):
            file_name = 'report.txt'
        else:
            file_name = 'report_batch-wise.txt'

        # set file path
        file_path = os.path.join(self.dir_path, file_name)

        # write file
        f = open(file_path, "w")

        #self.write_setting(f)
        self.write_euler_maruyama_parameters(f)
        self.write_sampling_parameters(f)

        if self.is_controlled and not self.is_optimal:
            self.ansatz.write_ansatz_parameters(f)

        if self.problem_name == 'langevin_stop-t':
            f.write('\nStatistics\n')

            f.write('trajectories which arrived: {:2.2f} %\n'
                    ''.format(100 * self.N_arrived / self.N))

            if self.N_arrived < self.N:
                f.write('used time steps: {:,d}\n\n'.format(self.k_lim))

                # close file
                f.close()

                # print file
                f = open(file_path, 'r')
                print(f.read())
                f.close()
                return
            else:
                f.write('used time steps: {:,d}\n\n'.format(self.k))

            f.write('First hitting time (fht)\n')
            f.write('first fht = {:2.3f}\n'.format(self.first_fht))
            f.write('last fht = {:2.3f}\n'.format(self.last_fht))
            f.write('m_N(fht) = {:2.3f}\n'.format(self.mean_fht))
            f.write('s_N^2(fht) = {:2.3f}\n'.format(self.var_fht))
            f.write('re_N(fht) = {:2.3f}\n'.format(self.re_fht))
            f.write('mc-error(fht) = {:2.3f}\n\n'.format(np.sqrt(self.var_fht / self.N)))

            f.write('First hitting time step (fhts)\n\n')
            f.write('m_N(fhts) = {:2.3f}\n'.format(self.mean_fht / self.dt))
            #f.write('s_N^2(fhts) = {:2.3f}\n'.format(self.var_fht / (self.dt **2)))
            #f.write('re_N(fhts) = {:2.3f}\n\n'.format(self.re_fht))

        if not self.is_controlled:
            f.write('\nQuantity of interest\n')
            f.write('m_N(I) = {:2.3e}\n'.format(self.mean_I))
            f.write('s_N^2(I) = {:2.3e}\n'.format(self.var_I))
            f.write('re_N(I) = {:2.3e}\n'.format(self.re_I))
            f.write('mc-error(I) = {:2.3e}\n'.format(np.sqrt(self.var_I / self.N)))
            f.write('-log(m_N(I)) = {:2.3e}\n\n'.format(-np.log(self.mean_I)))

        else:
            f.write('\nReweighted Quantity of interest\n')
            f.write('m_N(I^u) = {:2.3e}\n'.format(self.mean_I_u))
            f.write('s_N^2(I^u) = {:2.3e}\n'.format(self.var_I_u))
            f.write('re_N(I^u) = {:2.3e}\n'.format(self.re_I_u))
            f.write('mc-error(I^u) = {:2.3e}\n'.format(np.sqrt(self.var_I_u / self.N)))
            f.write('-log(m_N(I^u)) = {:2.3e}\n\n'.format(-np.log(self.mean_I_u)))

        h, m, s = get_time_in_hms(self.ct)
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

        # bias potential
        if not self.is_controlled:
            # bias potential and value function
            self.grid_bias_potential = np.zeros(self.Nx)
            self.grid_value_function = np.zeros(self.Nx)

        # gaussian ansatz
        elif self.is_controlled and self.ansatz is not None:

            # set value function constant
            if self.potential_name == 'nd_well':
                self.ansatz.set_value_function_constant_corner()
            elif self.potential_name == 'nd_well_asym':
                self.ansatz.set_value_function_target_set()

            # bias potential and value function
            self.grid_bias_potential = self.bias_potential(x).reshape(self.Nx)
            self.grid_value_function = self.ansatz.value_function(x).reshape(self.Nx)

        # controlled potential
        if self.grid_bias_potential is not None:
            # controlled potential
            self.grid_controlled_potential = self.grid_potential + self.grid_bias_potential

    def get_grid_value_function_i(self, i=0, x_j=0.):
        ''' computes the value of the value function and the bias potential along the i-th
            coordinate evaluated at x_j for all j != i.
        '''

        # inputs
        x = x_j * np.ones((self.Nh, self.n))
        x[:, i] = self.domain_i_h

        # potential
        self.grid_potential_i = self.potential(x)

        # bias potential
        if not self.is_controlled:
            # bias potential and value function
            self.grid_bias_potential_i = np.zeros(self.Nh)
            self.grid_value_function_i = np.zeros(self.Nh)

        # gaussian ansatz
        elif self.is_controlled and self.ansatz is not None:

            # bias potential and value function
            self.grid_bias_potential_i = self.bias_potential(x)
            self.grid_value_function_i = self.ansatz.value_function(x)

        # controlled potential
        if self.grid_bias_potential_i is not None:
            # controlled potential
            self.grid_controlled_potential_i = self.grid_potential_i + self.grid_bias_potential_i

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

        # nn control
        elif self.is_controlled and self.nn_func_appr is not None:
            inputs = torch.tensor(x, dtype=torch.float)
            control_flattened = self.nn_func_appr.model(inputs).detach().numpy()
            self.grid_control = control_flattened.reshape(self.domain_h.shape)

        # controlled drift
        self.grid_controlled_drift = - self.grid_gradient + np.sqrt(2) * self.grid_control

    def get_grid_control_i(self, i=0, x_j=0., k=None):
        ''' computes the value of the control along the i-th coordinate evaluated at x_j
            for all j != i. In case of working with the deterministic time horizont framework
            the control is evaluated at time k
        '''
        # check if time step k is given for the deterministic time horizont problem
        if self.problem_name == 'langevin_det-t':
            assert k is not None, ''

        # inputs
        x = x_j * np.ones((self.Nh, self.n))
        x[:, i] = self.domain_i_h

        # gaussian ansatz control
        if self.ansatz is not None:

            # evaluate control
            control = self.ansatz.control(x)

        # nn control
        if self.nn_func_appr is not None:

            # tensorize inputs
            inputs = torch.tensor(x, dtype=torch.float)

            # evaluate control
            if self.problem_name == 'langevin_det-t':
                control = self.nn_func_appr.model(k, inputs).detach().numpy()
            elif self.problem_name == 'langevin_stop-t':
                control = self.nn_func_appr.model(inputs).detach().numpy()

        # get i-th coordinate
        self.grid_control_i = control[:, i]

    def plot_trajectory(self, n_iter_avg=1, dir_path=None, file_name='trajectory'):
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        # initialize figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )

        # discretized time array
        traj_fhs = self.traj.shape[0]
        traj_fht = traj_fhs * self.dt
        x = np.linspace(0, traj_fht, traj_fhs)[n_iter_avg - 1:]

        # coordinates
        traj = np.moveaxis(self.traj, 0, -1)

        y = np.empty((self.n, traj_fhs - n_iter_avg + 1))
        for i in range(self.n):
            y[i] = np.convolve(traj[i], np.ones(n_iter_avg) / n_iter_avg, mode='valid')

        labels = [r'$x_{:d}$'.format(i+1) for i in np.arange(self.n)]

        fig.set_title(r'trajectory')
        fig.set_xlabel(r'$t$')
        fig.set_ylim(-1.5, 1.5)
        plt.subplots_adjust(left=0.14, right=0.96, bottom=0.12)
        fig.plot(x, y, labels=labels)
