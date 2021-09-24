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

    def __init__(self, potential_name, n, alpha, beta, target_set=None,
                 domain=None, h=None, is_controlled=None, is_optimal=None,
                 is_batch=False):
        '''
        '''

        super().__init__(potential_name, n, alpha, beta,
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

        # batch sampling
        self.is_batch = is_batch
        self.batch_id = None

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
        self.grad_loss = None
        self.ipa_loss = None
        self.logvar_loss = None

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

    def sample_not_controlled(self):
        self.start_timer()
        self.preallocate_fht()

        # initialize xt
        xt = self.initial_position()

        if self.save_trajectory:

            # preallocate array for the trajectory and save initial position
            self.traj = np.empty((self.k_lim + 1, self.n))
            self.traj[0] = xt[0, :]

        for k in np.arange(1, self.k_lim + 1):

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
            self.Nx = sol_hjb.Nx

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

        breakpoint()
        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)
        self.stop_timer()

    def sample_optimal_controlled(self, h):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # load hjb solver and get the "optimal" control
        sol_hjb = self.get_hjb_solver(h)
        u_opt = sol_hjb.u_opt
        self.Nx = sol_hjb.Nx

        for k in np.arange(1, self.k_lim +1):

            # control at xt
            idx_xt = self.get_index_vectorized(xt)
            ut = u_opt[idx_xt]

            # compute gradient
            gradient = self.tilted_gradient(xt, ut)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dB)

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
                ut = self.ansatz.control(x[k - 1])

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
            self.Nx = sol_hjb.Nx

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

            # ansatz basis for the control and control
            bt = self.ansatz.basis_control(xt)
            ut = self.ansatz.control(xt)

            # get Brownian increment
            dB = self.brownian_increment()

            # update running gradient of phi and running gradient of S
            grad_phi_t += np.sum(ut[:, np.newaxis, :] * bt, axis=2) * self.dt
            grad_S_t -= np.sqrt(self.beta) * np.sum(dB[:, np.newaxis, :] * bt, axis=2)

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
                loss_traj[idx] = self.fht[idx] + 0.5 * self.det_int_fht[idx]
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
            self.Nx = sol_hjb.Nx

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

            ut_norm = torch.linalg.norm(ut_tensor, axis=1)
            phi_t = phi_t + ((1 + 0.5 * (ut_norm ** 2)) * self.dt).reshape(self.N,)

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

        # compute loss
        a = torch.mean(phi_fht)
        self.loss = a.detach().numpy()

        # compute ipa loss
        b = - torch.mean(phi_fht.detach() * S_fht)
        self.ipa_loss = a + b

        self.compute_I_u_statistics()
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
            self.Nx = sol_hjb.Nx

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

    def sample_loss_logvar_nn(self, device):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        # preallocate process y at fht
        y_fht = torch.empty(self.N)

        if self.do_u_l2_error:

            # load hjb solution
            sol_hjb = self.get_hjb_solver()
            self.u_hjb = sol_hjb.u_opt
            self.Nx = sol_hjb.Nx

            # preallocate l2 error
            self.preallocate_l2_error()

        # nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # initialize phi
        phi_t = np.zeros(self.N)
        phi_fht = np.empty(self.N)

        # initialize trajectory of the process xt and process yt
        xt = self.initial_position()
        yt = torch.zeros(self.N).reshape(self.N)

        # initialize z sum
        zt_sum = np.zeros(self.N)

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

            # z process
            zt_tensor = - model.forward(xt_tensor)

            # sde update xt
            controlled_gradient = self.gradient(xt) - np.sqrt(2) * ut
            xt = self.sde_update(xt, controlled_gradient, dB)

            # norm
            ut_norm_tensor = torch.linalg.norm(ut_tensor, axis=1)
            ut_norm = ut_norm_tensor.detach().numpy()

            a = (
                - 1
                + 0.5 * ut_norm_tensor **2
                + torch.matmul(
                    torch.unsqueeze(zt_tensor, 1),
                    torch.unsqueeze(ut_tensor, 2),
                ).reshape(self.N,)
            ) * self.dt
            b = torch.matmul(
                    torch.unsqueeze(zt_tensor, 1),
                    torch.unsqueeze(dB_tensor, 2),
                ).reshape(self.N,)

            # sde update yt
            yt = yt + a + b

            # update phi
            phi_t += ((1 + 0.5 * (ut_norm ** 2)) * self.dt).reshape(self.N,)

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

                # save phi 
                phi_fht[idx] = phi_t[idx]

                # get tensor indices if there are new trajectories 
                idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)

                # save y
                y_fht[idx_tensor] = yt.index_select(0, idx_tensor)

                # u l2 error
                if self.do_u_l2_error:
                    self.u_l2_error_fht[idx] = self.u_l2_error_t[idx]

            # stop if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
               break

        # compute loss
        self.loss = np.mean(phi_fht)


        # compute logvar loss
        self.logvar_loss = torch.var(y_fht)

        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)

        self.stop_timer()

        return True


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
        # compute mean, variance and relative error of I
        I = np.exp(- self.fht)
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

    def compute_loss(self):
        pass

    def save(self):

        # set file name
        if not self.is_controlled and not self.is_batch:
            file_name = 'mc-sampling.npz'
        elif not self.is_controlled and self.is_batch:
            file_name = 'mc-sampling_batch-{}.npz'.format(str(self.batch_id))
        elif self.is_controlled:
            file_name = 'is.npz'
        else:
            return

        # set file path
        file_path = os.path.join(self.dir_path, file_name)

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
        if not self.is_controlled and not self.is_batch:
            file_name = 'mc-sampling.npz'
        elif not self.is_controlled and self.is_batch:
            file_name = 'mc-sampling_batch-{}.npz'.format(str(self.batch_id))
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
        if self.seed is None:
            file_name = 'report.txt'
        else:
            file_name = 'report_batch-{}.txt'.format(str(self.seed))

        # set file path
        file_path = os.path.join(self.dir_path, file_name)

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
            f.write('-log(E[exp(- fht)]) = {:2.3e}\n\n'.format(-np.log(self.mean_I)))

        else:
            f.write('\nReweighted Quantity of interest\n')
            f.write('E[exp(- fht) * M_fht] = {:2.3e}\n'.format(self.mean_I_u))
            f.write('Var[exp(- fht) * M_fht] = {:2.3e}\n'.format(self.var_I_u))
            f.write('RE[exp(- fht) * M_fht] = {:2.3e}\n\n'.format(self.re_I_u))
            f.write('-log(E[exp(- fht) * M_fht]) = {:2.3e}\n\n'.format(-np.log(self.mean_I_u)))

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

        if not self.is_controlled:
            # bias potential and value function
            self.grid_bias_potential = np.zeros(self.Nx)
            self.grid_value_function = np.zeros(self.Nx)

        elif self.is_controlled and self.ansatz is not None:
            # set value f constant
            self.ansatz.set_value_function_constant_corner()

            # bias potential and value function
            self.grid_bias_potential = self.bias_potential(x).reshape(self.Nx)
            self.grid_value_function = self.ansatz.value_function(x).reshape(self.Nx)

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
