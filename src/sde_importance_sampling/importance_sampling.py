import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.utils_path import get_not_controlled_dir_path, \
                                               get_controlled_dir_path, \
                                               get_time_in_hms

class Sampling(object):
    '''

    Attributes
    ----------

    # general
    sde: langevinSDE object
        overdamped langevin sde object
    dir_path: str
        directory path for the hjb solver

    # sampling
    is_controlled: bool
        True if the sampling is controlled
    is_optimal: bool
        True if the control is the solution of the hjb equation
    seed: int
        random seed
    xzero:

    K: int
        number of trajectories
    xt: array
        position of the trajectory
    save_trajectory: bool
        True if
    traj: array
        trajectory of the particles

    # Euler-Marujama
    dt: float
        time step discretization
    k_lim: int
        number of maximal time steps
    k: int
        number of time steps used

    # ansatz functions (gaussians) and coefficients
    ansatz: object
        gaussian ansatz object

    # function approximation by a neural network
    nn_func_appr: object
        function approximation object

    # grid evalutations
    grid_potential: array
        potential evaluated in the discretized domain
    grid_bias_potential: array
        bias potential evaluated in the discretized domain
    grid_perturbed_potential: array
        perturbed potential evaluated in the discretized domain
    grid_value_function: array
        value function evaluated in the discretized domain
    grid_gradient: array
        gradient evaluated in the discretized domain
    grid_control: array
        control evaluated in the discretized domain
    grid_perturbed_drift: array
        perturbed drift evaluated in the discretized domain

    # variables

    # deterministic and stochastic integrals
    stoch_int_t: array
        stochastic integral from 0 to t.
    stoch_int_fht: array
        stochastic integral from 0 to fht.
    det_int_t: array
        deterministic integral from 0 to t.
    det_int_fht: array
        deterministic integral from 0 to fht.

    # trajectories which arrived
    n_traj_arrived: int
        number of trajectories which arrived in the target set
    been_in_target_set: bool array
        True if trajectory arrived in the target set


    # first hitting time
    fht: array
        first hitting times of the trajectories in the target set
    first_fht: float
        first hitting time of the first trajectory
    last_fht: float
        last hitting time of the last trajectory
    mean_fht: float
        mean of the first hitting times
    var_fht: float
        variance of the first hitting times
    re_fht: float
        relative error of the first hitting times

    # quantity of interest
    mean_I: float
        mean of the quantity of interest
    var_I: float
        variance of the quantity of interest
    re_I: float
        relative error of the quantity of interest

    mean_I_u: float
        mean of the importance sampling quantity of interest
    var_I_u: float
        variance of the importance sampling quantity of interest
    re_I_u: float
        relative error of the importance sampling quantity of interest

    # loss and its gradient
    loss:

    var_loss:

    grad_loss:

    ipa_loss:


    # control l2 error wrt the hjb solution
    do_u_l2_error: bool
        True if the l2 error along the trajectories is computed
    u_l2_error: array
        l2 error between the control and the reference control along the trajectory

    # computational time
    ct_initial: float
        initial computational time
    ct_time: float
        final computational time
    ct: float
        computational time

    # dir_path
    dir_path = None

    Methods
    -------
    __init__()

    set_not_controlled_dir_path()

    set_controlled_dir_path(parent_dir_path)

    bias_potential(x, theta=None)

    bias_gradient(u)

    tilted_potential(x)

    perturbed_gradient(x, u)

    brownian_increment(tensor=False)

    set_sampling_parameters(dt, k_lim, xzero, N, seed=None)

    start_timer()

    stop_timer()

    preallocate_fht()

    preallocate_girsanov_martingale_terms()

    preallocate_integrals()

    preallocate_l2_error()

    initial_position(tensor=False)

    initialize_running_integrals()

    initialize_running_l2_error()

    initialize_running_l2_error()

    update_integrals(ut, dB)

    update_running_l2_error(xt, ut)

    update_running_l2_error_det(k, xt, ut)

    sde_update(x, gradient, dbt, tensor=False)

    get_idx_new_in_target_set(x)

    sample_not_controlled_det()

    sample_not_controlled()

    sample_controlled()

    sample_optimal_controlled_det(h, dt)

    sample_optimal_controlled(h)

    sample_meta()

    sample_loss_ipa_ansatz()

    sample_loss_ipa_nn(device)

    sample_loss_ipa_nn_det(device)

    sample_loss_ipa2_nn(device)

    sample_loss_re_nn(device)

    compute_mean_variance_and_rel_error(x)

    compute_fht_statistics()

    compute_I_statistics()

    compute_I_u_statistics()

    compute_I_u_statistics_det()

    compute_loss()

    save()

    load()

    write_euler_maruyama_parameters(f)

    write_sampling_parameters(f)

    write_report()

    get_grid_value_function()

    integrate_value_function_1d()

    get_grid_value_function_i(i=0, x_j=0.)

    get_grid_control()

    get_grid_control_i(i=0, x_j=0., k=None)

    plot_trajectory(n_iter_avg=1, dir_path=None, file_name='trajectory')


    '''

    def __init__(self, sde, h=None, is_controlled=True, is_optimal=False,
                 do_u_l2_error=False, save_trajectory=False):
        ''' init method

        Parameters
        ----------
        sde: langevinSDE object
            overdamped langevin sde object
        h: float, optional
            step size
        is_controlled: bool, optional
            True if the sampling is controlled
        is_optimal: bool, optimal
            True if the control is the solution of the hjb equation
        do_u_l2_error: bool, optimal
            True if the L2-error between the control and the reference solution along the
            trajectory is computed
        save_trajectory: bool, optimal
            True if the trajectory should be saved in an array
        '''

        # overdamped langevin sde
        self.sde = sde
        self.sde.h = h

        # sampling
        self.is_controlled = is_controlled
        self.is_optimal = is_optimal
        self.save_trajectory = save_trajectory
        self.do_u_l2_error = do_u_l2_error

    def set_not_controlled_dir_path(self):
        ''' computes not controlled directory path
        '''
        assert self.dt is not None, ''
        assert self.K is not None, ''

        self.dir_path = get_not_controlled_dir_path(
            self.sde.settings_dir_path,
            self.dt,
            self.K,
            self.seed,
        )

    def set_controlled_dir_path(self, parent_dir_path):
        ''' computes controlled directory path
        '''
        assert self.dt is not None, ''
        assert self.K is not None, ''

        self.dir_path = get_controlled_dir_path(
            parent_dir_path,
            self.dt,
            self.K,
            self.seed,
        )

    def bias_potential(self, x, theta=None):
        ''' computes the bias potential at x

        Parameters
        ----------
        x: (K, d)-array
            position
        theta: (m,)-array
            parameters

        Returns
        -------
        array
            bias potential evaluated at the x
        '''
        assert self.ansatz is not None, ''
        return self.ansatz.value_function(x, theta) * self.sde.sigma**2

    def bias_gradient(self, u):
        ''' computes the bias gradient at x

        Parameters
        ----------
        u: (K, d)-array
            control at x

        Returns
        -------
        array
            bias gradient evaluated at the x
        '''
        return - self.sde.sigma * u

    def perturbed_potential(self, x):
        ''' computes the perturbed potential at x

        Parameters
        ----------
        x: (K, d)-array
            position/s
        theta: (m,)-array
            parameters

        Returns
        -------
        array
            perturned potential at x
        '''
        return self.sde.potential(x) + self.bias_potential(x, theta)

    def perturbed_gradient(self, x, u):
        ''' computes the perturbed gradient at x

        Parameters
        ----------
        x: (K, d)-array
            position/s
        u: (K, d)-array
            control at x

        Returns
        -------
        array
            perturbed gradient at x
        '''
        assert x.shape == u.shape

        return self.sde.gradient(x) + self.bias_gradient(u)

    def brownian_increment(self, tensor=False):
        ''' computes brownian increments

        Parameters
        ----------
        tensor: bool, optional
            True if returned array is a pytorch tensor

        Returns
        -------
        (K, d)-array
            brownian increment
        '''
        dbt = np.sqrt(self.dt) \
            * np.random.normal(0, 1, self.K * self.sde.d).reshape(self.K, self.sde.d)
        if tensor:
            dbt = torch.tensor(dB, dtype=torch.float32)

        return dbt

    def set_sampling_parameters(self, dt=0.01, k_lim=10**8, xzero=None, K=1000, seed=None):
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
        self.K = K

    def start_timer(self):
        ''' start timer
        '''
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        ''' stop timer
        '''
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def preallocate_fht(self):
        '''
        '''
        assert self.K is not None, ''

        # boolean array telling us if a trajectory have been in the target set
        self.been_in_target_set = np.repeat([False], self.K).reshape(self.K, 1)

        # first hitting time of each trajectory
        self.fht = np.empty(self.K)

    def preallocate_girsanov_martingale_terms(self):
        ''' preallocates Girsanov Martingale terms, M_fht = e^(M1_fht + M2_fht)
        '''
        assert self.K is not None, ''

        self.M1_fht = np.empty(self.K)
        self.M2_fht = np.empty(self.K)

    def preallocate_integrals(self):
        '''
        '''
        assert self.K is not None, ''

        self.stoch_int_fht = np.empty(self.K)
        self.det_int_fht = np.empty(self.K)

    def preallocate_l2_error(self):
        '''
        '''
        assert self.K is not None, ''

        self.u_l2_error_fht = np.empty(self.K)


    def initial_position(self, tensor=False):
        ''' returns same initial posicion for all trajectories
        '''
        x_init = np.full((self.K, self.sde.d), self.xzero, dtype=np.float64)
        if tensor:
            x_init = torch.tensor(x_init, dtype=torch.float32)

        return x_init


    def initialize_running_integrals(self):
        '''
        '''
        assert self.K is not None, ''

        self.stoch_int_t = np.zeros(self.K)
        self.det_int_t = np.zeros(self.K)

    def initialize_running_l2_error(self):
        '''
        '''
        assert self.K is not None, ''

        self.u_l2_error_t = np.zeros(self.K)

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
        idx_xt = self.sde.get_index_vectorized(xt)
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

    def sde_update(self, xt, gradient, dbt, tensor=False):
        ''' updates position of the trajectories following the overdamped lanvevin sde

        Parameters
        ----------
        xt: (K, d)-array
            position
        gradient: (K, d)-array
            (perturbed) gradient of x
        dbt: (K,)-array
            brownian increment
        tensor: bool, optional
            True if the return array should be a pytorch tensor

        Returns
        -------
        (K, d)-array
            updated position
        '''
        # compute drift term
        drift = - gradient * self.dt

        # compute diffusion term
        if not tensor:
            diffusion = np.dot(dbt, self.sde.sigma * np.eye(self.sde.d))
        else:
            diffusion = torch.mm(dbt, self.sde.sigma * torch.eye(self.sde.d))

        return xt + drift + diffusion

    def get_idx_new_in_target_set(self, x):
        ''' computes the indices of the trajectories which are new in the target set and
            updates the been_in_target_set array.

        Parameters
        ----------
        x: (K, d)-array
            position

        Returns
        -------
        array
            indices of the trajectories
        '''
        # boolean array telling us if a trajectory is in the target set
        is_in_target_set = (
            (x >= self.sde.target_set[:, 0]) &
            (x <= self.sde.target_set[:, 1])
        ).all(axis=1).reshape(self.K, 1)

        # indices of trajectories new in the target set
        idx = np.where(
            (is_in_target_set == True) &
            (self.been_in_target_set == False)
        )[0]

        # update list of indices whose trajectories have been in the target set
        self.been_in_target_set[idx] = True

        return idx

    #TODO! revise det time horizont
    def sample_not_controlled_det(self):
        self.start_timer()

        # initialize xt
        xt = self.initial_position()

        for k in np.arange(1, self.k_lim + 1):

            # compute gradient
            gradient = self.sde.gradient(xt)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dB)

        # save work functional
        self.work = self.sde.g(xt)

        self.stop_timer()

    def sample_not_controlled(self):
        self.start_timer()
        self.preallocate_fht()

        # initialize xt
        xt = self.initial_position()

        # preallocate array for the trajectory
        if self.save_trajectory:
            self.traj = np.empty((self.k_lim + 1, self.sde.d))

        # start trajectories
        for k in np.arange(self.k_lim + 1):

            # save position of first trajectory at time k
            if self.save_trajectory:
                self.traj[k] = xt[0, :]

            # get indices from the trajectories which are new in target
            idx = self.get_idx_new_in_target_set(xt)

            # save first hitting time
            if idx.shape[0] != 0:
                self.fht[idx] = k * self.dt

            # break if all trajectories have arrived to the target set
            if self.been_in_target_set.all() == True:
                break

            # compute gradient
            gradient = self.sde.gradient(xt)

            # get Brownian increment
            dbt = self.brownian_increment()

            # sde update
            xt = self.sde_update(xt, gradient, dbt)

        self.stop_timer()

    def sample_controlled(self):
        self.start_timer()
        self.preallocate_fht()
        self.preallocate_integrals()

        if self.do_u_l2_error:

            # load hjb solution
            sol_hjb = self.get_hjb_solver()
            self.u_hjb = sol_hjb.u_opt

            # preallocate l2 error
            self.preallocate_l2_error()

        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        # start trajectories
        for k in np.arange(self.k_lim + 1):

            # control at xt
            if self.ansatz is not None:
                ut = self.ansatz.control(xt)
            elif self.nn_func_appr is not None:
                xt_tensor = torch.tensor(xt, dtype=torch.float)
                ut_tensor = self.nn_func_appr.model.forward(xt_tensor)
                ut = ut_tensor.detach().numpy()

            # get Brownian increment
            dbt = self.brownian_increment()

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dbt)

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

            # compute gradient
            gradient = self.perturbed_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dbt)


        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        if self.do_u_l2_error:
            self.u_l2_error = np.mean(self.u_l2_error_fht)
        self.stop_timer()

    #TODO! revise det time horizont
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
            gradient = self.perturbed_gradient(xt, ut)

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
        sol_hjb = self.sde.get_hjb_solver(h)
        u_opt = sol_hjb.u_opt

        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # preallocate array for the trajectory
        if self.save_trajectory:
            self.traj = np.empty((self.k_lim + 1, self.sde.d))

        # start trajectories
        for k in np.arange(self.k_lim + 1):

            # save position of first trajectory at time k
            if self.save_trajectory:
                self.traj[k] = xt[0, :]

            # control at xt
            idx_xt = self.sde.get_index_vectorized(xt)
            ut = u_opt[idx_xt]

            # get Brownian increment
            dbt = self.brownian_increment()

            # stochastic and deterministic integrals 
            self.update_integrals(ut, dbt)

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

            # compute gradient
            controlled_gradient = self.perturbed_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, controlled_gradient, dbt)

        self.compute_fht_statistics()
        self.compute_I_u_statistics()
        self.stop_timer()


    def sample_meta(self):
        self.preallocate_fht()

        # preallocate trajectory
        x = np.empty((self.k_lim + 1, self.K, self.sde.d))

        # initialize xt
        x[0] = self.initial_position()

        # start trajectory
        for k in np.arange(1, self.k_lim + 1):

            # compute gradient
            if not self.is_controlled:
                gradient = self.sde.gradient(x[k - 1])

            # or compute controlled gradient
            else:

                # control at xt
                if self.ansatz is not None:
                    ut = self.ansatz.control(x[k - 1])
                elif self.nn_func_appr is not None:
                    xt_tensor = torch.tensor(x[k - 1], dtype=torch.float)
                    ut_tensor = self.nn_func_appr.model.forward(xt_tensor)
                    ut = ut_tensor.detach().numpy()

                # compute gradient
                gradient = self.perturbed_gradient(x[k - 1], ut)

            # get Brownian increment
            dB = self.brownian_increment()

            # sde update
            x[k] = self.sde_update(x[k - 1], gradient, dB)

            # update been in target set
            _ = self.get_idx_new_in_target_set(x[k])

            # check if the half of the trajectories have arrived to the target set
            if np.sum(self.been_in_target_set) >= self.K / 2:
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
        loss_traj = np.empty(self.K)
        grad_loss_traj = np.empty((self.K, m))

        # initialize running gradient of phi and running gradient of S
        grad_phi_t = np.zeros((self.K, m))
        grad_S_t = np.zeros((self.K, m))

        # initialize xt
        xt = self.initial_position()

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        # start trajectories
        for k in np.arange(self.k_lim + 1):

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

            # compute gradient
            gradient = self.perturbed_gradient(xt, ut)

            # sde update
            xt = self.sde_update(xt, gradient, dB)

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
            self.h = sol_hjb.h

            # preallocate l2 error
            self.preallocate_l2_error()

        # nn model
        model = self.nn_func_appr.model

        # number of flattened parameters
        m = model.d_flat

        # initialize phi and S
        phi_t = torch.zeros(self.K)
        phi_fht = torch.empty(self.K)
        S_t = torch.zeros(self.K)
        S_fht = torch.empty(self.K)

        # initialize trajectory
        xt = self.initial_position()
        xt_tensor = torch.tensor(xt, dtype=torch.float32)

        # initialize deterministic and stochastic integrals at time t
        self.initialize_running_integrals()

        # initialize l2 error at time t
        if self.do_u_l2_error:
            self.initialize_running_l2_error()

        # start trajectories
        for k in np.arange(self.k_lim + 1):

            # control
            ut_tensor = model.forward(xt_tensor)
            ut = ut_tensor.detach().numpy()

            # get Brownian increment and tensorize it
            dB = self.brownian_increment()
            dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32)

            # update running phi
            ut_norm_tensor = torch.linalg.norm(ut_tensor, axis=1)
            phi_t = phi_t + ((1 + 0.5 * self.beta * (ut_norm_tensor ** 2)) * self.dt).reshape(self.K,)

            # update running discretized action
            S_t = S_t \
                - np.sqrt(self.beta) * torch.matmul(
                    torch.unsqueeze(ut_tensor, 1),
                    torch.unsqueeze(dB_tensor, 2),
                ).reshape(self.K,)

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

            # compute gradient
            controlled_gradient = self.gradient(xt) - np.sqrt(2) * ut

            # sde update
            xt = self.sde_update(xt, controlled_gradient, dB)
            xt_tensor = torch.tensor(xt, dtype=torch.float32)

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
        phi_t = torch.zeros(self.K)
        S_t = torch.zeros(self.K)

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
            phi_t = phi_t + (0.5 * self.beta * (ut_norm_tensor ** 2) * self.dt).reshape(self.K,)

            # update running discretized action
            S_t = S_t \
                - np.sqrt(self.beta) * torch.matmul(
                torch.unsqueeze(ut_tensor, 1),
                torch.unsqueeze(dB_tensor, 2),
            ).reshape(self.K,)

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
        loss_traj = np.zeros(self.K)
        ipa_loss_traj = torch.zeros(self.K)
        a_tensor = torch.zeros(self.K).to(device)
        b_tensor = torch.zeros(self.K).to(device)
        c_tensor = torch.zeros(self.K).to(device)

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
                     ).reshape(self.K,) * self.dt

            ut_norm_det = torch.linalg.norm(ut_tensor_det, axis=1)
            b_tensor = b_tensor + ((1 + 0.5 * (ut_norm_det ** 2)) * self.dt).reshape(self.K,)

            c_tensor = c_tensor \
                     - np.sqrt(self.beta) * torch.matmul(
                         torch.unsqueeze(ut_tensor, 1),
                         torch.unsqueeze(dB_tensor, 2),
                     ).reshape(self.K,)

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
        #loss_traj = np.zeros(self.K)
        re_loss_traj = torch.zeros(self.K)
        #phi_det = torch.zeros(self.K)
        phi = torch.zeros(self.K)

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
            #phi_det = phi_det + ((1 + 0.5 * (ut_norm_det ** 2)) * self.dt).reshape(self.K,)

            #ut_norm = torch.linalg.norm(ut_tensor, axis=1)
            #phi = phi + ((1 + 0.5 * (ut_norm ** 2)) * self.dt).reshape(self.K,)
            phi = phi + ((1 + 0.5 * torch.sum(ut_tensor ** 2, dim=1)) * self.dt).reshape(self.K)

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
        self.n_traj_arrived = self.fht[idx_arrived].shape[0]
        if self.n_traj_arrived != self.K:
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
        if self.sde.problem_name == 'langevin_stop-t':
            I = np.exp(- self.fht)
        elif self.sde.problem_name == 'langevin_det-t':
            I = np.exp(- self.work)
        self.mean_I, \
        self.var_I, \
        self.re_I = self.compute_mean_variance_and_rel_error(I)

    def compute_I_u_statistics(self):
        '''
        '''

        # reweighting factor
        M_fht = np.exp(
            - self.stoch_int_fht
            - (1 / 2) * self.det_int_fht
        )

        # compute mean of the reweighting factor
        self.mean_M_fht = np.mean(M_fht)

        # importance sampling quantity of interest
        I_u = np.exp(
            - self.fht
            - self.stoch_int_fht
            - (1 / 2) * self.det_int_fht
        )

        # compute mean, variance and relative error of I_u
        self.mean_I_u, \
        self.var_I_u, \
        self.re_I_u = self.compute_mean_variance_and_rel_error(I_u)

    #TODO! revise det time horizont
    def compute_I_u_statistics_det(self):
        '''
        '''

        # reweighting factor
        M_t = np.exp(
            - self.stoch_int_t
            - (1 / 2) * self.det_int_t
        )

        # compute mean of the reweighting factor
        self.mean_M_t = np.mean(M_t)

        # importance sampling quantity of interest
        I_u = np.exp(
            - self.work
            - self.stoch_int_t
            - (1 / 2) * self.det_int_t
        )

        # compute mean, variance and relative error of I_u
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
        files_dict['K'] = self.K

        # Euler-Marujama
        files_dict['dt'] = self.dt
        files_dict['k_lim'] =self.k_lim

        # fht
        if self.sde.problem_name == 'langevin_stop-t':
            files_dict['been_in_target_set'] = self.been_in_target_set
            files_dict['fht'] = self.fht
            self.k = int(np.max(self.fht) / self.dt)
            files_dict['k'] = self.k
            files_dict['n_traj_arrived'] = self.n_traj_arrived
            files_dict['first_fht'] = self.first_fht
            files_dict['last_fht'] = self.last_fht
            files_dict['mean_fht'] = self.mean_fht
            files_dict['var_fht'] = self.var_fht
            files_dict['re_fht'] = self.re_fht

        # quantity of interest
        if not self.is_controlled:
            files_dict['mean_I'] = self.mean_I
            files_dict['var_I'] = self.var_I
            files_dict['re_I'] = self.re_I

        # reweighted quantity of interest
        else:
            files_dict['stoch_int_fht'] = self.stoch_int_fht
            files_dict['det_int_fht'] = self.det_int_fht
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
                msg = 'no mc-sampling found with dt={:.4f} and K={:.0e}' \
                      ''.format(self.dt, self.K)
            else:
                msg = 'no importance-sampling found with dt={:.4f} and K={:.0e}' \
                      ''.format(self.dt, self.K)
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
        for i in range(self.sde.d):
            if i == 0:
                initial_posicion += '{:2.1f}'.format(self.xzero[i])
            else:
                initial_posicion += ', {:2.1f}'.format(self.xzero[i])
        initial_posicion += ')\n'
        f.write(initial_posicion)

        f.write('sampled trajectories: {:,d}\n'.format(self.K))

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

        # controll representation
        if self.is_controlled and not self.is_optimal and self.ansatz is not None:
            self.ansatz.write_ansatz_parameters(f)
        elif self.is_controlled and not self.is_optimal and self.nn_func_appr is not None:
            self.nn_func_appr.write_parameters(f)

        if self.sde.problem_name == 'langevin_stop-t':
            f.write('\nStatistics\n')

            f.write('trajectories which arrived: {:2.2f} %\n'
                    ''.format(100 * self.n_traj_arrived / self.K))

            if self.n_traj_arrived < self.K:
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
            f.write('m_K(fht) = {:2.3f}\n'.format(self.mean_fht))
            f.write('s_K^2(fht) = {:2.3f}\n'.format(self.var_fht))
            f.write('re_K(fht) = {:2.3f}\n'.format(self.re_fht))
            f.write('mc-error(fht) = {:2.3f}\n\n'.format(np.sqrt(self.var_fht / self.K)))

            f.write('First hitting time step (fhts)\n\n')
            f.write('m_K(fhts) = {:2.3f}\n'.format(self.mean_fht / self.dt))
            #f.write('s_K^2(fhts) = {:2.3f}\n'.format(self.var_fht / (self.dt **2)))
            #f.write('re_K(fhts) = {:2.3f}\n\n'.format(self.re_fht))

        if not self.is_controlled:
            f.write('\nQuantity of interest\n')
            f.write('m_K(I) = {:2.3e}\n'.format(self.mean_I))
            f.write('s_K^2(I) = {:2.3e}\n'.format(self.var_I))
            f.write('re_K(I) = {:2.3e}\n'.format(self.re_I))
            f.write('mc-error(I) = {:2.3e}\n'.format(np.sqrt(self.var_I / self.K)))
            f.write('-log(m_K(I)) = {:2.3e}\n\n'.format(-np.log(self.mean_I)))

        else:
            f.write('\nReweighted Quantity of interest\n')
            f.write('m_K(I^u) = {:2.3e}\n'.format(self.mean_I_u))
            f.write('s_K^2(I^u) = {:2.3e}\n'.format(self.var_I_u))
            f.write('re_K(I^u) = {:2.3e}\n'.format(self.re_I_u))
            f.write('mc-error(I^u) = {:2.3e}\n'.format(np.sqrt(self.var_I_u / self.K)))
            f.write('-log(m_K(I^u)) = {:2.3e}\n\n'.format(-np.log(self.mean_I_u)))

        h, m, s = get_time_in_hms(self.ct)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def get_grid_value_function(self):
        ''' evaluates the value function at the discretized domain.
        '''

        # flatten domain_h
        x = self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)

        # potential
        self.grid_potential = self.sde.potential(x).reshape(self.sde.Nx)

        # value function

        # not controlled, i.e zero
        if not self.is_controlled:
            self.grid_value_function = np.zeros(self.sde.Nx)

        # controlled with gaussian ansatz representation
        elif self.is_controlled and self.ansatz is not None:

            # set value function constant
            if self.sde.potential_name == 'nd_2well':
                self.ansatz.set_value_function_constant_corner()
            elif self.sde.potential_name == 'nd_2well_asym':
                self.ansatz.set_value_function_target_set()

            # evaluate value function
            self.grid_value_function = self.ansatz.value_function(x).reshape(self.sde.Nx)

        # controlled with nn representation
        elif self.is_controlled and self.nn_func_appr is not None:

            # 1d numerical integration
            if self.sde.d == 1:
                self.integrate_value_function_1d()
            else:
                raise "numerical integration is just supported by d=1"

        # bias potential
        self.grid_bias_potential = self.grid_value_function * self.sde.sigma**2

        # perturbed potential
        if hasattr(self, 'grid_bias_potential'):
            self.grid_perturbed_potential = self.grid_potential + self.grid_bias_potential

    def integrate_value_function_1d(self):
        ''' computes the value function on the grid (for 1d sde) by integrating numerically.
        '''
        assert self.sde.d == 1, ''
        assert self.sde.domain_h is not None, ''
        assert self.grid_control is not None, ''

        # grid
        x = self.sde.domain_h

        # control evaluated at the grid
        u = self.grid_control

        # initialize value function
        value_f = np.zeros(self.sde.Nx)

        # get indices where grid in the left / right of the target set
        target_set_lb, target_set_ub = self.sde.target_set[0]
        idx_l = np.where(x <= target_set_lb)[0]

        # compute value function in the left of the target set
        for k in np.flip(idx_l):
            value_f[k - 1] = value_f[k] + (1 / self.sde.sigma) * u[k] * self.sde.h

        self.grid_value_function = value_f

    def get_grid_value_function_i(self, i=0, x_j=0.):
        ''' computes the value of the value function and the bias potential along the i-th
            coordinate evaluated at x_j for all j != i.
        '''

        # inputs
        x = x_j * np.ones((self.Nh, self.sde.d))
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
            self.grid_perturbed_potential_i = self.grid_potential_i + self.grid_bias_potential_i

    def get_grid_control(self):
        # flattened domain_h
        x = self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)

        # gradient
        self.grid_gradient = self.sde.gradient(x).reshape(self.sde.domain_h.shape)

        # null control
        if not self.is_controlled:
            self.grid_control = np.zeros(self.sde.domain_h.shape)

        # gaussian ansatz control
        elif self.is_controlled and self.ansatz is not None:
            self.grid_control = self.ansatz.control(x).reshape(self.sde.domain_h.shape)

        # nn control
        elif self.is_controlled and self.nn_func_appr is not None:
            inputs = torch.tensor(x, dtype=torch.float)
            control_flattened = self.nn_func_appr.model(inputs).detach().numpy()
            self.grid_control = control_flattened.reshape(self.sde.domain_h.shape)

        # controlled drift
        self.grid_perturbed_drift = - self.grid_gradient + self.sde.sigma * self.grid_control

    def get_grid_control_i(self, i=0, x_j=0., k=None):
        ''' computes the value of the control along the i-th coordinate evaluated at x_j
            for all j != i. In case of working with the deterministic time horizont framework
            the control is evaluated at time k
        '''
        # check if time step k is given for the deterministic time horizont problem
        if self.problem_name == 'langevin_det-t':
            assert k is not None, ''

        # inputs
        x = x_j * np.ones((self.Nh, self.sde.d))
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

        y = np.empty((self.sde.d, traj_fhs - n_iter_avg + 1))
        for i in range(self.sde.d):
            y[i] = np.convolve(traj[i], np.ones(n_iter_avg) / n_iter_avg, mode='valid')

        labels = [r'$x_{:d}$'.format(i+1) for i in np.arange(self.sde.d)]

        fig.set_title(r'trajectory')
        fig.set_xlabel(r'$t$')
        fig.set_ylim(-1.5, 1.5)
        plt.subplots_adjust(left=0.14, right=0.96, bottom=0.12)
        fig.plot(x, y, labels=labels)
