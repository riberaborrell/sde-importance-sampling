import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from sde_importance_sampling.utils_path import make_dir_path, get_som_dir_path, get_time_in_hms
from sde_importance_sampling.utils_numeric import slice_1d_array
from sde_importance_sampling.utils_figures import COLORS_FIG, TITLES_FIG, LABELS_FIG


class StochasticOptimizationMethod(object):
    '''

    Attributes
    ----------

    sample: Sampling object
        Sampling object to estimate the loss and its gradient
    loss_type: str
        type of loss function
    optimizer: str
        type of optimization method
    lr: float
        (initial) learning rate and maximal number of iterations
    n_iterations_lim: int
        maximum number of iterations for the stochastic optimization method
    n_iterations: int
        number of iterations for used
    n_iterations_backup: int
        number of iterations between saving the control parameters
    m: int
        dimension of the parameter space for the control

    # array for each iterations
    thetas: array
        parameters of the control
    last_thetas: array
        parameters of the control of the last iteration
    save_thetas_all_it: bool
        flag is True if the parameters of the control are saved for all iterations
    losses: array
        array containing the loss values at each iterations
    vars_loss: array
        array containing the variance of the loss at each iteration
    eff_losses: array
        array containing the effective loss values at each iteration
    re_losses: array
        array containing the relative entropy loss
    grad_losses: array
        array containing the gradient of the loss function
    means_I_u: array
        array containing the estimation of the importance sampling quantity of interest
    vars_I_u: array
        array containing the variance of the importance sampling quantity of interest
    res_I_u: array
        array containing the relative error of the importance sampling quantity of interest
    u_l2_errors: array
        array containing the L^2-error between the control and the reference solution
    time_steps: array
        array containing the time steps used in the sampling
    cts: array
        array containing the computational time used at each iteration

    # computational time
    ct_initial: float
        initial computational time
    ct_time: float
        final computational time
    ct: float

    # directory paths
    dir_path: str
        absolute path of the directory for the soc problem
    iterations_dir_path: str
        absolute path of the directory for the iterations of the soc problem

    # hjb solver
    sol_hjb: HJB Solver object
        HJB solver object


    Methods
    -------
    __init__()

    set_dir_path()

    set_iterations_dir_path()

    start_timer()

    stop_timer()

    preallocate_arrays()

    update_arrays(i)

    compute_arrays_running_averages(n_iter_run_avg=None, n_iter_run_window=1)

    cut_array_given_threshold(attr_name='', epsilon=None)

    compute_cts_sum()

    compute_ct_arrays(Nx=1000, n_avg=10, ct_max=None)

    get_iteration_statistics(i)

    sgd_gaussian_ansatz()

    som_nn()

    som_nn_variance_gradient(N_grad)

    save()

    save_var_grad()

    load(dir_path=None, file_name='som.npz')

    load_var_grad()

    write_report()

    write_iteration_report(f)

    load_mc_sampling(dt_mc=0.01, N_mc=10**3, seed=None)

    load_hjb_solution_and_sampling(h_hjb=0.1, dt_hjb=0.01, N_hjb=10**3, seed=None)

    load_plot_labels_colors_and_linestyles()

    plot_loss()

    plot_var_loss()

    plot_mean_I_u()

    plot_var_I_u()

    plot_re_I_u()

    plot_error_bar_I_u()

    plot_time_steps()

    plot_cts()

    plot_u_l2_error()

    plot_u_l2_error_change()

    plot_re_I_u_cts()

    plot_control_i(it=-1, i=0, x_j=0.)

    plot_control_slices_ith_coordinate(it=-1, dir_path=None, file_name='control')

    plot_control_i_det(it=-1, i=0, t=None)

    get_control(i=None, h=0.1)

    plot_1d_iteration(i=None)

    plot_1d_iterations()

    plot_2d_iteration(i=None)

    '''
    def __init__(self, sample, loss_type=None, optimizer=None, lr=None, n_iterations_lim=None,
                 n_iterations_backup=None, save_thetas_all_it=True):
        ''' init methods

        Parameters
        ----------

        sample: Sampling object
            Sampling object to estimate the loss and its gradient
        loss_type: str
            type of loss function
        optimizer: str
            type of optimization method
        lr: float
            (initial) learning rate and maximal number of iterations
        n_iterations_lim: int
            maximum number of iterations for the stochastic optimization method
        n_iterations_backup: int
            number of iterations between saving the control parameters
        save_thetas_all_it: bool
            flag is True if the parameters of the control are saved for all iterations
        '''

        # sampling object to estimate the loss and its gradient
        self.sample = sample

        # type of loss function
        self.loss_type = loss_type

        # type of optimization method
        self.optimizer = optimizer

        # (initial) learning rate and maximal number of iterations 
        self.lr = lr
        self.n_iterations_lim = n_iterations_lim
        if n_iterations_backup is not None:
            self.n_iterations_backup = n_iterations_backup
        else:
            self.n_iterations_backup = n_iterations_lim

        self.save_thetas_all_it = save_thetas_all_it
        self.set_dir_path()

    def set_dir_path(self):
        ''' sets the directory absolute path for the soc problem
        '''
        if hasattr(self.sample, 'ansatz'):
            func_appr_dir_path = self.sample.ansatz.dir_path
        elif hasattr(self.sample, 'nn_func_appr'):
            func_appr_dir_path = self.sample.nn_func_appr.dir_path

        self.dir_path = get_som_dir_path(
            func_appr_dir_path,
            self.loss_type,
            self.optimizer,
            self.lr,
            self.sample.dt,
            self.sample.K,
            self.sample.seed,
        )

    def set_iterations_dir_path(self):
        '''
        '''
        self.iterations_dir_path = os.path.join(self.dir_path, 'SGD iterations')
        make_dir_path(self.iterations_dir_path)

    def start_timer(self):
        ''' start timer
        '''
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        ''' stop timer
        '''
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def preallocate_arrays(self):
        ''' preallocate arrays containing information of the soc approximation problem at each
            iteration.
        '''

        # parameters
        if self.save_thetas_all_it:
            self.thetas = np.empty((self.n_iterations_lim, self.m))

        # objective function and variance of objective function
        self.losses = np.empty(self.n_iterations_lim)
        self.vars_loss = np.empty(self.n_iterations_lim)

        # reweighted quantity of interest
        self.means_I_u = np.empty(self.n_iterations_lim)
        self.vars_I_u = np.empty(self.n_iterations_lim)
        self.res_I_u = np.empty(self.n_iterations_lim)

        # u l2 error
        if self.sample.do_u_l2_error:
            self.u_l2_errors = np.empty(self.n_iterations_lim)

        # time steps and computational time
        self.time_steps = np.empty(self.n_iterations_lim, dtype=int)
        self.cts = np.empty(self.n_iterations_lim)

        # gradient of the loss function
        if hasattr(self.sample, 'ansatz'):
            self.grad_losses = np.empty((self.n_iterations_lim, self.m))


    def update_arrays(self, i):
        ''' update arrays

        Parameters
        ----------
        i: int
            index of the iteration
        '''

        # update number of iterations used
        self.n_iterations = i + 1

        # add parameters for each iteration
        if self.save_thetas_all_it:
            if hasattr(self.sample, 'ansatz'):
                self.thetas[i, :] = self.sample.ansatz.theta
            elif hasattr(self.sample, 'nn_func_appr'):
                model = self.sample.nn_func_appr.model
                self.thetas[i, :] = model.get_parameters()

        # add parameters for the last iteration
        if not self.save_thetas_all_it and i == self.n_iterations_lim -1 :
            if hasattr(self.sample, 'ansatz'):
                self.last_thetas = self.sample.ansatz.theta
            elif hasattr(self.sample, 'nn_func_appr'):
                model = self.sample.nn_func_appr.model
                self.last_thetas = model.get_parameters()

        # add loss, variance of the loss and gradient
        self.losses[i] = self.sample.loss
        self.vars_loss[i] = self.sample.var_loss

        if hasattr(self.sample, 'ansatz'):
            self.grad_losses[i, :] = self.sample.grad_loss

        # add I_u statistics
        self.means_I_u[i] = self.sample.mean_I_u
        self.vars_I_u[i] = self.sample.var_I_u
        self.res_I_u[i] = self.sample.re_I_u

        # add l2 error 
        if self.sample.do_u_l2_error:
            self.u_l2_errors[i] = self.sample.u_l2_error

        # add time statistics
        if self.sample.sde.problem_name == 'langevin_stop-t':
            self.time_steps[i] = int(np.max(self.sample.fht) / self.sample.dt)
        elif self.sample.sde.problem_name == 'langevin_det-t':
            self.time_steps[i] = self.sample.k_lim
        self.cts[i] = self.sample.ct


    def compute_arrays_running_averages(self, n_iter_run_avg=None, n_iter_run_window=1):
        ''' Computes the running averages of all the sgd iterations arrays if they exist.
            Also cuts the running averaged array.

        Parameters
        ----------
        n_iter_run_avg: int
            number of iterations of the running averaged array
        n_iter_run_window: int
            number of iterations used to averaged. Length of the running average window.
        '''

        # number of iterations of the running window
        assert n_iter_run_window <= self.n_iterations, ''
        self.n_iter_run_window = n_iter_run_window

        # number of iterations to compute the running averages
        if n_iter_run_avg is not None:
            assert n_iter_run_avg >= n_iter_run_window, ''
            assert n_iter_run_avg <= self.n_iterations - n_iter_run_window + 1, ''
            self.n_iter_run_avg = n_iter_run_avg
        else:
            self.n_iter_run_avg = self.n_iterations - n_iter_run_window + 1

        # list of attributes where the running averages should be computed
        attr_names = [
            'losses',
            'vars_loss',
            'means_I_u',
            'vars_I_u',
            'res_I_u',
            'time_steps',
            'cts',
            'u_l2_errors',
        ]

        for attr_name in attr_names:

            # skip l2 error if it is not computed
            if attr_name == 'u_l2_errors' and not hasattr(self, 'u_l2_errors'):
                continue

            # get cutted array
            array = getattr(self, attr_name)[:self.n_iter_run_avg+n_iter_run_window-1]

            # compute running averages
            run_avg_array = np.convolve(
                array,
                np.ones(n_iter_run_window) / n_iter_run_window,
                mode='valid',
            )

            # save running average array
            setattr(self, 'run_avg_' + attr_name, run_avg_array)


    def cut_array_given_threshold(self, attr_name='', epsilon=None):
        ''' cut the array given by the attribute name up to the point where their running
            average is smaller than epsilon

        Parameters
        ----------
        attr_name: str
            name of the attribute containing the string that we want to cut.
        epsilon: float
            threshold value

        '''
        assert attr_name in ['vars_I_u', 'res_I_u', 'losses', 'u_l2_errors'], ''
        assert epsilon is not None, ''
        assert self.n_iter_avg is not None, ''
        attr_name_run_avg = 'run_avg_' + attr_name
        attr_name_eps_cut = attr_name + '_eps_cut'

        # get idx of the iterations which running average is smaller than epsilon
        idx = np.where(getattr(self, attr_name_run_avg) < epsilon)[0]

        # get the smaller index
        if idx.shape[0] != 0:
            idx_iter = idx[0]
        else:
            idx_iter = self.n_iterations - self.n_iter_avg + 1

        # cut array up to this indx
        attr_name_run_avg_cut = getattr(self, attr_name_run_avg)[:idx_iter]
        setattr(self, attr_name_eps_cut, attr_name_run_avg_cut)

        return idx_iter

    def compute_cts_sum(self):
        ''' Computes the accomulated computational time at each iteration
        '''
        self.cts_sum = np.array([np.sum(self.cts[:i]) for i in range(self.n_iterations)])

    def compute_ct_arrays(self, Nx=1000, n_avg=10, ct_max=None):
        ''' Computes ct arrays. The ct array is a linear discretization. The arrays are
            directly computed with running averages.

        Parameters
        ----------
        Nx: int
            number of points of the ct interval where we are going to interpolate
        n_avg: int
            number of iteration of the running windows
        '''

        # total ct
        if ct_max is None:
            ct_max = np.sum(self.cts[:self.n_iterations])

        # ct linear array
        self.ct_ct = np.linspace(0, ct_max, Nx)

        # ct accumulated at each iteration
        self.compute_cts_sum()

        # ct index
        self.ct_iter_index = np.argmin(
            np.abs(self.ct_ct.reshape(1, Nx) - self.cts_sum.reshape(self.n_iterations, 1)),
            axis=0,
        )

        # list of attributes where the ct array should be computed
        attr_names = [
            'losses',
            'vars_loss',
            'means_I_u',
            'vars_I_u',
            'res_I_u',
            'time_steps',
            'u_l2_errors',
        ]

        for attr_name in attr_names:

            # skip l2 error if it is not computed
            if attr_name == 'u_l2_errors' and self.u_l2_errors is None:
                continue

            # get array and compute ct array
            array = getattr(self, attr_name)
            ct_array = array[self.ct_iter_index]
            setattr(self, 'ct_' + attr_name, ct_array)

            # preallocate rung avg ct array
            run_avg_ct_array = np.empty(Nx)
            setattr(self, 'run_avg_ct_' + attr_name, run_avg_ct_array)

            # compute running averages
            idx = Nx - n_avg + 1
            run_avg_ct_array[:idx] = np.convolve(
                ct_array,
                np.ones(n_avg) / n_avg,
                mode='valid',
            )
            run_avg_ct_array[idx:] = np.nan


    def get_iteration_statistics(self, i):
        ''' get relevant information of the soc problem iteration

        Parameters
        ----------
        i: int
            index of the iteration
        '''
        msg = 'it.: {:d}, loss: {:2.3f}, mean I^u: {:2.3e}, re I^u: {:2.3f}' \
              ', time steps: {:2.1e}'.format(
                  i,
                  self.losses[i],
                  self.means_I_u[i],
                  self.res_I_u[i],
                  self.time_steps[i],
              )
        return msg

    def sgd_gaussian_ansatz(self):
        ''' stochastic gradient descent with gaussian ansatz parametrization
        '''
        self.start_timer()

        # number of parameters
        self.m = self.sample.ansatz.m

        # preallocate parameters and losses
        self.preallocate_arrays()

        for i in np.arange(self.n_iterations_lim):

            # compute loss and its gradient 
            succ = self.sample.sample_grad_loss_ansatz()

            # check if sample succeeded
            if not succ:
                break

            # save parameters, statistics and losses
            self.update_arrays(i)

            # print iteration info
            msg = self.get_iteration_statistics(i)
            print(msg)

            # back up save
            if (i + 1) % self.n_iterations_backup == 0:
                self.stop_timer()
                self.save()

            # update coefficients
            self.sample.ansatz.theta = self.thetas[i, :] - self.lr * self.grad_losses[i, :]

        self.stop_timer()
        self.save()

    def som_nn(self):
        ''' stochastic gradient based method with nn parametrization
        '''
        self.start_timer()

        # model and number of parameters
        model = self.sample.nn_func_appr.model
        self.m = model.d_flat

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define optimizer
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.lr,
            )
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
            )

        # preallocate parameters and losses
        self.preallocate_arrays()

        for i in np.arange(self.n_iterations_lim):

            # reset gradients
            optimizer.zero_grad()

            # compute ipa loss 
            if self.loss_type == 'ipa' and self.sample.sde.problem_name == 'langevin_stop-t':
                succ = self.sample.sample_loss_ipa_nn(device)
            elif self.loss_type == 'ipa' and self.sample.sde.problem_name == 'langevin_det-t':
                succ = self.sample.sample_loss_ipa_nn_det(device)

            # compute relative error loss
            elif self.loss_type == 're':
                succ = self.sample.sample_loss_re_nn(device)

            # check if sample succeeded
            if not succ:
                break

            # save information of the iteration
            self.update_arrays(i)

            # print iteration info
            msg = self.get_iteration_statistics(i)
            print(msg)

            # back up save
            if (i + 1) % self.n_iterations_backup == 0:
                self.stop_timer()
                self.save()

            # compute gradients
            if self.loss_type == 'ipa':
                self.sample.ipa_loss.backward()
            elif self.loss_type == 're':
                self.sample.re_loss.backward()

            # update parameters
            optimizer.step()

        self.stop_timer()
        self.save()

    def som_nn_variance_gradient(self, K_grad):
        ''' stochastic gradient based optimization with nn approximation and estimation of the
            variance of the gradient at each iteration

        Parameters
        ----------
        K_grad: int
            number of batches used to estimate the gradient
        '''
        assert self.loss_type == 'ipa', ''
        assert self.sample.problem_name == 'langevin_stop-t', ''

        self.start_timer()

        # save number of times the gradient is sampled
        self.K_grad = K_grad

        # model and number of parameters
        model = self.sample.nn_func_appr.model
        self.m = model.d_flat

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define optimizer
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.lr,
            )
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
            )

        # preallocate parameters and losses
        self.thetas_grad = np.empty((self.n_iterations_lim, self.K_grad, self.m))

        for i in np.arange(self.n_iterations_lim):
            for j in np.arange(self.K_grad):

                # reset gradients
                optimizer.zero_grad()

                # compute ipa loss 
                succ = self.sample.sample_loss_ipa_nn(device)

                # check if sample succeeded
                if not succ:
                    break

                # compute gradients
                self.sample.ipa_loss.backward(retain_graph=True)

                # compute the accomulated gradient on the model paramters
                self.thetas_grad[i, j, :] = model.get_gradient_parameters()

                # update parameters
                optimizer.step()

        self.stop_timer()
        self.save_var_grad()

    def save(self):
        ''' saves the relevant arrays as a npz files
        '''

        # set file path
        file_path = os.path.join(self.dir_path, 'som.npz')

        # create directories of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # create dictionary
        files_dict = {}

        # add sampling attributes
        files_dict['seed'] = self.sample.seed
        files_dict['K'] = self.sample.K

        # Euler-Marujama
        files_dict['dt'] = self.sample.dt

        # sgd
        # iterations
        files_dict['n_iterations'] = self.n_iterations

        # parameters
        if hasattr(self, 'thetas'):
            files_dict['thetas'] = self.thetas[:self.n_iterations]
        if hasattr(self, 'last_thetas'):
            files_dict['last_thetas'] = self.last_thetas

        # loss and its variance
        files_dict['losses'] = self.losses[:self.n_iterations]
        if hasattr(self, 'vars_loss'):
            files_dict['vars_loss'] = self.vars_loss[:self.n_iterations]

        if hasattr(self, 'eff_losses'):
            files_dict['eff_losses'] = self.eff_losses[:self.n_iterations]

        if hasattr(self, 're_losses'):
            files_dict['re_losses'] = self.re_losses[:self.n_iterations]

        # gradient of the loss
        if hasattr(self, 'grad_losses'):
            files_dict['grad_losses'] = self.grad_losses[:self.n_iterations, :]

        # quantity of interest and time steps
        files_dict['means_I_u'] = self.means_I_u[:self.n_iterations]
        files_dict['vars_I_u'] = self.vars_I_u[:self.n_iterations]
        files_dict['res_I_u'] = self.res_I_u[:self.n_iterations]
        files_dict['time_steps'] = self.time_steps[:self.n_iterations]

        # u l2 error
        if hasattr(self, 'u_l2_errors'):
            files_dict['u_l2_errors'] = self.u_l2_errors[:self.n_iterations]

        # computational time
        files_dict['cts'] = self.cts[:self.n_iterations]
        files_dict['ct'] = self.ct

        # save npz file
        np.savez(file_path, **files_dict)

    def save_var_grad(self):
        ''' saves the relevant arrays as a npz files when estimating the variance of the gradient
        '''

        # set file path
        file_path = os.path.join(self.dir_path, 'som_var_grad.npz')

        # create directories of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # create dictionary
        files_dict = {}

        # add sampling attributes
        files_dict['seed'] = self.sample.seed
        files_dict['K'] = self.sample.K

        # Euler-Marujama
        files_dict['dt'] = self.sample.dt

        # iterations
        files_dict['n_iterations'] = self.n_iterations

        # gradient samples
        files_dict['K_grad'] = self.K_grad

        files_dict['thetas_grad'] = self.thetas_grad

        # save npz file
        np.savez(file_path, **files_dict)


    def load(self, dir_path=None, file_name='som.npz'):
        '''
        '''

        # directory path
        if dir_path is None:
            dir_path = self.dir_path

        # load npz arrays
        try:
            data = np.load(
                  os.path.join(dir_path, file_name),
                  allow_pickle=True,
            )
            for file_name in data.files:
                setattr(self, file_name, data[file_name])
            return True

        except:
            print('no som found')
            return False

    def load_var_grad(self):
        '''
        '''
        try:
            data = np.load(
                  os.path.join(self.dir_path, 'som_var_grad.npz'),
                  allow_pickle=True,
            )
            for file_name in data.files:
                setattr(self, file_name, data[file_name])
            return True

        except:
            print('no som found')
            return False


    def write_report(self):
        ''' opens file and writes report on it. Also prints its content
        '''

        sample = self.sample

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, 'w')

        if hasattr(self.sample, 'ansatz'):
            pass
        elif hasattr(self.sample, 'nn_func_appr'):
            pass
            #sample.nn_func_appr.write_parameters(f)

        f.write('\nStochastic optimization method parameters\n')
        f.write('loss type: {}\n'.format(self.loss_type))
        f.write('som type: {}\n\n'.format(self.optimizer))

        f.write('lr: {}\n'.format(self.lr))
        f.write('# iterations lim: {}\n'.format(self.n_iterations_lim))

        f.write('# iterations used: {}\n'.format(self.n_iterations))
        f.write('total time steps: {:,d}\n'.format(int(self.time_steps.sum())))

        f.write('\nLast iteration\n')
        f.write('E[I_u]: {:2.3e}\n'.format(self.means_I_u[-1]))
        f.write('Var[I_u]: {:2.3e}\n'.format(self.vars_I_u[-1]))
        f.write('RE[I_u]: {:2.3f}\n'.format(self.res_I_u[-1]))
        f.write('value function: {:2.3f}\n'.format(- np.log(self.means_I_u[-1])))
        f.write('loss function: {:2.3f}\n'.format(self.losses[-1]))
        if hasattr(self, 'u_l2_errors'):
           f.write('u l2 error: {:2.3f}\n'.format(self.u_l2_errors[-1]))

        self.compute_arrays_running_averages(n_iter_run_window=10)
        f.write('\nRunning averages of last {:d} iterations\n'.format(self.n_iter_run_window))
        f.write('E[I_u]: {:2.3e}\n'.format(self.run_avg_means_I_u[-1]))
        f.write('Var[I_u]: {:2.3e}\n'.format(self.run_avg_vars_I_u[-1]))
        f.write('RE[I_u]: {:2.3f}\n'.format(self.run_avg_res_I_u[-1]))
        f.write('value function: {:2.3f}\n'.format(- np.log(self.run_avg_means_I_u[-1])))
        f.write('loss function: {:2.3f}\n\n'.format(self.run_avg_losses[-1]))
        if hasattr(self, 'u_l2_errors'):
            f.write('u l2 error: {:2.3f}\n'.format(self.run_avg_u_l2_errors[-1]))

        h, m, s = get_time_in_hms(self.ct)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        #self.write_iteration_report(f)
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def write_iteration_report(self, f):
        ''' writes in the given file the relevant information of the iteration

        Parameters
        ----------
        f: pyObject
            file where we write on
        '''
        for i in range(self.n_iterations):
            f.write('iteration = {:d}\n'.format(i))
            f.write('theta = {}\n'.format(self.thetas[i]))
            f.write('loss = {:2.4e}\n'.format(self.losses[i]))
            f.write('grad_loss = {}\n'.format(self.grad_losses[i]))
            f.write('|grad_loss|_2 = {:2.4e}\n\n'
                    ''.format(np.linalg.norm(self.grad_losses[i])))
            f.write('time steps = {}\n'.format(self.time_steps[i]))

    def load_mc_sampling(self, dt_mc=0.01, K_mc=10**3, seed=None):
        '''
        '''

        # save parameters
        self.dt_mc = dt_mc
        self.K_mc = K_mc

        # load mc sampling
        sample_mc = self.sample.sde.get_not_controlled_sampling(dt_mc, K_mc, seed)

        # length of the arrrays
        n_iter = self.n_iter_run_avg# - self.n_iter_run_window

        if sample_mc is not None:
            self.value_f_mc = np.full(n_iter, - np.log(sample_mc.mean_I))
            self.mean_I_mc = np.full(n_iter, sample_mc.mean_I)
            self.var_I_mc = np.full(n_iter, sample_mc.var_I)
            self.re_I_mc = np.full(n_iter, sample_mc.re_I)
            self.time_steps_mc = np.full(n_iter, sample_mc.k)
            self.ct_mc = np.full(n_iter, sample_mc.ct)
        else:
            self.value_f_mc = np.full(n_iter, np.nan)
            self.mean_I_mc = np.full(n_iter, np.nan)
            self.var_I_mc = np.full(n_iter, np.nan)
            self.re_I_mc = np.full(n_iter, np.nan)
            self.time_steps_mc = np.full(n_iter, np.nan)
            self.ct_mc = np.full(n_iter, np.nan)

    def load_hjb_solution_and_sampling(self, h_hjb=0.1, dt_hjb=0.01, K_hjb=10**3, seed=None):
        '''
        '''
        from sde_importance_sampling.importance_sampling import Sampling

        # save parameters
        self.h_hjb = h_hjb
        self.dt_hjb = dt_hjb
        self.K_hjb = K_hjb

        # load hjb solver
        if self.sample.sde.problem_name == 'langevin_stop-t':
            self.sol_hjb = self.sample.sde.get_hjb_solver(h_hjb)
        elif self.sample.sde.problem_name == 'langevin_det-t':
            self.sol_hjb = self.sample.sde.get_hjb_solver_det(h_hjb, dt_hjb)

        # break if there is no hjb solution
        if self.sol_hjb is None:
            return

        # length of the arrrays
        n_iter = self.n_iter_run_avg

        if self.sample.sde.problem_name == 'langevin_stop-t':
            hjb_psi_at_x = self.sol_hjb.get_psi_at_x(self.sample.xzero)
            hjb_value_f_at_x = self.sol_hjb.get_value_function_at_x(self.sample.xzero)
            self.psi_hjb = np.full(n_iter, hjb_psi_at_x)
            self.value_f_hjb = np.full(n_iter, hjb_value_f_at_x)

        elif self.sample.sde.problem_name == 'langevin_det-t':
            hjb_psi_at_x = self.sol_hjb.get_psi_t_x(0., self.sample.xzero)
            hjb_value_f_at_x = self.sol_hjb.get_value_funtion_t_x(0., self.sample.xzero)
            self.psi_hjb = np.full(n_iter, hjb_psi_at_x)
            self.value_f_hjb = np.full(n_iter, hjb_value_f_at_x)

        # load hjb sampling
        sample_hjb = self.sample.sde.get_hjb_sampling(self.sol_hjb.dir_path, dt_hjb, K_hjb, seed)

        # break if there is no hjb sampling
        if sample_hjb is None:
            return

        self.value_f_is_hjb = np.full(n_iter, -np.log(sample_hjb.mean_I_u))
        self.mean_I_u_hjb = np.full(n_iter, sample_hjb.mean_I_u)
        self.var_I_u_hjb = np.full(n_iter, sample_hjb.var_I_u)
        self.re_I_u_hjb = np.full(n_iter, sample_hjb.re_I_u)
        self.time_steps_is_hjb = np.full(n_iter, sample_hjb.k)
        self.ct_is_hjb = np.full(n_iter, sample_hjb.ct)

    def load_plot_labels_colors_and_linestyles(self):
        '''
        '''

        if self.sol_hjb is None:
            self.colors = [
                COLORS_FIG['nn'],
                COLORS_FIG['mc-sampling'],
            ]
            self.linestyles = ['-', '-']
            self.labels = ['SOC', 'MC sampling']

        else :
            self.colors = [
                COLORS_FIG['nn'],
                COLORS_FIG['mc-sampling'],
                COLORS_FIG['optimal-is'],
                COLORS_FIG['hjb-solution'],
            ]
            self.linestyles = ['-', '-', '-', ':']
            self.labels = [
                'SOC',
                'MC sampling',
                'Optimal',
                'Reference solution',
            ]


    def plot_loss(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_losses'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # loss, mc sampling, hjb sampling and hjb solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_losses,
                self.value_f_mc,
            ))
        else:
            y = np.vstack((
                self.run_avg_losses,
                self.value_f_mc,
                self.value_f_is_hjb,
                self.value_f_hjb,
            ))

        # loss figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='loss',
        )
        fig.set_title(TITLES_FIG['loss'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels, self.colors, self.linestyles)

    def plot_var_loss(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_vars_loss'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # var loss
        y = self.run_avg_vars_loss
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='var-loss',
        )
        fig.set_title(TITLES_FIG['var-loss'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels[0], self.colors[0], self.linestyles[0])


    def plot_mean_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_means_I_u'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # mean I^u, mc sampling, hjb sampling and hjb solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_means_I_u,
                self.mean_I_mc,
            ))
        else:
            y = np.vstack((
                self.run_avg_means_I_u,
                self.mean_I_mc,
                self.mean_I_u_hjb,
                self.psi_hjb,
            ))

        # mean I^u figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='mean',
        )
        fig.set_title(TITLES_FIG['psi'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels, self.colors, self.linestyles)

    def plot_var_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_vars_I_u'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # re I^u, mc sampling, hjb sampling and hjb solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_vars_I_u,
                self.var_I_mc,
            ))
        else:
            y = np.vstack((
                self.run_avg_vars_I_u,
                self.var_I_mc,
                self.var_I_u_hjb,
            ))

        # relative error figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='var',
        )
        fig.set_title(TITLES_FIG['var-i-u'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        if self.sol_hjb is None:
            fig.plot(x, y, self.labels, self.colors, self.linestyles)
        else:
            fig.plot(x, y, self.labels[:3], self.colors[:3], self.linestyles[:3])


    def plot_re_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_res_I_u'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # without reference solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_res_I_u,
                self.re_I_mc,
            ))
            labels = self.labels[:2]
            colors = self.colors[:2]
            linestyles = self.linestyles[:2]

        # with reference solution
        else:
            y = np.vstack((
                self.run_avg_res_I_u,
                self.re_I_mc,
                self.re_I_u_hjb,
            ))
            labels = self.labels
            colors = self.colors
            linestyles = self.linestyles

        # figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='re',
        )
        fig.set_title(TITLES_FIG['re-i-u'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, labels, colors, linestyles)

    def plot_error_bar_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iter_run_avg)

        y = self.means_I_u
        y_err = np.sqrt(self.vars_I_u / self.sample.N)

        # plot mean I u with error bars
        fig, ax = plt.subplots()
        ax.set_yscale("log", nonpositive='clip')
        plt.plot(x, y, color='b', linestyle='-')
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.5)
        plt.show()
        #plt.savefig(plt.file_path)
        #plt.close()

    def plot_time_steps(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_time_steps'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # loss, mc sampling, hjb sampling and hjb solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_time_steps,
                self.time_steps_mc,
            ))
        else:
            y = np.vstack((
                self.run_avg_time_steps,
                self.time_steps_mc,
                self.time_steps_is_hjb,
            ))

        # loss figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='time-steps',
        )
        fig.set_title(TITLES_FIG['time-steps'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels[:3], self.colors[:3], self.linestyles[:3])


    def plot_cts(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_cts'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # computational time, mc sampling, hjb sampling and hjb solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_cts,
                self.ct_mc,
            ))
        else:
            y = np.vstack((
                self.run_avg_cts,
                self.ct_mc,
                self.ct_is_hjb,
            ))

        # cts figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='cts',
        )
        fig.set_title(TITLES_FIG['ct'])
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')

        if not hasattr(self, 'sol_hjb'):
            fig.plot(x, y, self.labels, self.colors, self.linestyles)
        else:
            fig.plot(x, y, self.labels[:3], self.colors[:3], self.linestyles[:3])

    def plot_u_l2_error(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if u L^2 error is computed
        if not hasattr(self, 'u_l2_errors'):
            return

        # check if running averages are computed
        if not hasattr(self, 'run_avg_u_l2_errors'):
            self.compute_arrays_running_averages(n_iter_run_window=1)

        # iterations
        x = np.arange(self.n_iter_run_avg)

        # u l2 error
        y = self.run_avg_u_l2_errors

        # u l2 error figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='u-l2-error',
        )
        fig.set_xlabel(LABELS_FIG['grad-steps'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels[0], self.colors[0], self.linestyles[0])

    def plot_u_l2_error_change(self):
        '''
        '''
        from figures.myfigure import MyFigure

        x = np.arange(1, self.n_iterations)
        y = self.u_l2_errors[:-1] - self.u_l2_errors[1:]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='u-l2-error-change',
        )
        fig.set_xlabel('SGD iterations')
        fig.plot(x, y, self.labels[0], self.colors[0], self.linestyles[0])


    def plot_re_I_u_cts(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_ct_res_I_u'):
            self.compute_ct_arrays(n_avg=10)

        # computational time
        x = self.ct_ct

        # without reference solution
        if not hasattr(self, 'sol_hjb'):
            y = np.vstack((
                self.run_avg_ct_res_I_u,
                self.re_I_mc,
            ))
            labels = self.labels[:2]
            colors = self.colors[:2]
            linestyles = self.linestyles[:2]

        # with reference solution
        else:
            y = np.vstack((
                self.run_avg_res_ct_I_u,
                self.re_I_mc,
                self.re_I_u_hjb,
            ))
            labels = self.labels
            colors = self.colors
            linestyles = self.linestyles

        # figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='ct-re',
        )
        fig.set_title(TITLES_FIG['re-i-u'])
        fig.set_xlabel(LABELS_FIG['ct'])
        fig.set_plot_scale('semilogy')
        fig.plot(x, y)

    def plot_control_i(self, it=-1, i=0, x_j=0.):
        '''
        '''
        from figures.myfigure import MyFigure

        # load given iteration
        self.sample.nn_func_appr.model.load_parameters(self.thetas[it])

        # get control
        self.sample.discretize_domain_ith_coordinate(h=0.01)
        self.sample.get_grid_control_i(i=i, x_j=x_j)

        x = self.sample.domain_i_h
        y = self.sample.grid_control_i

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='control-{:d}'.format(i+1),
        )
        fig.set_title(r'$u(x; \theta)$')
        fig.set_xlabel(r'$x_{:d}$'.format(i+1))
        fig.set_xlim(-2, 2)
        fig.plot(x, y)

    def plot_control_slices_ith_coordinate(self, it=-1, dir_path=None, file_name='control'):
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

        # load given iteration
        self.sample.nn_func_appr.model.load_parameters(self.thetas[it])

        # discretize ith coordinate
        self.sample.discretize_domain_ith_coordinate(h=0.01)

        # get control
        self.sample.get_grid_control_i(i=0, x_j=-1)
        control_1 = self.sample.grid_control_i
        self.sample.get_grid_control_i(i=1, x_j=-1)
        control_2 = self.sample.grid_control_i
        self.sample.get_grid_control_i(i=0, x_j=0)
        control_3 = self.sample.grid_control_i
        self.sample.get_grid_control_i(i=1, x_j=0)
        control_4 = self.sample.grid_control_i
        self.sample.get_grid_control_i(i=0, x_j=1)
        control_5 = self.sample.grid_control_i
        self.sample.get_grid_control_i(i=1, x_j=1)
        control_6 = self.sample.grid_control_i

        labels = [
                r'$x_i = x_1, \, x_j = {:.1f}$'.format(-1),
                r'$x_i = x_2, \, x_j = {:.1f}$'.format(-1),
                r'$x_i = x_1, \, x_j = {:.1f}$'.format(0),
                r'$x_i = x_2, \, x_j = {:.1f}$'.format(0),
                r'$x_i = x_1, \, x_j = {:.1f}$'.format(1),
                r'$x_i = x_2, \, x_j = {:.1f}$'.format(1),
        ]

        x = self.sample.domain_i_h
        y = np.vstack((
                control_1,
                control_2,
                control_3,
                control_4,
                control_5,
                control_6,
        ))

        #fig.set_title(r'$u_i(x; \theta_{2999})$, $\theta_0$ = meta')
        fig.set_title(r'Control $u_{i}(\theta)$ (initial)')
        fig.set_xlabel(r'$x_i$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        #fig.turn_legend_off()
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12)
        fig.plot(x, y, labels)


    def plot_control_i_det(self, it=-1, i=0, t=None):
        '''
        '''
        from figures.myfigure import MyFigure

        assert t is not None, ''



        #TODO! adapat for det-t. check for thetas of last_thetas
        # load given iteration
        self.sample.nn_func_appr.model.load_parameters(self.thetas[it])
        #self.sample.nn_func_appr.model.load_parameters(self.last_thetas)

        # get time index
        k = self.sample.get_time_index(t)

        # get control
        self.sample.discretize_domain_ith_coordinate(h=0.01)
        self.sample.get_grid_control_i(i=i, k=k)

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver_det(h=0.01, dt=0.005)
        u_opt_i = sol_hjb.u_opt_i[k, :]

        x = self.sample.domain_i_h
        y = np.vstack((
            self.sample.grid_control_i,
            u_opt_i,
        ))

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='control-i',
        )
        fig.set_xlabel(r'$x_{:d}$'.format(i))
        fig.plot(x, y)


    def get_control(self, i=None, h=0.1):
        '''
        '''

        # last iteration if i is not given
        if i is None:
            i = self.n_iterations - 1
        assert i < self.n_iterations, ''

        # set theta
        if hasattr(self.sample, 'ansatz'):
            self.sample.ansatz.theta = self.thetas[i]
        elif hasattr(self.sample, 'nn_func_appr'):
            self.sample.nn_func_appr.model.load_parameters(self.thetas[i])

        # discretize domain and evaluate control in grid
        self.sample.discretize_domain(h=h)
        self.sample.get_grid_control()

        # evaluate value function in grid
        if self.sample.n == 1:
            self.sample.integrate_value_function_1d()

        return np.copy(self.sample.grid_control), np.copy(self.sample.grid_controlled_potential)

    def plot_1d_iteration(self, i=None):
        from figures.myfigure import MyFigure

        # last iteration if i is not given
        if i is None:
            i = self.n_iterations - 1

        assert i < self.n_iterations, ''

        self.set_iterations_dir_path()
        ext = '_iter{}'.format(i)

        # set theta
        if hasattr(self.sample, 'ansatz'):
            self.sample.ansatz.theta = self.thetas[i]
        elif hasattr(self.sample, 'nn_func_appr'):
            self.sample.nn_func_appr.model.load_parameters(self.thetas[i])

        # discretize domain and evaluate in grid
        self.sample.sde.discretize_domain(h=0.001)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        # get hjb solution
        sol_hjb = self.sample.sde.get_hjb_solver(h=0.001)
        sol_hjb.get_perturbed_potential_and_drift()

        # colors and labels
        labels = [r'SOC (iteration: {})'.format(i), 'HJB solution']
        colors = ['tab:blue', 'tab:cyan']

        # domain
        x = self.sample.sde.domain_h[:, 0]

        if hasattr(self.sample, 'grid_value_function'):

            # plot value function
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='value-function' + ext,
            )
            y = np.vstack((
                self.sample.grid_value_function,
                sol_hjb.value_function,
            ))
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, y, labels=labels, colors=colors)

            # plot controlled potential
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='perturbed-potential' + ext,
            )
            y = np.vstack((
                self.sample.grid_perturbed_potential,
                sol_hjb.perturbed_potential,
            ))
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, y, labels=labels, colors=colors)

        # plot control
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.iterations_dir_path,
            file_name='control' + ext,
        )
        y = np.vstack((
            self.sample.grid_control[:, 0],
            sol_hjb.u_opt[:, 0],
        ))
        fig.set_xlabel = 'x'
        fig.set_xlim(-2, 2)
        fig.plot(x, y, labels=labels, colors=colors)


    def plot_1d_iterations(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # discretize domain and evaluate in grid
        self.sample.sde.discretize_domain(h=0.001)
        x = self.sample.sde.domain_h[:, 0]

        # filter iterations to show
        iterations = np.arange(self.n_iterations)
        sliced_iterations = slice_1d_array(iterations, n_elements=5)
        n_sliced_iterations = sliced_iterations.shape[0]

        # preallocate arrays
        perturbed_potentials = np.zeros((n_sliced_iterations + 1, x.shape[0]))
        value_fs = np.zeros((n_sliced_iterations + 1, x.shape[0]))
        controls = np.zeros((n_sliced_iterations + 1, x.shape[0]))

        # preallocate labels and colors
        labels = []
        colors = []

        for idx, i in enumerate(sliced_iterations):
            labels.append(r'SOC (iteration = {:d})'.format(i))
            colors.append(None)

            # set theta
            if self.sample.ansatz is not None:
                self.sample.ansatz.theta = self.thetas[i]
            elif self.sample.nn_func_appr is not None:
                self.sample.nn_func_appr.model.load_parameters(self.thetas[i])

            self.sample.get_grid_value_function()
            self.sample.get_grid_control()

            # update functions
            if self.sample.grid_value_function is not None:
                perturbed_potentials[idx, :] = self.sample.grid_perturbed_potential
                value_fs[idx, :] = self.sample.grid_value_function
            controls[idx, :] = self.sample.grid_control[:, 0]

        # get hjb solution
        sol_hjb = self.sample.sde.get_hjb_solver(h=0.001)
        sol_hjb.get_perturbed_potential_and_drift()
        labels.append('HJB solution')
        colors.append('tab:cyan')

        if self.sample.grid_value_function is not None:

            # plot value function
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.dir_path,
                file_name='value-function',
            )
            value_fs[-1, :] = sol_hjb.value_function
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, value_fs, labels=labels, colors=colors)

            # plot controlled potential
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.dir_path,
                file_name='perturbed-potential',
            )
            perturbed_potentials[-1, :] = sol_hjb.perturbed_potential
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, perturbed_potentials, labels=labels, colors=colors)


        # plot control
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='control',
        )
        controls[-1, :] = sol_hjb.u_opt[:, 0]
        fig.set_xlabel = 'x'
        fig.set_xlim(-2, 2)
        fig.plot(x, controls, labels=labels, colors=colors)

    def plot_2d_iteration(self, i=None):
        from figures.myfigure import MyFigure

        # last iteration is i is not given
        if i is None:
            i = self.n_iterations - 1

        assert i < self.n_iterations, ''

        self.set_iterations_dir_path()
        ext = '_iter{}'.format(i)

        # set theta
        if self.sample.ansatz is not None:
            self.sample.ansatz.theta = self.thetas[i]
        elif self.sample.nn_func_appr is not None:
            self.sample.nn_func_appr.model.load_parameters(self.thetas[i])

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.005)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.005)
        sol_hjb.get_controlled_potential_and_drift()

        # domain
        X = self.sample.domain_h[:, :, 0]
        Y = self.sample.domain_h[:, :, 1]

        if self.sample.grid_value_function is not None:

            # plot value function
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='value-function' + ext,
            )
            fig.set_xlim(-2, 2)
            fig.set_ylim(-2, 2)
            fig.set_xlabel(r'$x_1$')
            fig.set_ylabel(r'$x_2$')
            fig.set_contour_levels_scale('log')
            fig.contour(X, Y, self.sample.grid_value_function)

            # plot controlled potential
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='controlled-potential' + ext,
            )
            fig.set_xlim(-2, 2)
            fig.set_ylim(-2, 2)
            fig.set_xlabel(r'$x_1$')
            fig.set_ylabel(r'$x_2$')
            fig.set_contour_levels_scale('log')
            fig.contour(X, Y, self.sample.grid_controlled_potential)

        # plot control
        U = self.sample.grid_control[:, :, 0]
        V = self.sample.grid_control[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.iterations_dir_path,
            file_name='control' + ext,
        )
        fig.set_title('control')
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.vector_field(X, Y, U, V, scale=10)

        # plot controlled drift
        U = self.sample.grid_controlled_drift[:, :, 0]
        V = self.sample.grid_controlled_drift[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.iterations_dir_path,
            file_name='controlled-drift' + ext,
        )
        fig.set_title('controlled drift')
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.vector_field(X, Y, U, V, scale=100)

        # plot difference between control and hjb solution
        U = np.abs(self.sample.grid_control[:, :, 0] - sol_hjb.u_opt[:, :, 0])
        V = np.abs(self.sample.grid_control[:, :, 1] - sol_hjb.u_opt[:, :, 1])
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.iterations_dir_path,
            file_name='control-minus-hjb' + ext,
        )
        fig.set_title('control vs hjb')
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.set_colormap('PuRd')
        fig.vector_field(X, Y, U, V, scale=10)

        # plot difference between controlled drift and hjb solution
        U = np.abs(self.sample.grid_controlled_drift[:, :, 0] - sol_hjb.controlled_drift[:, :, 0])
        V = np.abs(self.sample.grid_controlled_drift[:, :, 1] - sol_hjb.controlled_drift[:, :, 1])
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.iterations_dir_path,
            file_name='controlled-drift-minus-hjb' + ext,
        )
        fig.set_title('controlled drift vs hjb')
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.set_colormap('PuRd')
        fig.vector_field(X, Y, U, V, scale=10)
