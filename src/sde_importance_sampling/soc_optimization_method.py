from sde_importance_sampling.utils_path import make_dir_path, get_som_dir_path, get_time_in_hms
from sde_importance_sampling.utils_numeric import slice_1d_array

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import time
import os

class StochasticOptimizationMethod:
    '''
    '''
    def __init__(self, sample, loss_type=None, optimizer=None, lr=None, n_iterations_lim=None,
                 save_thetas_all_it=True):
        '''
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

        # per iteration
        self.m = None
        self.n_iterations = None
        self.thetas = None
        self.last_thetas = None
        self.save_thetas_all_it = save_thetas_all_it
        self.losses = None
        self.vars_loss = None
        self.ipa_losses = None
        self.re_losses = None
        self.grad_losses = None
        self.means_I_u = None
        self.vars_I_u = None
        self.res_I_u = None
        self.u_l2_errors = None
        self.time_steps = None
        self.cts = None

        # computational time
        self.ct_initial = None
        self.ct_final = None
        self.ct = None

        # set path
        self.dir_path = None
        self.set_dir_path()
        self.iterations_dir_path = None

        # hjb solver
        self.sol_hjb = None

    def set_dir_path(self):
        if self.sample.ansatz is not None:
            func_appr_dir_path = self.sample.ansatz.dir_path
        elif self.sample.nn_func_appr is not None:
            func_appr_dir_path = self.sample.nn_func_appr.dir_path

        self.dir_path = get_som_dir_path(
            func_appr_dir_path,
            self.loss_type,
            self.optimizer,
            self.lr,
            self.sample.dt,
            self.sample.N,
            self.sample.seed,
        )

    def set_iterations_dir_path(self):
        self.iterations_dir_path = os.path.join(self.dir_path, 'SGD iterations')
        make_dir_path(self.iterations_dir_path)

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def preallocate_arrays(self):

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

        if self.sample.ansatz is not None:
            self.grad_losses = np.empty((self.n_iterations_lim, self.m))


    def update_arrays(self, i):

        # update number of iterations used
        self.n_iterations = i + 1

        # add parameters for each iteration
        if self.save_thetas_all_it:
            if self.sample.ansatz is not None:
                self.thetas[i, :] = self.sample.ansatz.theta
            elif self.sample.nn_func_appr is not None:
                model = self.sample.nn_func_appr.model
                self.thetas[i, :] = model.get_parameters()

        # add parameters for the last iteration
        if not self.save_thetas_all_it and i == self.n_iterations_lim -1 :
            if self.sample.ansatz is not None:
                self.last_thetas = self.sample.ansatz.theta
            elif self.sample.nn_func_appr is not None:
                model = self.sample.nn_func_appr.model
                self.last_thetas = model.get_parameters()

        # add loss, variance of the loss and gradient
        self.losses[i] = self.sample.loss
        self.vars_loss[i] = self.sample.var_loss

        if self.sample.ansatz is not None:
            self.grad_losses[i, :] = self.sample.grad_loss

        # add I_u statistics
        self.means_I_u[i] = self.sample.mean_I_u
        self.vars_I_u[i] = self.sample.var_I_u
        self.res_I_u[i] = self.sample.re_I_u

        # add l2 error 
        if self.sample.do_u_l2_error:
            self.u_l2_errors[i] = self.sample.u_l2_error

        # add time statistics
        if self.sample.problem_name == 'langevin_stop-t':
            self.time_steps[i] = int(np.max(self.sample.fht) / self.sample.dt)
        elif self.sample.problem_name == 'langevin_det-t':
            self.time_steps[i] = self.sample.k_lim
        self.cts[i] = self.sample.ct

    def cut_arrays(self, n_iter=None):
        '''
        '''
        if n_iter is not None:
            self.n_iterations = n_iter
        assert self.n_iterations < self.n_iterations_lim, ''

        # parameters
        self.thetas = self.thetas[:self.n_iterations]

        # losses
        self.losses = self.losses[:self.n_iterations]
        if self.vars_loss is not None:
            self.vars_loss = self.vars_loss[:self.n_iterations]
        if self.sample.ansatz is not None:
            self.grad_losses = self.grad_losses[:self.n_iterations, :]

        if self.ipa_losses is not None:
            self.ipa_losses = self.ipa_losses[:self.n_iterations]

        elif self.re_losses is not None:
            self.re_losses = self.re_losses[:self.n_iterations]

        # quantity of interest and time steps
        self.means_I_u = self.means_I_u[:self.n_iterations]
        self.vars_I_u = self.vars_I_u[:self.n_iterations]
        self.res_I_u = self.res_I_u[:self.n_iterations]

        # u l2 error
        if self.u_l2_errors is not None:
            self.u_l2_errors = self.u_l2_errors[:self.n_iterations]

        # computational time
        self.time_steps = self.time_steps[:self.n_iterations]
        self.cts = self.cts[:self.n_iterations]

    def get_iteration_statistics(self, i):
        msg = 'it.: {:d}, loss: {:2.3f}, mean I^u: {:2.3e}, re I^u: {:2.3f}' \
              ', time steps: {:2.1e}'.format(
                  i,
                  self.losses[i],
                  self.means_I_u[i],
                  self.res_I_u[i],
                  self.time_steps[i],
              )
        return msg

    def sgd_ipa_gaussian_ansatz(self):
        self.start_timer()

        # number of parameters
        self.m = self.sample.ansatz.m

        # preallocate parameters and losses
        self.preallocate_arrays()

        for i in np.arange(self.n_iterations_lim):

            # compute loss and its gradient 
            succ = self.sample.sample_loss_ipa_ansatz()

            # check if sample succeeded
            if not succ:
                break

            # save parameters, statistics and losses
            self.update_arrays(i)

            # print iteration info
            msg = self.get_iteration_statistics(i)
            print(msg)

            # back up save
            if i % 100 == 0:
                self.save()

            # update coefficients
            self.sample.ansatz.theta = self.thetas[i, :] - self.lr * self.grad_losses[i, :]

        self.stop_timer()
        self.save()

    def som_nn(self):
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
            if self.loss_type == 'ipa' and self.sample.problem_name == 'langevin_stop-t':
                succ = self.sample.sample_loss_ipa_nn(device)
            elif self.loss_type == 'ipa' and self.sample.problem_name == 'langevin_det-t':
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
            if i % 100 == 0:
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

    def save(self):

        # set file path
        file_path = os.path.join(self.dir_path, 'som.npz')

        # create directories of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # create dictionary
        files_dict = {}

        # add sampling attributes
        files_dict['seed'] = self.sample.seed
        files_dict['N'] = self.sample.N

        # Euler-Marujama
        files_dict['dt'] = self.sample.dt

        # sgd
        # iterations
        files_dict['n_iterations'] = self.n_iterations

        # parameters
        if self.thetas is not None:
            files_dict['thetas'] = self.thetas
        if self.last_thetas is not None:
            files_dict['last_thetas'] = self.last_thetas

        # loss and its variance
        files_dict['losses'] = self.losses
        if self.vars_loss is not None:
            files_dict['vars_loss'] = self.vars_loss

        if self.ipa_losses is not None:
            files_dict['ipa_losses'] = self.ipa_losses

        if self.re_losses is not None:
            files_dict['re_losses'] = self.re_losses

        # gradient of the loss
        if self.grad_losses is not None:
            files_dict['grad_losses'] = self.grad_losses

        # quantity of interest and time steps
        files_dict['means_I_u'] = self.means_I_u
        files_dict['vars_I_u'] = self.vars_I_u
        files_dict['res_I_u'] = self.res_I_u
        files_dict['time_steps'] = self.time_steps

        # u l2 error
        if self.u_l2_errors is not None:
            files_dict['u_l2_errors'] = self.u_l2_errors

        # computational time
        files_dict['cts'] = self.cts
        files_dict['ct'] = self.ct

        # save npz file
        np.savez(file_path, **files_dict)

    def load(self):
        try:
            data = np.load(
                  os.path.join(self.dir_path, 'som.npz'),
                  allow_pickle=True,
            )
            for file_name in data.files:
                setattr(self, file_name, data[file_name])
            return True

        except:
            print('no som found')
            return False

    def compute_running_averages(self, n_iter_avg=1):
        '''
        '''
        # number of iterations of the running window
        self.n_iter_avg = n_iter_avg

        # loss
        self.run_avg_losses = np.convolve(
            self.losses, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )

        # variance of loss
        if self.vars_loss is not None:
            self.run_avg_vars_loss = np.convolve(
                self.vars_loss, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
            )

        # mean I^u
        self.run_avg_means_I_u = np.convolve(
            self.means_I_u, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )

        # var I^u
        self.run_avg_vars_I_u = np.convolve(
            self.vars_I_u, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )

        # re I^u
        self.run_avg_res_I_u = np.convolve(
            self.res_I_u, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )

        # time steps
        self.run_avg_time_steps = np.convolve(
            self.time_steps, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )

        # computational time
        self.run_avg_cts = np.convolve(
            self.cts, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )

        # u l2 error
        if self.u_l2_errors is not None:
            self.run_avg_u_l2_errors = np.convolve(
                self.u_l2_errors, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
            )

    def cut_vars_I_u(self, epsilon=None):
        ''' cut the array which keep the variance I^u up to the point where their running
            average is smaller than epsilon
        '''
        assert epsilon is not None, ''
        assert self.n_iter_avg is not None, ''

        # get idx of the iterations which running average is smaller than epsilon
        idx = np.where(self.run_avg_vars_I_u < epsilon)[0]

        # get the smaller index
        if idx.shape[0] != 0:
            idx_iter = idx[0]
        else:
            idx_iter = self.n_iterations

        # cut array up to this indx
        self.vars_I_u_epsilon_cut = self.run_avg_vars_I_u[:idx_iter]

    def cut_res_I_u(self, epsilon=None):
        ''' cut the array which keep the relative error I^u up to the point where their running
            average is smaller than epsilon
        '''
        assert epsilon is not None, ''
        assert self.n_iter_avg is not None, ''

        # get idx of the iterations which running average is smaller than epsilon
        idx = np.where(self.run_avg_res_I_u < epsilon)[0]

        # get the smaller index
        if idx.shape[0] != 0:
            idx_iter = idx[0]
        else:
            idx_iter = self.n_iterations

        # cut array up to this indx
        self.res_I_u_epsilon_cut = self.run_avg_res_I_u[:idx_iter]

    def cut_losses(self, epsilon=None):
        ''' cut the array which keep the losses up to the point where their running
            average is smaller than epsilon
        '''
        assert epsilon is not None, ''
        assert self.n_iter_avg is not None, ''

        # get idx of the iterations which running average is smaller than epsilon
        idx = np.where(self.run_avg_losses < epsilon)[0]

        # get the smaller index
        if idx.shape[0] != 0:
            idx_iter = idx[0]
        else:
            idx_iter = self.n_iterations

        # cut array up to this indx
        self.losses_epsilon_cut = self.run_avg_losses[:idx_iter]



    def write_report(self):
        sample = self.sample

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, 'w')

        if self.sample.ansatz is not None:
            pass
        elif self.sample.nn_func_appr is not None:
            sample.nn_func_appr.write_parameters(f)

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
        if self.u_l2_errors is not None:
           f.write('u l2 error: {:2.3f}\n'.format(self.u_l2_errors[-1]))

        self.compute_running_averages(n_iter_avg=10)
        f.write('\nRunning averages of last {:d} iterations\n'.format(self.n_iter_avg))
        f.write('E[I_u]: {:2.3e}\n'.format(self.run_avg_means_I_u[-1]))
        f.write('Var[I_u]: {:2.3e}\n'.format(self.run_avg_vars_I_u[-1]))
        f.write('RE[I_u]: {:2.3f}\n'.format(self.run_avg_res_I_u[-1]))
        f.write('value function: {:2.3f}\n'.format(- np.log(self.run_avg_means_I_u[-1])))
        f.write('loss function: {:2.3f}\n\n'.format(self.run_avg_losses[-1]))
        if self.u_l2_errors is not None:
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
        for i in range(self.n_iterations):
            f.write('iteration = {:d}\n'.format(i))
            f.write('theta = {}\n'.format(self.thetas[i]))
            f.write('loss = {:2.4e}\n'.format(self.losses[i]))
            f.write('grad_loss = {}\n'.format(self.grad_losses[i]))
            f.write('|grad_loss|_2 = {:2.4e}\n\n'
                    ''.format(np.linalg.norm(self.grad_losses[i])))
            f.write('time steps = {}\n'.format(self.time_steps[i]))

    def load_mc_sampling(self, dt_mc=0.01, N_mc=10**3, seed=None):
        '''
        '''

        # save parameters
        self.dt_mc = dt_mc
        self.N_mc = N_mc

        # load mc sampling
        sample_mc = self.sample.get_not_controlled_sampling(dt_mc, N_mc, seed)

        if sample_mc is not None:
            self.value_f_mc = np.full(self.n_iterations, - np.log(sample_mc.mean_I))
            self.mean_I_mc = np.full(self.n_iterations, sample_mc.mean_I)
            self.var_I_mc = np.full(self.n_iterations, sample_mc.var_I)
            self.re_I_mc = np.full(self.n_iterations, sample_mc.re_I)
            self.time_steps_mc = np.full(self.n_iterations, sample_mc.k)
            self.ct_mc = np.full(self.n_iterations, sample_mc.ct)
        else:
            self.value_f_mc = np.full(self.n_iterations, np.nan)
            self.mean_I_mc = np.full(self.n_iterations, np.nan)
            self.var_I_mc = np.full(self.n_iterations, np.nan)
            self.re_I_mc = np.full(self.n_iterations, np.nan)
            self.time_steps_mc = np.full(self.n_iterations, np.nan)
            self.ct_mc = np.full(self.n_iterations, np.nan)

    def load_hjb_solution_and_sampling(self, h_hjb=0.1, dt_hjb=0.01, N_hjb=10**3, seed=None):
        '''
        '''
        from sde_importance_sampling.importance_sampling import Sampling

        # save parameters
        self.h_hjb = h_hjb
        self.dt_hjb = dt_hjb
        self.N_hjb = N_hjb

        # load hjb solver
        if self.sample.problem_name == 'langevin_stop-t':
            self.sol_hjb = self.sample.get_hjb_solver(h_hjb)
        elif self.sample.problem_name == 'langevin_det-t':
            self.sol_hjb = self.sample.get_hjb_solver_det(h_hjb, dt_hjb)

        # break if there is no hjb solution
        if self.sol_hjb is None:
            return

        if self.sample.problem_name == 'langevin_stop-t':
            hjb_psi_at_x = self.sol_hjb.get_psi_at_x(self.sample.xzero)
            hjb_value_f_at_x = self.sol_hjb.get_value_function_at_x(self.sample.xzero)
            self.psi_hjb = np.full(self.n_iterations, hjb_psi_at_x)
            self.value_f_hjb = np.full(self.n_iterations, hjb_value_f_at_x)
        elif self.sample.problem_name == 'langevin_det-t':
            hjb_psi_at_x = self.sol_hjb.get_psi_t_x(0., self.sample.xzero)
            hjb_value_f_at_x = self.sol_hjb.get_value_funtion_t_x(0., self.sample.xzero)
            self.psi_hjb = np.full(self.n_iterations, hjb_psi_at_x)
            self.value_f_hjb = np.full(self.n_iterations, hjb_value_f_at_x)

        # load hjb sampling
        sample_hjb = self.sample.get_hjb_sampling(self.sol_hjb.dir_path, dt_hjb, N_hjb, seed)

        # break if there is no hjb sampling
        if sample_hjb is None:
            return

        self.value_f_is_hjb = np.full(self.n_iterations, -np.log(sample_hjb.mean_I_u))
        self.mean_I_u_hjb = np.full(self.n_iterations, sample_hjb.mean_I_u)
        self.var_I_u_hjb = np.full(self.n_iterations, sample_hjb.var_I_u)
        self.re_I_u_hjb = np.full(self.n_iterations, sample_hjb.re_I_u)
        self.time_steps_is_hjb = np.full(self.n_iterations, sample_hjb.k)
        self.ct_is_hjb = np.full(self.n_iterations, sample_hjb.ct)

    def load_plot_labels_colors_and_linestyles(self):
        '''
        '''

        if self.sol_hjb is None:
            self.colors = ['tab:blue', 'tab:orange']
            self.linestyles = ['-', 'dashed']
            self.labels = ['SOC', 'Not contronlled Sampling']

        else :
            self.colors = ['tab:blue', 'tab:orange', 'tab:grey', 'tab:cyan']
            self.linestyles = ['-', 'dashed', 'dashdot', 'dashdot']
            self.labels = [
                'SOC',
                'not contronlled sampling',
                'HJB sampling',
                r'HJB solution (h={:.0e})'.format(self.h_hjb),
            ]


    def plot_loss(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_losses'):
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # loss, mc sampling, hjb sampling and hjb solution
        if self.sol_hjb is None:
            y = np.vstack((
                self.run_avg_losses,
                self.value_f_mc[self.n_iter_avg - 1:],
            ))
        else:
            y = np.vstack((
                self.run_avg_losses,
                self.value_f_mc[self.n_iter_avg - 1:],
                self.value_f_is_hjb[self.n_iter_avg - 1:],
                self.value_f_hjb[self.n_iter_avg - 1:],
            ))

        # loss figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='loss',
        )
        fig.set_title(r'$\tilde{J}(\theta; x_0)$')
        fig.set_xlabel('SGD iterations')
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels, self.colors, self.linestyles)

    def plot_var_loss(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_vars_loss'):
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # var loss
        y = self.run_avg_vars_loss
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='var-loss',
        )
        fig.set_title(r'${var}_N(\tilde{J}(\theta; x_0))$')
        fig.set_xlabel('SGD iterations')
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels[0], self.colors[0], self.linestyles[0])


    def plot_mean_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_means_I_u'):
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # mean I^u, mc sampling, hjb sampling and hjb solution
        if self.sol_hjb is None:
            y = np.vstack((
                self.run_avg_means_I_u,
                self.mean_I_mc[self.n_iter_avg - 1:],
            ))
        else:
            y = np.vstack((
                self.run_avg_means_I_u,
                self.mean_I_mc[self.n_iter_avg - 1:],
                self.mean_I_u_hjb[self.n_iter_avg - 1:],
                self.psi_hjb[self.n_iter_avg - 1:],
            ))

        # mean I^u figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='mean',
        )
        fig.set_title(r'$m_N(I^u)$')
        fig.set_xlabel('SGD iterations')
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.labels, self.colors, self.linestyles)

    def plot_var_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_vars_I_u'):
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # re I^u, mc sampling, hjb sampling and hjb solution
        if self.sol_hjb is None:
            y = np.vstack((
                self.run_avg_vars_I_u,
                self.var_I_mc[self.n_iter_avg - 1:],
            ))
        else:
            y = np.vstack((
                self.run_avg_vars_I_u,
                self.var_I_mc[self.n_iter_avg - 1:],
                self.var_I_u_hjb[self.n_iter_avg - 1:],
            ))

        # relative error figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='var',
        )
        fig.set_title(r'${var}_N(I^u)$')
        fig.set_xlabel('SGD iterations')
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
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # re I^u, mc sampling, hjb sampling and hjb solution
        if self.sol_hjb is None:
            y = np.vstack((
                self.run_avg_res_I_u,
                self.re_I_mc[self.n_iter_avg - 1:],
            ))
        else:
            y = np.vstack((
                self.run_avg_res_I_u,
                self.re_I_mc[self.n_iter_avg - 1:],
                self.re_I_u_hjb[self.n_iter_avg - 1:],
            ))

        # relative error figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='re',
        )
        fig.set_title(r'${re}_N(I^u)$')
        fig.set_xlabel('SGD iterations')
        fig.set_plot_scale('semilogy')
        if self.sol_hjb is None:
            fig.plot(x, y, self.labels, self.colors, self.linestyles)
        else:
            fig.plot(x, y, self.labels[:3], self.colors[:3], self.linestyles[:3])

    def plot_error_bar_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iterations)

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
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # time steps, mc sampling, hjb sampling and hjb solution
        if self.sol_hjb is None:
            y = np.vstack((
                self.run_avg_time_steps,
                self.time_steps_mc[self.n_iter_avg - 1:],
            ))
        else:
            y = np.vstack((
                self.run_avg_time_steps,
                self.time_steps_mc[self.n_iter_avg - 1:],
                self.time_steps_is_hjb[self.n_iter_avg - 1:],
            ))

        # time steps figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='time-steps',
        )
        fig.set_title(r'TS')
        fig.set_xlabel('SGD iterations')
        fig.set_plot_scale('semilogy')

        if self.sol_hjb is None:
            fig.plot(x, y, self.labels, self.colors, self.linestyles)
        else:
            fig.plot(x, y, self.labels[:3], self.colors[:3], self.linestyles[:3])

    def plot_cts(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        if not hasattr(self, 'run_avg_cts'):
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # computational time, mc sampling, hjb sampling and hjb solution
        if self.sol_hjb is None:
            y = np.vstack((
                self.run_avg_cts,
                self.ct_mc[self.n_iter_avg - 1:],
            ))
        else:
            y = np.vstack((
                self.run_avg_cts,
                self.ct_mc[self.n_iter_avg - 1:],
                self.ct_is_hjb[self.n_iter_avg - 1:],
            ))

        # cts figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='cts',
        )
        fig.set_title(r'CT (s)')
        fig.set_xlabel('SGD iterations')
        fig.set_plot_scale('semilogy')

        if self.sol_hjb is None:
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
            self.compute_running_averages(n_iter_avg=1)

        # iterations
        x = np.arange(self.n_iterations)[self.n_iter_avg - 1:]

        # u l2 error
        y = self.run_avg_u_l2_errors

        # u l2 error figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='u-l2-error',
        )
        fig.set_xlabel('SGD iterations')
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

    def compute_cts_sum(self):
        self.cts_sum = np.array([np.sum(self.cts[:i]) for i in range(self.n_iterations+1)])

    def plot_re_I_u_cts(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # check if running averages are computed
        #if not hasattr(self, 'run_avg_res_I_u'):
        #    self.compute_running_averages(n_iter_avg=1)

        # computational time used
        x = self.cts_sum[:-1]

        # re I^u, mc sampling, hjb sampling and hjb solution
        y = self.res_I_u

        # relative error figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='ct-re',
        )
        fig.set_title(r'${re}_N(I^u)$')
        fig.set_xlabel('CT (s)')
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

        #fig.set_title(r'$u_i(x; \theta_{3999})$, $\theta_0$ = meta')
        fig.set_xlabel(r'$x_i$')
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

        # last iteration if i is not given
        if i is None:
            i = self.n_iterations - 1
        assert i < self.n_iterations, ''

        # set theta
        if self.sample.ansatz is not None:
            self.sample.ansatz.theta = self.thetas[i]
        elif self.sample.nn_func_appr is not None:
            self.sample.nn_func_appr.model.load_parameters(self.thetas[i])

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=h)
        self.sample.get_grid_control()

        return np.copy(self.sample.grid_control)

    def plot_1d_iteration(self, i=None):
        from figures.myfigure import MyFigure

        # last iteration if i is not given
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
        self.sample.discretize_domain(h=0.001)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.get_controlled_potential_and_drift()

        # colors and labels
        labels = [r'SOC (iteration: {})'.format(i), 'HJB solution']
        colors = ['tab:blue', 'tab:cyan']

        # domain
        x = self.sample.domain_h[:, 0]

        if self.sample.grid_value_function is not None:

            # plot value function
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='value-function' + ext,
            )
            y = np.vstack((
                self.sample.grid_value_function,
                sol_hjb.value_f,
            ))
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, y, labels=labels, colors=colors)

            # plot controlled potential
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='controlled-potential' + ext,
            )
            y = np.vstack((
                self.sample.grid_controlled_potential,
                sol_hjb.controlled_potential,
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
        from figures.myfigure import MyFigure

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        x = self.sample.domain_h[:, 0]

        # filter iterations to show
        iterations = np.arange(self.n_iterations)
        sliced_iterations = slice_1d_array(iterations, n_elements=5)
        n_sliced_iterations = sliced_iterations.shape[0]

        # preallocate arrays
        controlled_potentials = np.zeros((n_sliced_iterations + 1, x.shape[0]))
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
                controlled_potentials[idx, :] = self.sample.grid_controlled_potential
                value_fs[idx, :] = self.sample.grid_value_function
            controls[idx, :] = self.sample.grid_control[:, 0]

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.get_controlled_potential_and_drift()
        labels.append('HJB solution')
        colors.append('tab:cyan')

        if self.sample.grid_value_function is not None:

            # plot value function
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.dir_path,
                file_name='value-function',
            )
            value_fs[-1, :] = sol_hjb.value_f
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, value_fs, labels=labels, colors=colors)

            # plot controlled potential
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.dir_path,
                file_name='controlled-potential',
            )
            controlled_potentials[-1, :] = sol_hjb.controlled_potential
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, controlled_potentials, labels=labels, colors=colors)


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
