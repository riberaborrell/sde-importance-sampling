from mds.utils_path import make_dir_path, get_som_dir_path, get_time_in_hms
from mds.utils_numeric import slice_1d_array

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
        self.iterations_dir_path = os.path.join(self.dir_path, 'iterations')
        make_dir_path(self.iterations_dir_path)

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def preallocate_arrays(self):

        if self.save_thetas_all_it:
            self.thetas = np.empty((self.n_iterations_lim, self.m))
        self.losses = np.empty(self.n_iterations_lim)
        self.means_I_u = np.empty(self.n_iterations_lim)
        self.vars_I_u = np.empty(self.n_iterations_lim)
        self.res_I_u = np.empty(self.n_iterations_lim)

        if self.sample.do_u_l2_error:
            self.u_l2_errors = np.empty(self.n_iterations_lim)

        self.time_steps = np.empty(self.n_iterations_lim, dtype=int)
        self.cts = np.empty(self.n_iterations_lim)

        if self.sample.ansatz is not None:
            self.grad_losses = np.empty((self.n_iterations_lim, self.m))

        elif self.sample.nn_func_appr is not None and self.loss_type == 'logvar':
            self.logvar_losses = np.empty(self.n_iterations_lim)

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

        # add loss and gradient
        self.losses[i] = self.sample.loss

        if self.sample.ansatz is not None:
            self.grad_losses[i, :] = self.sample.grad_loss
        elif self.sample.nn_func_appr is not None and self.loss_type == 'logvar':
            self.logvar_losses[i] = self.sample.logvar_loss

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

    def cut_arrays(self):
        assert self.n_iterations < self.n_iterations_lim, ''

        self.thetas = self.thetas[:self.n_iterations, :]
        self.losses = self.losses[:self.n_iterations]
        self.means_I_u = self.means_I_u[:self.n_iterations]
        self.vars_I_u = self.vars_I_u[:self.n_iterations]
        self.res_I_u = self.res_I_u[:self.n_iterations]

        if self.sample.do_u_l2_error:
            self.u_l2_errors = self.u_l2_errors[:self.n_iterations]

        self.time_steps = self.time_steps[:self.n_iterations]
        self.cts = self.cts[:self.n_iterations]

        if self.sample.ansatz is not None:
            self.grad_losses = self.grad_losses[:self.n_iterations, :]

        if self.sample.nn_func_appr is not None and self.loss_type == 'ipa':
            self.ipa_losses = self.ipa_losses[:self.n_iterations, :]
        elif self.sample.nn_func_appr is not None and self.loss_type == 're':
            self.re_losses = self.re_losses[:self.n_iterations, :]
        elif self.sample.nn_func_appr is not None and self.loss_type == 'logvar':
            self.logvar_losses = self.logvar_losses[:self.n_iterations, :]

    def get_iteration_statistics(self, i):
        msg = 'it.: {:d}, loss: {:2.3f}, mean I^u: {:2.3e}, re I^u: {:2.3f},' \
              'time steps: {:2.1e}'.format(
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

            # compute logvar loss
            elif self.loss_type == 'logvar':
                succ = self.sample.sample_loss_logvar_nn(device)

            # check if sample succeeded
            if not succ:
                break

            # save information of the iteration
            self.update_arrays(i)

            # print iteration info
            msg = self.get_iteration_statistics(i)
            print(msg)

            # compute gradients
            if self.loss_type == 'ipa':
                self.sample.ipa_loss.backward()
            elif self.loss_type == 're':
                self.sample.re_loss.backward()
            elif self.loss_type == 'logvar':
                self.sample.logvar_loss.backward()

            # update parameters
            optimizer.step()

        self.stop_timer()
        self.save()


    def save(self):
        # create directories of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save npz file
        np.savez(
            os.path.join(self.dir_path, 'som.npz'),
            seed=self.sample.seed,
            dt=self.sample.dt,
            N=self.sample.N,
            n_iterations=self.n_iterations,
            thetas=self.thetas,
            last_thetas=self.last_thetas,
            losses=self.losses,
            ipa_losses=self.ipa_losses,
            re_losses=self.re_losses,
            grad_losses=self.grad_losses,
            means_I_u=self.means_I_u,
            vars_I_u=self.vars_I_u,
            res_I_u=self.res_I_u,
            u_l2_errors=self.u_l2_errors,
            time_steps=self.time_steps,
            cts=self.cts,
            ct=self.ct,
        )

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

    def compute_running_averages(self, n_iter_avg=10):
        self.n_iter_avg = n_iter_avg
        self.run_avg_losses = np.convolve(
            self.losses, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )
        self.run_avg_means_I_u = np.convolve(
            self.means_I_u, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )
        self.run_avg_vars_I_u = np.convolve(
            self.vars_I_u, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )
        self.run_avg_res_I_u = np.convolve(
            self.res_I_u, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )
        self.run_avg_time_steps = np.convolve(
            self.time_steps, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
        )
        if self.u_l2_errors is not None and self.u_l2_errors.ndim != 0:
            self.run_avg_u_l2_errors = np.convolve(
                self.u_l2_errors, np.ones(n_iter_avg) / n_iter_avg, mode='valid'
            )

    def write_report(self):
        sample = self.sample

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, 'w')

        #sample.write_setting(f)
        #sample.write_euler_maruyama_parameters(f)
        #sample.write_sampling_parameters(f)

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
        f.write('F: {:2.3f}\n'.format(- np.log(self.means_I_u[-1])))
        f.write('loss function: {:2.3f}\n'.format(self.losses[-1]))
        if self.u_l2_errors is not None:
           f.write('u l2 error: {:2.3f}\n'.format(self.u_l2_errors[-1]))

        self.compute_running_averages()
        f.write('\nRunning averages of last {:d} iterations\n'.format(self.n_iter_avg))
        f.write('E[I_u]: {:2.3e}\n'.format(self.run_avg_means_I_u[-1]))
        f.write('Var[I_u]: {:2.3e}\n'.format(self.run_avg_vars_I_u[-1]))
        f.write('RE[I_u]: {:2.3f}\n'.format(self.run_avg_res_I_u[-1]))
        f.write('F: {:2.3f}\n'.format(- np.log(self.run_avg_means_I_u[-1])))
        f.write('loss function: {:2.3f}\n\n'.format(self.run_avg_losses[-1]))
        if self.run_avg_u_l2_errors is not None:
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

    def load_mc_sampling(self, dt_mc=0.01, N_mc=10**3):
        '''
        '''

        # save parameters
        self.dt_mc = dt_mc
        self.N_mc = N_mc

        # load mc sampling
        sample_mc = self.sample.get_not_controlled_sampling(dt_mc, N_mc)

        if sample_mc.mean_I is not None:
            self.value_f_mc = np.full(self.n_iterations, - np.log(sample_mc.mean_I))
            self.mean_I_mc = np.full(self.n_iterations, sample_mc.mean_I)
            self.re_I_mc = np.full(self.n_iterations_lim, sample_mc.re_I)
            self.time_steps_mc = np.full(self.n_iterations_lim, sample_mc.k)
            self.ct_mc = np.full(self.n_iterations_lim, sample_mc.ct)
        else:
            self.value_f_mc = np.full(self.n_iterations, np.nan)
            self.mean_I_mc = np.full(self.n_iterations, np.nan)
            self.re_I_mc = np.full(self.n_iterations, np.nan)
            self.time_steps_mc = np.full(self.n_iterations, np.nan)
            self.ct_mc = np.full(self.n_iterations, np.nan)

    def load_hjb_solution_and_sampling(self, h_hjb=0.1, dt_hjb=0.01, N_hjb=10**3):
        '''
        '''
        from mds.langevin_nd_importance_sampling import Sampling
        # save parameters
        self.h_hjb = h_hjb
        self.dt_hjb = dt_hjb
        self.N_hjb = N_hjb

        # load hjb solver
        sol_hjb = self.sample.get_hjb_solver_det(h_hjb, dt_hjb)
        if sol_hjb.F is not None:
            hjb_psi_at_x = sol_hjb.get_psi_at_x(self.sample.xzero)
            hjb_f_at_x = sol_hjb.get_f_at_x(self.sample.xzero)
            self.hjb_is_loaded = True
            self.f_hjb = np.full(self.n_iterations, hjb_f_at_x)
            self.psi_hjb = np.full(self.n_iterations_lim, hjb_psi_at_x)
        else:
            self.hjb_is_loaded = False
            self.f_hjb = np.full(self.n_iterations, np.nan)
            self.psi_hjb = np.full(self.n_iterations_lim, np.nan)

        # load hjb sampling
        sample_hjb = Sampling.new_from(self.sample)
        sample_hjb.is_controlled = True
        sample_hjb.N = N_hjb
        sample_hjb.dt = dt_hjb
        sample_hjb.set_controlled_dir_path(sol_hjb.dir_path)
        sample_hjb.load()
        if sample_hjb.mean_I_u is not None:
            self.value_f_is_hjb = np.full(self.n_iterations_lim, -np.log(sample_hjb.mean_I_u))
            self.mean_I_u_hjb = np.full(self.n_iterations_lim, sample_hjb.mean_I_u)
            self.re_I_u_hjb = np.full(self.n_iterations_lim, sample_hjb.re_I_u)
            self.time_steps_is_hjb = np.full(self.n_iterations_lim, sample_hjb.k)
            self.ct_is_hjb = np.full(self.n_iterations_lim, sample_hjb.ct)
        else:
            self.value_is_hjb = np.full(self.n_iterations, np.nan)
            self.mean_I_u_hjb = np.full(self.n_iterations, np.nan)
            self.re_I_u_hjb = np.full(self.n_iterations, np.nan)
            self.time_steps_is_hjb = np.full(self.n_iterations, np.nan)
            self.ct_is_hjb = np.full(self.n_iterations, np.nan)

    def load_plot_labels_colors_and_linestyles(self):

        self.colors = ['tab:blue', 'tab:orange', 'tab:grey', 'tab:cyan']
        self.linestyles = ['-', 'dashed', 'dashdot', 'dashdot']
        self.labels = [
            'SOC',
            'Not contronlled Sampling',
            'HJB Sampling',
            #r'SOC (dt={:.0e}, N={:.0e})'.format(self.sample.dt, self.sample.N),
            #r'Not contronlled Sampling (dt={:.0e}, N={:.0e})'.format(self.dt_mc, self.N_mc),
            #r'HJB Sampling (dt={:.0e}, N={:.0e})'.format(self.dt_hjb, self.N_hjb),
            r'HJB solution (h={:.0e})'.format(self.h_hjb),
        ]

    def plot_loss(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iterations)

        # plot losses
        y = np.vstack((
            self.losses,
            self.value_f_mc,
            self.value_f_is_hjb,
            self.f_hjb,
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='loss',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.colors, self.linestyles, self.labels)

        # plot running averages losses
        if not hasattr(self, 'run_avg_losses'):
            return

        y = np.vstack((
            self.run_avg_losses,
            self.value_f_mc[self.n_iter_avg - 1:],
            self.value_f_is_hjb[self.n_iter_avg - 1:],
            self.f_hjb[self.n_iter_avg - 1:],
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='loss-avg',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x[self.n_iter_avg - 1:], y, self.colors, self.linestyles, self.labels)


    def plot_mean_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iterations)

        # plot mean I^u
        y = np.vstack((
            self.means_I_u,
            self.mean_I_mc,
            self.mean_I_u_hjb,
            self.psi_hjb,
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='mean',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.colors, self.linestyles, self.labels)

        # plot running averages mean I^u
        if not hasattr(self, 'run_avg_means_I_u'):
            return

        y = np.vstack((
            self.run_avg_means_I_u,
            self.mean_I_mc[self.n_iter_avg - 1:],
            self.mean_I_u_hjb[self.n_iter_avg - 1:],
            self.psi_hjb[self.n_iter_avg - 1:],
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='mean-avg',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x[self.n_iter_avg - 1:], y, self.colors, self.linestyles, self.labels)

    def plot_re_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iterations)

        # plot relative error I^u
        y = np.vstack((
            self.res_I_u,
            self.re_I_mc,
            self.re_I_u_hjb,
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='re',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.colors[:3], self.linestyles[:3], self.labels[:3])

        # plot relative error I^u
        if not hasattr(self, 'run_avg_res_I_u'):
            return

        y = np.vstack((
            self.run_avg_res_I_u,
            self.value_f_mc[self.n_iter_avg - 1:],
            self.value_f_is_hjb[self.n_iter_avg - 1:],
            self.f_hjb[self.n_iter_avg - 1:],
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='re-avg',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x[self.n_iter_avg - 1:], y, self.colors, self.linestyles, self.labels)

    def plot_error_bar_I_u(self):
        '''
        '''
        from figures.myfigure import MyFigure
        #TODO! debug
        x = np.arange(self.n_iterations)
        plt.plot(x, self.means_I_u, color='b', linestyle='-')
        plt.fill_between(
            self.iterations,
            self.means_I_u - self.vars_I_u,
            self.means_I_u + self.vars_I_u,
        )
        plt.savefig(plt.file_path)
        plt.close()

    def plot_time_steps(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iterations)

        # plot time steps
        y = np.vstack((
            self.time_steps,
            self.time_steps_mc,
            self.time_steps_is_hjb,
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='time-steps',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.colors[:3], self.linestyles[:3], self.labels[:3])

        # plot running averages time steps
        if not hasattr(self, 'run_avg_losses'):
            return

        y = np.vstack((
            self.run_avg_time_steps,
            self.time_steps_mc[self.n_iter_avg - 1:],
            self.time_steps_is_hjb[self.n_iter_avg - 1:],
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='time-steps-avg',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x[self.n_iter_avg - 1:], y, self.colors[:3], self.linestyles[:3], self.labels[:3])

    def plot_cts(self):
        '''
        '''
        from figures.myfigure import MyFigure

        x = np.arange(self.n_iterations)
        y = np.vstack((
            self.cts,
            self.ct_mc,
            self.ct_is_hjb,
        ))
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='cts',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.colors[:3], self.linestyles[:3], self.labels[:3])

    def plot_u_l2_error(self):
        '''
        '''
        from figures.myfigure import MyFigure

        # iterations
        x = np.arange(self.n_iterations)

        # plot u l2 error
        y = self.u_l2_errors
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='u-l2-error',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x, y, self.colors[0], self.linestyles[0], self.labels[0])

        # plot running averages u l2 errors
        if not hasattr(self, 'run_avg_u_l2_errors'):
            return

        y = self.run_avg_u_l2_errors
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='u-l2-error-avg',
        )
        fig.set_xlabel = 'iterations'
        fig.set_plot_scale('semilogy')
        fig.plot(x[self.n_iter_avg - 1:], y, self.colors[0], self.linestyles[0], self.labels[0])

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
        fig.set_xlabel = 'iterations'
        fig.plot(x, y, self.colors[0], self.linestyles[0], self.labels[0])

    def plot_1d_iteration(self, i=None):
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
        self.sample.discretize_domain(h=0.001)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.get_controlled_potential_and_drift()

        # colors and labels
        labels = [r'SOC (iteration: {})'.format(i), 'num sol HJB PDE']
        colors = ['tab:blue', 'tab:cyan']

        # domain
        x = self.sample.domain_h[:, 0]

        if self.sample.grid_value_function is not None:

            # plot free energy
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='free-energy' + ext,
            )
            y = np.vstack((
                self.sample.grid_value_function,
                sol_hjb.F,
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
        frees = np.zeros((n_sliced_iterations + 1, x.shape[0]))
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
                frees[idx, :] = self.sample.grid_value_function
            controls[idx, :] = self.sample.grid_control[:, 0]

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.get_controlled_potential_and_drift()
        labels.append('num sol HJB PDE')
        colors.append('tab:cyan')

        if self.sample.grid_value_function is not None:

            # plot free energy
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.dir_path,
                file_name='free-energy',
            )
            frees[-1, :] = sol_hjb.F
            fig.set_xlabel = 'x'
            fig.set_xlim(-2, 2)
            fig.plot(x, frees, labels=labels, colors=colors)

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

            # plot free energy
            fig = plt.figure(
                FigureClass=MyFigure,
                dir_path=self.iterations_dir_path,
                file_name='free-energy' + ext,
            )
            fig.set_xlim(-2, 2)
            fig.set_ylim(-2, 2)
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
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.vector_field(X, Y, U, V, scale=30)

        # plot controlled drift
        U = self.sample.grid_controlled_drift[:, :, 0]
        V = self.sample.grid_controlled_drift[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.iterations_dir_path,
            file_name='controlled-drift' + ext,
        )
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
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.set_colormap('PuRd')
        fig.vector_field(X, Y, U, V, scale=10)
