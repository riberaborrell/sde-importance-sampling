from mds.plots import Plot
from mds.utils import make_dir_path, get_som_dir_path, get_time_in_hms
from mds.numeric_utils import slice_1d_array

import numpy as np
import torch
import torch.optim as optim

import time
import os

class StochasticOptimizationMethod:
    '''
    '''
    def __init__(self, sample, loss_type, optimizer, lr, iterations_lim,
                 do_iteration_plots=False):
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
        self.iterations_lim = iterations_lim

        # per iteration
        self.m = None
        self.iterations = None
        self.thetas = None
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

        # running averages
        self.n_last_iter = None
        self.run_avg_mean_I_u = None
        self.run_avg_var_I_u = None
        self.run_avg_re_I_u = None
        self.run_avg_loss = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # flag for plotting at each iteration
        self.do_iteration_plots = do_iteration_plots

        # flag for computing l2 error
        self.do_u_l2_error = False

        # set path
        self.dir_path = None
        self.set_dir_path()
        if do_iteration_plots:
            self.set_iterations_dir_path()
        else:
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
        )

    def set_iterations_dir_path(self):
        self.iterations_dir_path = os.path.join(self.dir_path, 'iterations')
        make_dir_path(self.iterations_dir_path)

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def preallocate_arrays(self):

        self.thetas = np.empty((self.iterations_lim, self.m))
        self.losses = np.empty(self.iterations_lim)
        self.means_I_u = np.empty(self.iterations_lim)
        self.vars_I_u = np.empty(self.iterations_lim)
        self.res_I_u = np.empty(self.iterations_lim)

        if self.sample.do_u_l2_error:
            self.u_l2_errors = np.empty(self.iterations_lim)

        self.time_steps = np.empty(self.iterations_lim, dtype=int)
        self.cts = np.empty(self.iterations_lim)

        if self.sample.ansatz is not None:
            self.grad_losses = np.empty((self.iterations_lim, self.m))

        if self.sample.nn_func_appr is not None and self.loss_type == 'ipa':
            self.ipa_losses = np.empty(self.iterations_lim)
        elif self.sample.nn_func_appr is not None and self.loss_type == 're':
            self.re_losses = np.empty(self.iterations_lim)
        elif self.sample.nn_func_appr is not None and self.loss_type == 'logvar':
            self.logvar_losses = np.empty(self.iterations_lim)

    def cut_arrays(self):
        assert self.n_iterations < self.iterations_lim, ''

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

    def sgd_ipa_gaussian_ansatz(self):
        self.start_timer()

        # number of parameters
        self.m = self.sample.ansatz.m

        # preallocate parameters and losses
        self.preallocate_arrays()

        for i in np.arange(self.iterations_lim):
            # plot control, free_energy and tilted potential
            if self.do_iteration_plots:
                pass

            # compute loss and its gradient 
            succ = self.sample.sample_loss_ipa_ansatz()
            print('{:d}, {:2.3f}'.format(i, self.sample.loss))

            # check if sample succeeded
            if not succ:
                break

            # save parameters, statistics and losses
            self.n_iterations = i + 1
            self.thetas[i, :] = self.sample.ansatz.theta
            self.losses[i] = self.sample.loss
            self.grad_losses[i, :] = self.sample.grad_loss
            self.means_I_u[i] = self.sample.mean_I_u
            self.vars_I_u[i] = self.sample.var_I_u
            self.res_I_u[i] = self.sample.re_I_u

            if self.sample.do_u_l2_error:
                self.u_l2_errors[i] = self.sample.u_l2_error

            self.time_steps[i] = int(np.max(self.sample.fht) / self.sample.dt)
            self.cts[i] = self.sample.ct

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

        for i in np.arange(self.iterations_lim):

            # u l2 error flag
            if self.do_u_l2_error:
                self.sample.do_u_l2_error = True

            # compute ipa loss
            if self.loss_type == 'ipa':
                succ, time_steps = self.sample.sample_loss_ipa_nn(device)

            # compute relative entropy loss
            elif self.loss_type == 're':
                succ, time_steps = self.sample.sample_re_loss_nn(device)

            print('{:d}, {:2.3f}'.format(i, self.sample.loss))
            #print('{:d}, {:2.3f}'.format(i, self.sample.re_loss.detach()))

            # check if sample succeeded
            if not succ:
                break

            # allocate
            self.iterations = np.append(self.iterations, i)
            self.thetas = np.vstack((self.thetas, model.get_parameters()))
            self.means_I_u = np.append(self.means_I_u, self.sample.mean_I_u)
            self.vars_I_u = np.append(self.vars_I_u, self.sample.var_I_u)
            self.res_I_u = np.append(self.res_I_u, self.sample.re_I_u)
            self.time_steps = np.append(self.time_steps, time_steps)

            self.losses = np.append(self.losses, self.sample.loss)
            if self.loss_type == 'ipa':
                self.ipa_losses = np.append(
                    self.ipa_losses, self.sample.ipa_loss.detach().numpy()
                )
            elif self.loss_type == 're':
                self.re_losses = np.append(self.re_losses, self.sample.re_loss.detach().numpy())

            if self.sample.u_l2_error is not None:
                self.u_l2_errors = np.append(self.u_l2_errors, self.sample.u_l2_error)


            # reset gradients
            optimizer.zero_grad()

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
        file_path = os.path.join(self.dir_path, 'som.npz')
        np.savez(
            file_path,
            iterations=self.iterations,
            thetas=self.thetas,
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
            t_initial=self.t_initial,
            t_final=self.t_final,
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

    def compute_running_averages(self, n_last_iter=10):
        self.n_last_iter = n_last_iter
        self.run_avg_mean_I_u = np.mean(self.means_I_u[n_last_iter:])
        self.run_avg_var_I_u = np.mean(self.vars_I_u[n_last_iter:])
        self.run_avg_re_I_u = np.mean(self.res_I_u[n_last_iter:])
        self.run_avg_loss = np.mean(self.losses[n_last_iter:])

    def write_report(self):
        sample = self.sample

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, 'w')

        sample.write_setting(f)
        sample.write_euler_maruyama_parameters(f)
        sample.write_sampling_parameters(f)

        if self.sample.ansatz is not None:
            pass
        elif self.sample.nn_func_appr is not None:
            sample.nn_func_appr.write_parameters(f)

        f.write('\nStochastic optimization method parameters\n')
        f.write('loss type: {}\n'.format(self.loss_type))
        f.write('som type: {}\n\n'.format(self.optimizer))

        f.write('lr: {}\n'.format(self.lr))
        f.write('iterations lim: {}\n'.format(self.iterations_lim))

        f.write('iterations used: {}\n'.format(self.iterations[-1]))
        f.write('total time steps: {:,d}\n'.format(int(self.time_steps.sum())))

        f.write('\nLast iteration\n')
        f.write('E[I_u]: {:2.3e}\n'.format(self.means_I_u[-1]))
        f.write('Var[I_u]: {:2.3e}\n'.format(self.vars_I_u[-1]))
        f.write('RE[I_u]: {:2.3f}\n'.format(self.res_I_u[-1]))
        f.write('F: {:2.3f}\n'.format(- np.log(self.means_I_u[-1])))
        f.write('loss function: {:2.3f}\n'.format(self.losses[-1]))

        self.compute_running_averages()
        f.write('\nRunning averages of last {:d} iterations\n'.format(self.n_last_iter))
        f.write('E[I_u]: {:2.3e}\n'.format(self.run_avg_mean_I_u))
        f.write('Var[I_u]: {:2.3e}\n'.format(self.run_avg_var_I_u))
        f.write('RE[I_u]: {:2.3f}\n'.format(self.run_avg_re_I_u))
        f.write('F: {:2.3f}\n'.format(- np.log(self.run_avg_mean_I_u)))
        f.write('loss function: {:2.3f}\n\n'.format(self.run_avg_loss))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        #self.write_iteration_report(f)
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def write_iteration_report(self, f):
        for i in self.iterations:
            f.write('iteration = {:d}\n'.format(i))
            f.write('theta = {}\n'.format(self.thetas[i]))
            f.write('loss = {:2.4e}\n'.format(self.losses[i]))
            f.write('grad_loss = {}\n'.format(self.grad_losses[i]))
            f.write('|grad_loss|_2 = {:2.4e}\n\n'
                    ''.format(np.linalg.norm(self.grad_losses[i])))
            f.write('time steps = {}\n'.format(self.time_steps[i]))

    def plot_losses(self, h_hjb=0.1, dt_mc=0.001, N_mc=100000):

        # hjb F at xzero
        h_hjb=0.001
        sol = self.sample.get_hjb_solver(h_hjb)
        if sol.F is not None:
            hjb_f_at_x = sol.get_f_at_x(self.sample.xzero)
            value_f_hjb = np.full(self.iterations.shape[0], hjb_f_at_x)
        else:
            value_f_hjb = np.full(self.iterations.shape[0], np.nan)

        # mc F 
        sample_mc = self.sample.get_not_controlled_sampling(dt_mc, N_mc)
        if sample_mc.mean_I is not None:
            mc_f = - np.log(sample_mc.mean_I)
            value_f_mc = np.full(self.iterations.shape[0], mc_f)
        else:
            value_f_mc = np.full(self.iterations.shape[0], np.nan)

        ys = np.vstack((self.losses, value_f_hjb, value_f_mc))
        colors = ['tab:blue', 'tab:orange', 'tab:cyan']
        linestyles = ['-', 'dashed', 'dashdot']
        labels = [
            r'SGD',
            'MC Sampling (dt={:.0e}, N={:.0e})'.format(dt_mc, N_mc),
            'HJB solution (h={:.0e})'.format(h_hjb),
        ]

        plt = Plot(self.dir_path, 'loss')
        plt.xlabel = 'iterations'
        #plt.set_ylim(1, 3) # n=1, beta=1
        #plt.set_ylim(3, 5) # n=2, beta=1
        #plt.set_ylim(4, 10) # n=3, beta=1
        #plt.set_ylim(5, 10) # n=4, beta=1
        breakpoint()
        plt.multiple_lines_plot(self.iterations, ys, colors, linestyles, labels)


    def plot_I_u(self, dt_mc=0.001, N_mc=100000):

        # not controlled sampling 
        sample_mc = self.sample.get_not_controlled_sampling(dt_mc, N_mc)

        # mc mean I
        if sample_mc.mean_I is not None:
            mean_I_mc = np.full(self.iterations.shape[0], sample_mc.mean_I)
        else:
            mean_I_mc = np.full(self.iterations.shape[0], np.nan)

        # mc var I
        if sample_mc.var_I is not None:
            var_I_mc = np.full(self.iterations.shape[0], sample_mc.var_I)
        else:
            var_I_mc = np.full(self.iterations.shape[0], np.nan)

        # mc re I
        if sample_mc.re_I is not None:
            re_I_mc = np.full(self.iterations.shape[0], sample_mc.re_I)
        else:
            re_I_mc = np.full(self.iterations.shape[0], np.nan)

        colors = ['tab:blue', 'tab:orange']
        linestyles = ['-', 'dashed']
        labels = [
            'SGD',
            'MC Sampling (dt={:.0e}, N={:.0e})'.format(dt_mc, N_mc),
        ]

        # plot mean I_u
        plt = Plot(self.dir_path, 'I_u_mean')
        plt.xlabel = 'iterations'
        #plt.set_ylim(0.125, 0.2) # n=1, beta=1
        #plt.set_ylim(0.03, 0.05) # n=2, beta=1
        #plt.set_ylim(0.005, 0.02) # n=3, beta=1
        #plt.set_ylim(0, 0.005) # n=4, beta=1
        breakpoint()
        ys = np.vstack((self.means_I_u, mean_I_mc))
        plt.multiple_lines_plot(self.iterations, ys, colors, linestyles, labels)

        # plot var I_u
        plt = Plot(self.dir_path, 'I_u_var')
        plt.xlabel = 'iterations'
        plt.set_logplot()
        ys = np.vstack((self.vars_I_u, var_I_mc))
        plt.multiple_lines_plot(self.iterations, ys, colors, linestyles, labels)

        # plot re I_u
        plt = Plot(self.dir_path, 'I_u_re')
        plt.xlabel = 'iterations'
        plt.set_logplot()
        ys = np.vstack((self.res_I_u, re_I_mc))
        plt.multiple_lines_plot(self.iterations, ys, colors, linestyles, labels)
        return

        plt = Plot(self.dir_path, 'I_u')
        plt.plt.plot(self.iterations, self.means_I_u, color='b', linestyle='-')
        plt.plt.fill_between(
            self.iterations,
            self.means_I_u - self.vars_I_u,
            self.means_I_u + self.vars_I_u,
        )
        #plt.plt.ylim(0.03, 0.05) # n=2, beta=1
        breakpoint()
        plt.plt.savefig(plt.file_path)
        plt.plt.close()

    def plot_time_steps(self):
        plt = Plot(self.dir_path, 'time_steps_log')
        plt.set_scientific_notation('y')
        plt.xlabel = 'iterations'
        plt.set_logplot()
        plt.one_line_plot(self.iterations, self.time_steps, color='purple', label='TS')

    def plot_1d_iteration(self, i=None):
        # last iteration is i is not given
        if i is None:
            i = self.iterations[-1]

        assert i in self.iterations, ''

        self.set_iterations_dir_path()
        ext = '_iter{}'.format(i)
        label = r'iteration: {}'.format(i)

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
        sol = self.sample.get_hjb_solver(h=0.001)
        sol.get_controlled_potential_and_drift()

        # do plots
        if self.sample.grid_value_function is not None:
            self.sample.plot_1d_free_energy(self.sample.grid_value_function, sol.F,
                                            label=label, dir_path=self.iterations_dir_path,
                                            ext=ext)

        self.sample.plot_1d_control(self.sample.grid_control[:, 0], sol.u_opt[:, 0],
                                    label=label, dir_path=self.iterations_dir_path, ext=ext)


    def plot_1d_iterations(self):

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        x = self.sample.domain_h[:, 0]

        # filter iterations to show
        sliced_iterations = slice_1d_array(self.iterations, n_elements=5)

        # preallocate functions
        labels = []
        controlled_potentials = np.zeros((sliced_iterations.shape[0], x.shape[0]))
        frees = np.zeros((sliced_iterations.shape[0], x.shape[0]))
        controls = np.zeros((sliced_iterations.shape[0], x.shape[0]))

        for idx, i in enumerate(sliced_iterations):
            labels.append(r'iteration = {:d}'.format(i))

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
        sol = self.sample.get_hjb_solver(h=0.001)
        sol.get_controlled_potential_and_drift()

        # do plots
        if self.sample.grid_value_function is not None:
            self.sample.plot_1d_controlled_potentials(controlled_potentials,
                                                      controlledV_hjb=sol.controlled_potential,
                                                      labels=labels[:], dir_path=self.dir_path)
            self.sample.plot_1d_free_energies(frees, F_hjb=sol.F, labels=labels[:],
                                              dir_path=self.dir_path)
        self.sample.plot_1d_controls(controls, u_hjb=sol.u_opt[:, 0],
                                     labels=labels[:], dir_path=self.dir_path)

    def plot_2d_iteration(self, i=None):

        # last iteration is i is not given
        if i is None:
            i = self.iterations[-1]

        assert i in self.iterations, ''

        self.set_iterations_dir_path()
        ext = '_iter{}'.format(i)

        # set theta
        if self.sample.ansatz is not None:
            self.sample.ansatz.theta = self.thetas[i]
        elif self.sample.nn_func_appr is not None:
            self.sample.nn_func_appr.model.load_parameters(self.thetas[i])

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.05)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        if self.sample.grid_value_function is not None:
            self.sample.plot_2d_controlled_potential(self.sample.grid_controlled_potential,
                                                     self.iterations_dir_path, ext)
        self.sample.plot_2d_control(self.sample.grid_control, self.iterations_dir_path, ext)
        self.sample.plot_2d_controlled_drift(self.sample.grid_controlled_drift,
                                             self.iterations_dir_path, ext)
