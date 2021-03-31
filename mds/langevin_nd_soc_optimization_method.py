from mds.plots import Plot
from mds.utils import make_dir_path, get_som_dir_path, get_time_in_hms

import numpy as np
import torch
import torch.optim as optim

import time
import os

class StochasticOptimizationMethod:
    '''
    '''
    def __init__(self, sample, parametrization, optimizer, grad_estimator, lr, iterations_lim,
                 do_iteration_plots=False):
        '''
        '''
        # type of parametrization
        self.parametrization = parametrization

        # type of estimator for the gradient of the loss function
        self.grad_estimator = grad_estimator

        # type of optimization method
        self.optimizer = optimizer

        # sampling object to estimate the loss and its gradient
        self.sample = sample

        # (initial) learning rate and maximal number of iterations 
        self.lr = lr
        self.iterations_lim = iterations_lim

        # per iteration
        self.iterations = None
        self.thetas = None
        self.losses = None
        self.tilted_losses = None
        self.grad_losses = None
        self.means_I_u = None
        self.vars_I_u = None
        self.res_I_u = None
        self.time_steps = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # flag for plotting at each iteration
        self.do_iteration_plots = do_iteration_plots

        # set path
        self.dir_path = None
        self.set_dir_path()
        if do_iteration_plots:
            self.set_iterations_dir_path()
        else:
            self.iterations_dir_path = None

    def set_dir_path(self):
        if self.parametrization == 'gaussian-value-f':
            parametrization_dir_path = self.sample.ansatz.dir_path
        elif self.parametrization == 'two-layer-nn-control':
            parametrization_dir_path = self.sample.nn_model.dir_path

        self.dir_path = get_som_dir_path(
            parametrization_dir_path,
            self.grad_estimator,
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

    def sgd_ipa_gaussian_ansatz(self):
        self.start_timer()

        # sampling object
        sample = self.sample

        # number of parameters
        m = sample.ansatz.m

        # preallocate parameters and losses
        self.iterations = np.empty(0, dtype=int)
        self.thetas = np.empty((0, m))
        self.losses = np.empty(0)
        self.grad_losses = np.empty((0, m))
        self.time_steps = np.empty(0)

        for i in np.arange(self.iterations_lim):
            # plot control, free_energy and tilted potential
            if self.do_iteration_plots:
                pass

            # compute loss and its gradient 
            succ, loss, grad_loss, time_steps = sample.sample_loss_ansatz()
            print('{:d}, {:2.3f}'.format(i, loss))

            # check if sample succeeded
            if not succ:
                break

            # allocate
            self.iterations = np.append(self.iterations, i)
            self.thetas = np.vstack((self.thetas, sample.ansatz.theta))
            self.losses = np.append(self.losses, loss)
            self.grad_losses = np.vstack((self.grad_losses, grad_loss))
            self.time_steps = np.append(self.time_steps, time_steps)

            # update coefficients
            sample.ansatz.theta = self.thetas[i, :] - self.lr * self.grad_losses[i, :]

        self.stop_timer()
        self.save_som()

    def som_ipa_nn(self):
        self.start_timer()

        # model and number of parameters
        model = self.sample.nn_model
        m = model.d_flatten

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
        )

        # preallocate parameters and losses
        self.iterations = np.empty(0, dtype=int)
        self.thetas = np.empty((0, m))
        self.losses = np.empty(0)
        self.tilted_losses = np.empty(0)
        self.means_I_u = np.empty(0)
        self.vars_I_u = np.empty(0)
        self.res_I_u = np.empty(0)
        self.time_steps = np.empty(0)

        for i in np.arange(self.iterations_lim):

            # compute loss
            succ, time_steps = self.sample.sample_loss_nn(device)
            print('{:d}, {:2.3f}'.format(i, self.sample.loss))

            # check if sample succeeded
            if not succ:
                break

            # allocate
            self.iterations = np.append(self.iterations, i)
            self.thetas = np.vstack((self.thetas, model.get_flatten_parameters()))
            self.losses = np.append(self.losses, self.sample.loss)
            self.tilted_losses = np.append(self.tilted_losses,
                                           self.sample.tilted_loss.detach().numpy())
            self.means_I_u = np.append(self.means_I_u, self.sample.mean_I_u)
            self.vars_I_u = np.append(self.vars_I_u, self.sample.var_I_u)
            self.res_I_u = np.append(self.res_I_u, self.sample.re_I_u)
            self.time_steps = np.append(self.time_steps, time_steps)

            # reset gradients
            optimizer.zero_grad()

            # compute gradients
            self.sample.tilted_loss.backward()

            # update parameters
            optimizer.step()

        self.stop_timer()
        self.save_som()


    def save_som(self):
        file_path = os.path.join(self.dir_path, 'som.npz')
        np.savez(
            file_path,
            iterations=self.iterations,
            thetas=self.thetas,
            losses=self.losses,
            tilted_losses=self.tilted_losses,
            grad_losses=self.grad_losses,
            means_I_u=self.means_I_u,
            vars_I_u=self.vars_I_u,
            res_I_u=self.res_I_u,
            time_steps=self.time_steps,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_som(self):
        try:
            som = np.load(
                  os.path.join(self.dir_path, 'som.npz'),
                  allow_pickle=True,
            )
            self.iterations = som['iterations']
            self.thetas = som['thetas']
            self.losses = som['losses']
            self.tilted_losses = som['tilted_losses']
            self.grad_losses = som['grad_losses']
            self.means_I_u=som['means_I_u']
            self.vars_I_u=som['vars_I_u']
            self.res_I_u=som['res_I_u']
            self.time_steps = som['time_steps']
            self.t_initial = som['t_initial']
            self.t_final = som['t_final']
            return True

        except:
            print('no som found')
            return False

    def write_report(self):
        sample = self.sample

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, 'w')

        sample.write_setting(f)
        sample.write_sampling_parameters(f)
        sample.ansatz.write_ansatz_parameters(f)

        f.write('Stochastic optimization method parameters\n')
        f.write('som type: {}\n\n'.format(self.optimizer_type))
        f.write('grad type: {}\n\n'.format(self.grad_type))

        f.write('lr: {}\n'.format(self.lr))
        f.write('iterations lim: {}\n'.format(self.iterations_lim))

        f.write('iterations used: {}\n'.format(self.iterations[-1]))
        f.write('total time steps: {:,d}\n'.format(int(self.time_steps.sum())))
        f.write('approx value function at xzero: {:2.3f}\n\n'.format(self.losses[-1]))

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

    def plot_losses(self, h_hjb, dt_mc, N_mc):
        # hjb F at xzero
        sol = self.sample.get_hjb_solver(h_hjb)
        hjb_f_at_x = sol.get_f_at_x(self.sample.xzero)
        if hjb_f_at_x is not None:
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
        colors = ['tab:blue', 'tab:green', 'tab:orange']
        linestyles = ['-', 'dashed', 'dashdot']
        labels = [
            r'$J(x_0)$',
            'hjb (h={:.0e})'.format(h_hjb),
            'MC Sampling (N={:.0e})'.format(N_mc),
        ]

        plt = Plot(self.dir_path, 'losses_line')
        plt.xlabel = 'iterations'
        #plt.set_ylim(0, 1.2 * np.max(ys))
        plt.multiple_lines_plot(self.iterations, ys, colors, linestyles, labels)

    def plot_time_steps(self):
        plt = Plot(self.dir_path, 'time_steps_line')
        plt.set_scientific_notation('y')
        plt.xlabel = 'iterations'
        plt.set_ylim(0, 1.2 * np.max(self.time_steps))
        plt.one_line_plot(self.iterations, self.time_steps, color='purple', label='TS')

        return
        plt = Plot(self.dir_path, 'time_steps_bar')
        plt.xlabel = 'iterations'
        plt.set_ylim(0, 1.2 * np.max(self.time_steps))
        plt.one_bar_plot(self.iterations, self.time_steps, color='purple', label='TS')


    def plot_1d_iteration(self, i):
        assert i in self.iterations, ''

        self.set_iterations_dir_path()
        ext = '_iter{}'.format(i)
        label = r'iteration: {}'.format(i)

        # set theta
        if self.parametrization == 'gaussian-value-f':
            self.sample.ansatz.theta = self.thetas[i]
        elif self.parametrization == 'two-layer-nn-control':
            self.sample.nn_model.load_parameters(self.thetas[i])

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

    def get_sliced_iterations(self, n_sliced_iterations=5, start=None, stop=None):
        step = self.iterations[start:stop].shape[0] // n_sliced_iterations
        sliced_iterations = self.iterations[start:stop:step]
        if self.iterations[-1] != sliced_iterations[-1]:
            sliced_iterations = np.append(sliced_iterations, self.iterations[-1])
        return sliced_iterations

    def plot_1d_iterations(self):

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        x = self.sample.domain_h[:, 0]

        # filter iterations to show
        sliced_iterations = self.get_sliced_iterations()

        # preallocate functions
        labels = []
        controlled_potentials = np.zeros((sliced_iterations.shape[0], x.shape[0]))
        frees = np.zeros((sliced_iterations.shape[0], x.shape[0]))
        controls = np.zeros((sliced_iterations.shape[0], x.shape[0]))

        for idx, i in enumerate(sliced_iterations):
            labels.append(r'iteration = {:d}'.format(i))

            # set theta
            if self.parametrization == 'gaussian-value-f':
                self.sample.ansatz.theta = self.thetas[i]
            elif self.parametrization == 'two-layer-nn-control':
                self.sample.nn_model.load_parameters(self.thetas[i])

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

    def plot_2d_iteration(self, i):
        assert i in self.iterations, ''

        self.set_iterations_dir_path()
        ext = '_iter{}'.format(i)

        # set theta
        if self.parametrization == 'gaussian-value-f':
            self.sample.ansatz.theta = self.thetas[i]
        elif self.parametrization == 'two-layer-nn-control':
            self.sample.nn_model.load_parameters(self.thetas[i])

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
