from mds.plots_1d import Plot1d
from mds.utils import get_gd_dir_path, make_dir_path, get_time_in_hms

import time
import numpy as np
import os

class GradientDescent:
    '''
    '''
    def __init__(self, sample, grad_type, lr, epochs_lim,
                 do_epoch_plots=False):
        '''
        '''
        self.sample = sample

        self.grad_type = grad_type

        self.lr = lr
        self.epochs_lim = epochs_lim
        self.epochs = None

        self.thetas = None
        self.losses = None
        self.grad_losses = None
        self.time_steps = None

        # computational time
        self.t_initial = None
        self.t_final = None

        self.do_epoch_plots = do_epoch_plots

        # set path
        self.dir_path = None
        self.set_dir_path()
        if do_epoch_plots:
            self.set_epochs_dir_path()
        else:
            self.epochs_dir_path = None

    def set_dir_path(self):
        self.dir_path = get_gd_dir_path(
            self.sample.ansatz.dir_path,
            self.grad_type,
            self.lr,
            self.sample.N,
        )

    def set_epochs_dir_path(self):
        self.epochs_dir_path = os.path.join(self.dir_path, 'epochs')
        make_dir_path(self.epochs_dir_path)

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def gd_ipa(self):
        self.start_timer()

        sample = self.sample
        m = sample.ansatz.m

        # initialize coefficients and losses
        self.epochs = np.empty(0, dtype=int)
        self.thetas = np.empty((0, m))
        self.losses = np.empty(0)
        self.grad_losses = np.empty((0, m))
        self.time_steps = np.empty(0)

        for epoch in np.arange(self.epochs_lim):
            # plot control, free_energy and tilted potential
            if self.do_epoch_plots:
                pass

            # compute loss and its gradient 
            sample_succ, loss, grad_loss, time_steps = sample.sample_loss()
            print('{:d}, {:2.3f}'.format(epoch, loss))

            # check if sample succeeded
            if not sample_succ:
                break

            # allocate
            self.epochs = np.append(self.epochs, epoch)
            self.thetas = np.vstack((self.thetas, sample.ansatz.theta))
            self.losses = np.append(self.losses, loss)
            self.grad_losses = np.vstack((self.grad_losses, grad_loss))
            self.time_steps = np.append(self.time_steps, time_steps)

            # update coefficients
            sample.ansatz.theta = self.thetas[epoch, :] - self.lr * self.grad_losses[epoch, :]

        self.stop_timer()
        self.save_gd()

    def save_gd(self):
        file_path = os.path.join(self.dir_path, 'gd.npz')
        np.savez(
            file_path,
            epochs=self.epochs,
            thetas=self.thetas,
            losses=self.losses,
            grad_losses=self.grad_losses,
            time_steps=self.time_steps,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_gd(self):
        file_path = os.path.join(self.dir_path, 'gd.npz')
        gd = np.load(file_path, allow_pickle=True)
        self.epochs = gd['epochs']
        self.thetas = gd['thetas']
        self.losses = gd['losses']
        self.grad_losses = gd['grad_losses']
        self.time_steps = gd['time_steps']
        self.t_initial = gd['t_initial']
        self.t_final = gd['t_final']

    def write_report(self):
        sample = self.sample

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, "w")

        sample.write_setting(f)
        sample.write_sampling_parameters(f)
        sample.ansatz.write_ansatz_parameters(f)

        f.write('GD parameters\n')
        f.write('grad type: {}\n\n'.format(self.grad_type))

        f.write('lr: {}\n'.format(self.lr))
        f.write('epochs lim: {}\n'.format(self.epochs_lim))

        f.write('epochs used: {}\n'.format(self.epochs[-1]))
        f.write('total time steps: {:,d}\n'.format(int(self.time_steps.sum())))
        f.write('approx value function at xzero: {:2.3f}\n\n'.format(self.losses[-1]))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        #self.write_epoch_report(f)
        f.close()

    def write_epoch_report(self, f):
        for epoch in self.epochs:
            f.write('epoch = {:d}\n'.format(epoch))
            f.write('theta = {}\n'.format(self.thetas[epoch]))
            f.write('loss = {:2.4e}\n'.format(self.losses[epoch]))
            f.write('grad_loss = {}\n'.format(self.grad_losses[epoch]))
            f.write('|grad_loss|_2 = {:2.4e}\n\n'
                    ''.format(np.linalg.norm(self.grad_losses[epoch])))
            f.write('time steps = {}\n'.format(self.time_steps[epoch]))

    def plot_gd_losses(self, h_hjb, N_mc):
        # hjb F at xzero
        sol = self.sample.get_hjb_solver(h_hjb)
        hjb_f_at_x = sol.get_f_at_x(self.sample.xzero)
        if hjb_f_at_x is not None:
            value_f_hjb = np.full(self.epochs.shape[0], hjb_f_at_x)
        else:
            value_f_hjb = np.full(self.epochs.shape[0], np.nan)

        # mc F
        mcs = self.sample.get_not_controlled(N_mc)
        if mcs is not None:
            mc_psi = mcs['mean_I']
            mc_f = - np.log(mc_psi)
            value_f_mc = np.full(self.epochs.shape[0], mc_f)
        else:
            value_f_mc = np.full(self.epochs.shape[0], np.nan)

        ys = np.vstack((self.losses, value_f_hjb, value_f_mc))
        colors = ['tab:blue', 'tab:green', 'tab:orange']
        linestyles = ['-', 'dashed', 'dashdot']
        labels = [
            r'$J(x_0)$',
            'hjb (h={:.0e})'.format(h_hjb),
            'MC Sampling (N={:.0e})'.format(N_mc),
        ]

        plt1d = Plot1d(self.dir_path, 'gd_losses_line')
        plt1d.xlabel = 'epochs'
        #plt1d.set_ylim(0, 1.2 * np.max(ys))
        plt1d.multiple_lines_plot(self.epochs, ys, colors, linestyles, labels)

    def plot_gd_time_steps(self):
        plt1d = Plot1d(self.dir_path, 'gd_time_steps_bar')
        plt1d.xlabel = 'epochs'
        plt1d.set_ylim(0, 1.2 * np.max(self.time_steps))
        plt1d.one_bar_plot(self.epochs, self.time_steps, color='purple', label='TS')

        plt1d = Plot1d(self.dir_path, 'gd_time_steps_line')
        plt1d.xlabel = 'epochs'
        plt1d.set_ylim(0, 1.2 * np.max(self.time_steps))
        plt1d.one_line_plot(self.epochs, self.time_steps, color='purple', label='TS')
