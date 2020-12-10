from mds.plots_1d import Plot1d
from mds.utils import get_gd_data_path, make_dir_path, get_time_in_hms

import time
import numpy as np
import os

class GradientDescent:
    '''
    '''
    def __init__(self, sample, grad_type, theta_init, lr, epochs_lim,
                 do_epoch_plots=False, atol=0.01, rtol=0.01):
        '''
        '''
        self.sample = sample
        self.theta_init = theta_init

        self.grad_type = grad_type

        # finite differences
        #self.delta = delta if delta is not None else None

        self.lr = lr
        self.epochs_lim = epochs_lim
        self.epochs = None
        self.atol = atol
        self.rtol = rtol

        self.thetas = None
        #self.thetas_dif = None
        self.losses = None
        self.grad_losses = None
        self.N = None

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
        ansatz_dir_path = self.sample.ansatz.dir_path
        grad_type = self.grad_type
        theta_init = self.theta_init
        lr = self.lr
        self.dir_path = get_gd_data_path(ansatz_dir_path, grad_type, theta_init, lr)

    def set_epochs_dir_path(self):
        self.epochs_dir_path = os.path.join(self.dir_path, 'epochs')
        make_dir_path(self.epochs_dir_path)

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    #TODO: test method. test grad_type: fd
    def perturb_a(self, a, j, sign, delta=None):
        # assert j in np.arange(self.sample.m)
        # assert sign = 1 or sign = -1
        # assert delta > 0
        if delta is None:
            delta = self.delta
        perturbation = np.zeros(self.sample.m)
        perturbation[j] = sign * delta

        return a + perturbation

    #TODO: test
    def gd_fd(self):
        sample = self.sample
        lr = self.lr
        M = self.sample.M

        # finite differences
        if self.grad_type == 'fd':
            sample.sample_soc()
            sample.compute_statistics()
            loss = sample.mean_J
            grad_loss = np.zeros(sample.m)

            for j in np.arange(self.sample.m):
                a_pert_j_p = self.perturb_a(a, j, 1)
                sample.a = a_pert_j_p
                sample.sample_soc()
                sample.compute_statistics()
                mean_J_pert_j_p = sample.mean_J

                a_pert_j_m = self.perturb_a(a, j, -1)
                sample.a = a_pert_j_m
                sample.sample_soc()
                sample.compute_statistics()
                mean_J_pert_j_m = sample.mean_J

                grad_loss[j] = (mean_J_pert_j_p - mean_J_pert_j_m) \
                             / (2 * self.delta)

        # Update parameters
        a_updated = a - lr * grad_loss

        # Returns the parameters, the loss and the gradient
        return a_updated, loss, grad_loss

    def gd_ipa(self):
        self.start_timer()

        sample = self.sample
        x = self.sample.domain_h
        m = sample.ansatz.m

        # coefficients, performance function, control and free energy
        thetas = np.zeros((self.epochs_lim + 1, m))
        losses = np.zeros(self.epochs_lim)
        #grad_losses = np.zeros((self.epochs_lim + 1, m))
        N = np.zeros(self.epochs_lim)

        # set initial coefficients
        thetas[0, :] = sample.theta

        for epoch in np.arange(self.epochs_lim):
            # plot control, free_energy and tilted potential
            if self.do_epoch_plots:
                epochs_dir_path = self.epochs_dir_path
                epoch_stamp = '_epoch{}'.format(epoch)
                sample.plot_free_energy('appr_free_energy' + epoch_stamp, epochs_dir_path)
                sample.plot_control('control' + epoch_stamp, epochs_dir_path)
                sample.plot_tilted_potential('tilted_potential' + epoch_stamp, epochs_dir_path)

            # get loss and its gradient 
            sample_succ, \
            losses[epoch], \
            grad_losses, \
            N[epoch] = sample.sample_loss()

            # check if sample succeeded
            if not sample_succ:
                break

            print('{:d}, {:2.3f}'.format(epoch, losses[epoch]))

            # update coefficients
            #thetas[epoch + 1, :] = thetas[epoch, :] - self.lr * grad_losses[epoch, :]
            thetas[epoch + 1, :] = thetas[epoch, :] - self.lr * grad_losses
            sample.theta = thetas[epoch + 1, :]


        # save thetas, losses and grad_losses
        if sample_succ:
            self.epochs = epoch + 1
            self.thetas = thetas
            self.losses = losses
            #self.grad_losses = grad_losses
            self.N = N
        else:
            self.epochs = epoch
            self.thetas = thetas[:epoch]
            self.losses = losses[:epoch]
            #self.grad_losses = grad_losses[:epoch]
            self.N = N[:epoch]

        # compute eucl dist between thetas and thetas opt
        #self.a_dif = np.linalg.norm(self.a_s - self.sample.a_opt, axis=1)

        self.stop_timer()

    def save_gd(self):
        file_path = os.path.join(self.dir_path, 'gd.npz')
        np.savez(
            file_path,
            epochs=self.epochs,
            thetas=self.thetas,
            losses=self.losses,
            #grad_losses=self.grad_losses,
            N=self.N,
        )

    def load_gd(self):
        file_path = os.path.join(self.dir_path, 'gd.npz')
        gd = np.load(file_path, allow_pickle=True)
        self.epochs = gd['epochs']
        self.thetas = gd['thetas']
        self.losses = gd['losses']
        #self.grad_losses = gd['grad_losses']
        self.N = gd['N']

    def write_report(self):
        sample = self.sample

        # set path
        trajectories_stamp = 'M{:.0e}'.format(sample.M)
        file_name = 'report_' + trajectories_stamp + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        sample.write_sde_parameters(f)
        sample.write_sampling_parameters(f)
        sample.ansatz.write_ansatz_parameters(f)

        f.write('GD parameters\n')
        f.write('grad type: {}\n\n'.format(self.grad_type))

        f.write('lr: {}\n'.format(self.lr))
        f.write('epochs lim: {}\n'.format(self.epochs_lim))
        #f.write('atol: {}\n'.format(self.atol))
        #f.write('rtol: {}\n\n'.format(self.rtol))

        f.write('epochs needed: {}\n'.format(self.epochs))
        f.write('N total: {:,d}\n'.format(int(self.N.sum())))
        f.write('approx value function at xzero: {:2.3f}\n\n'.format(self.losses[-1]))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

    #TODO: 
    def write_epoch_report(self):
        for epoch in np.arange(self.epochs):
            f.write('epoch = {:d}\n'.format(epoch))
            f.write('theta = {}\n'.format(self.thetas[epoch]))
            f.write('|theta_dif|_2 = {:2.4e}\n'.format(self.theta_dif[epoch]))
            f.write('loss = {:2.4e}\n'.format(self.losses[epoch]))
            f.write('grad_loss = {}\n'.format(self.grad_losses[epoch]))
            f.write('|grad_loss|_2 = {:2.4e}\n\n'
                    ''.format(np.linalg.norm(self.grad_losses[epoch])))
            f.write('N = {}\n'.format(self.N[epoch]))

            #f.write('epoch = {:d}\n'.format(epoch + 1))
            #f.write('a = {}\n'.format(self.a_s[epoch + 1]))
            #f.write('|a_dif|_2 = {:2.4e}\n'.format(self.a_dif[epoch + 1]))

            f.close()

    def get_epochs_to_show(self, num_epochs_to_show=5):
        num_epochs = self.epochs
        k = num_epochs // num_epochs_to_show
        epochs = np.arange(num_epochs)
        epochs_to_show = np.where(epochs % k == 0)[0]
        if epochs[-1] != epochs_to_show[-1]:
            epochs_to_show = np.append(epochs_to_show, epochs[-1])
        return epochs_to_show

    def plot_gd_1d_free_energies(self):
        sample = self.sample
        x = sample.domain_h
        epochs_to_show = self.get_epochs_to_show()

        labels = []
        F = np.zeros((epochs_to_show.shape[0], x.shape[0]))
        for i, epoch in enumerate(epochs_to_show):
            labels.append(r'epoch = {:d}'.format(epoch))
            sample.theta = self.thetas[epoch, :]
            F[i, :] = sample.value_function(x, sample.theta)

        sample.load_reference_solution()
        F_hjb = sample.ref_sol['F']
        labels.append(r'hjb')

        ys = np.vstack((F, F_hjb))
        colors = [None for i in range(ys.shape[0])]
        colors[-1] = 'c'
        linestyles = ['-' for i in range(ys.shape[0])]
        linestyles[-1] = 'dashdot'

        plt1d = Plot1d(self.dir_path, 'gd_free_energy')
        #plt1d.set_ylim(0, sample.alpha * 3)
        plt1d.set_ylim(-0.25, 3)
        #plt1d.set_ylim(-0.25, 4)
        #plt1d.set_ylim(-0.25, 10)
        #plt1d.gd_appr_free_energies(x, epochs_to_show, F_appr, F)
        plt1d.multiple_lines_plot(x, ys, colors, linestyles, labels)

    def plot_gd_1d_controls(self):
        sample = self.sample
        x = sample.domain_h
        epochs_to_show = self.get_epochs_to_show()

        labels = []
        u = np.zeros((epochs_to_show.shape[0], x.shape[0]))
        for i, epoch in enumerate(epochs_to_show):
            labels.append(r'epoch = {:d}'.format(epoch))
            sample.theta = self.thetas[epoch, :]
            u[i, :] = sample.control(x, sample.theta)

        sample.load_reference_solution()
        u_hjb = sample.ref_sol['u_opt']
        labels.append(r'hjb')

        ys = np.vstack((u, u_hjb))
        colors = [None for i in range(ys.shape[0])]
        colors[-1] = 'c'
        linestyles = ['-' for i in range(ys.shape[0])]
        linestyles[-1] = 'dashdot'

        plt1d = Plot1d(self.dir_path, 'gd_controls')
        plt1d.set_xlim(-2.5, 2.5)
        #plt1d.set_ylim(-sample.alpha * 5, sample.alpha * 5)
        plt1d.set_ylim(-5, 5)
        #plt1d.set_ylim(-10, 15)
        #plt1d.set_ylim(-15, 25)
        plt1d.multiple_lines_plot(x, ys, colors, linestyles, labels)

    def plot_gd_1d_tilted_potentials(self):
        sample = self.sample
        x = sample.domain_h
        epochs_to_show = self.get_epochs_to_show()

        labels = []
        Vb = np.zeros((epochs_to_show.shape[0], x.shape[0]))
        for i, epoch in enumerate(epochs_to_show):
            labels.append(r'epoch = {:d}'.format(epoch))
            sample.theta = self.thetas[epoch, :]
            Vb[i, :] = 2 * sample.value_function(x, sample.theta)

        sample.load_reference_solution()
        Vb_hjb = 2 * sample.ref_sol['F']
        labels.append(r'hjb')

        V = sample.potential(x)
        ys = np.vstack((V + Vb, V + Vb_hjb))
        colors = [None for i in range(ys.shape[0])]
        colors[-1] = 'c'
        linestyles = ['-' for i in range(ys.shape[0])]
        linestyles[-1] = 'dashdot'

        plt1d = Plot1d(self.dir_path, 'gd_tilted_potentials')
        plt1d.set_xlim(-2.5, 2.5)
        #plt1d.set_ylim(0, sample.alpha * 10)
        plt1d.set_ylim(-0.25, 10)
        #plt1d.set_ylim(-0.25, 40)
        #plt1d.set_ylim(-0.25, 100)
        plt1d.multiple_lines_plot(x, ys, colors, linestyles, labels)

    def plot_gd_losses(self):
        sample = self.sample
        losses = self.losses
        epochs = np.arange(losses.shape[0])
        sample.get_value_f_at_xzero()
        sample.load_not_drifted(M=10000000)
        value_f_hjb = np.full(epochs.shape[0], sample.value_f_at_xzero)
        value_f_mc = np.full(epochs.shape[0], - np.log(sample.mean_I))

        ys = np.vstack((losses, value_f_hjb, value_f_mc))
        colors = ['tab:blue', 'tab:green', 'tab:orange']
        linestyles = ['-', 'dashed', 'dashdot']
        labels = [r'$J(x_0)$', 'hjb', 'MC Sampling']

        plt1d = Plot1d(self.dir_path, 'gd_losses_line')
        plt1d.xlabel = 'epochs'
        plt1d.set_ylim(0, 1.2 * np.max(ys))
        plt1d.multiple_lines_plot(epochs, ys, colors, linestyles, labels)

    def plot_gd_time_steps(self):
        N = self.N
        epochs = np.arange(N.shape[0])

        plt1d = Plot1d(self.dir_path, 'gd_time_steps_bar')
        plt1d.xlabel = 'epochs'
        plt1d.set_ylim(0, 1.2 * np.max(N))
        plt1d.one_bar_plot(epochs, N, color='purple', label='TS')

        plt1d = Plot1d(self.dir_path, 'gd_time_steps_line')
        plt1d.xlabel = 'epochs'
        plt1d.set_ylim(0, 1.2 * np.max(N))
        plt1d.one_line_plot(epochs, N, color='purple', label='TS')

