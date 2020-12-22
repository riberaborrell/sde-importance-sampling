from mds.utils import make_dir_path, empty_dir, get_time_in_hms

import time
import numpy as np
import os

class Metadynamics:
    '''
    '''

    def __init__(self, sample, N, xzero, N_lim, k, seed=None, do_updates_plots=False):

        # sampling object
        self.sample = sample

        # seed
        self.seed = seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.N = N
        self.xzero = xzero
        self.N_lim = N_lim
        self.k = k
        self.updates_lim = self.N_lim // self.k

        # metadynamics coefficients
        self.m = None
        self.theta = None
        self.mus = None
        self.sigmas = None
        self.time_steps = None

        # succeeded
        self.succ = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # set path
        self.dir_path = None
        self.updates_dir_path = None
        self.set_dir_path()

        # plots after each update
        self.do_updates_plots = do_updates_plots

    def set_dir_path(self):
        self.dir_path = os.path.join(self.sample.example_dir_path, 'metadynamics')
        self.updates_dir_path = os.path.join(self.dir_path, 'updates')
        make_dir_path(self.updates_dir_path)
        empty_dir(self.updates_dir_path)

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def metadynamics_algorithm(self):
        # start timer
        self.start_timer()

        # initialize bias potentials coefficients
        self.ms = np.empty(self.N, dtype=np.intc)
        self.thetas = np.empty((self.N, self.updates_lim))
        self.thetas[:, :] = np.nan
        self.mus = np.empty((self.N, self.updates_lim))
        self.mus[:, :] = np.nan
        self.sigmas = np.empty((self.N, self.updates_lim))
        self.sigmas[:, :] = np.nan
        self.time_steps = np.empty(self.N)

        # boolean array telling us if the algorithm succeeded or not for each sample
        self.succ = np.empty(self.N, dtype=bool)

        # metadynamics algorythm for different samples
        for i in np.arange(self.N):
            self.metadynamics_trajectory(i)

        # stop timer
        self.stop_timer()

    def metadynamics_trajectory(self, i):
        '''
        '''
        # reset sampling
        sample = self.sample
        sample.is_drifted = False
        sample.xzero = np.full(sample.M, self.xzero)

        #preallocate means and standard deviation
        mus = np.empty(self.updates_lim)
        sigmas = np.empty(self.updates_lim)

        # set the weights of the bias functions
        #omegas = 1 * np.ones(updates)
        omegas = 0.99 * np.ones(self.updates_lim)
        omegas = np.array([w**(i+1) for i, w in enumerate(omegas)])

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(self.updates_lim):
            # plot after update
            if self.do_updates_plots:
                N_ext = '_i_{:d}'.format(i)
                update_ext = '_j_{:d}'.format(j)
                ext = N_ext + update_ext
                sample.plot_tilted_potential('tilted_potential' + ext, self.updates_dir_path)
                #sample.plot_tilted_drift('tilted_drift' + ext, self.updates_dir_path)

            # sample with the given weights
            succ, xtemp = sample.sample_meta()

            if succ:
                self.succ[i] = succ
                # update used time steps
                time_steps += xtemp.shape[0]
                break

            # add new bias ansatz function
            #print('{:2.2f}, {:2.3f}, {:2.3f}'.format(np.mean(xtemp), np.std(xtemp), np.var(xtemp)))
            mus[j] = np.mean(xtemp)
            sigmas[j] = 5 * np.std(xtemp)

            # update ansatz
            sample.is_drifted = True
            sample.theta = omegas[:j+1] / 2
            sample.ansatz.mus = mus[:j+1]
            sample.ansatz.sigmas = sigmas[:j+1]
            sample.xzero = np.mean(xtemp[-1])

            # update used time steps
            time_steps += sample.N_lim

        if not succ:
            self.succ[i] = succ

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.thetas[i, :j] = sample.theta
        self.mus[i, :j] = sample.ansatz.mus
        self.sigmas[i, :j] = sample.ansatz.sigmas
        self.time_steps[:j] = time_steps
        return j

    def save_bias_potential(self):
        file_path = os.path.join(self.dir_path, 'bias_potential.npz')
        np.savez(
            file_path,
            ms=self.ms,
            thetas=self.thetas,
            mus=self.mus,
            sigmas=self.sigmas,
        )

    def write_report(self):
        sample = self.sample
        sample.N_lim = self.N_lim
        sample.xzero = self.xzero

        k_steps_ext = '_k{:d}'.format(self.k)
        file_name = 'report' + k_steps_ext + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        sample.write_sde_parameters(f)
        sample.write_euler_maruyama_parameters(f)
        sample.write_sampling_parameters(f)

        f.write('Metadynamics parameters and statistics\n')
        if self.seed:
            f.write('seed: {:d}\n'.format(self.seed))
        f.write('N: {:d}\n'.format(self.N))
        f.write('k: {:d}\n\n'.format(self.k))

        f.write('samples succeeded: {:2.2f} %\n'
                ''.format(100 * np.sum(self.succ) / self.N))
        f.write('m: {:d}\n'.format(int(np.sum(self.ms))))
        f.write('used time steps: {:,d}\n\n'.format(int(np.sum(self.time_steps))))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))
        f.close()

    def get_updates_to_show(self, num_updates, num_updates_to_show=5):
        k = num_updates // num_updates_to_show
        updates = np.arange(num_updates)
        updates_to_show = np.where(updates % k == 0)[0]
        if updates[-1] != updates_to_show[-1]:
            updates_to_show = np.append(updates_to_show, updates[-1])
        return updates_to_show

    def plot_meta_1d_tilted_potentials(self):
        sample = self.sample
        x = sample.domain_h

        updates_to_show = self.get_updates_to_show()
        V = sample.potential(x)
        Vbias = np.zeros((updates_to_show.shape[0], x.shape[0]))
        for i, update in enumerate(updates_to_show):
            label = r'update = {:d}'.format(update)
            sample.theta = self.thetas[update, :]
            Vbias[i, :] = 2 * sample.value_function(x, sample.theta)

        sample.load_reference_solution()
        F = sample.ref_sol['F']
        Vbias_opt = 2 * F

        plt1d = Plot1d(self.dir_path, 'gd_tilted_potentials')
        plt1d.set_xlim(-2.5, 2.5)
        #plt1d.set_ylim(0, sample.alpha * 10)
        #plt1d.set_ylim(-0.25, 10)
        #plt1d.set_ylim(-0.25, 40)
        plt1d.set_ylim(-0.25, 100)
        plt1d.gd_tilted_potentials(x, V, epochs_to_show, Vbias, Vbias_opt)


