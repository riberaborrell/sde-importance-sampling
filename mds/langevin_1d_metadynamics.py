from mds.utils import make_dir_path, empty_dir, get_time_in_hms

import time
import numpy as np
import os

class Metadynamics:
    '''
    '''

    def __init__(self, sample, num_samples, xzero, N_lim, k, seed=None, do_updates_plots=False):

        # sampling object
        self.sample = sample

        # seed
        self.seed = seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.num_samples = num_samples
        self.xzero = xzero
        self.N_lim = N_lim
        self.k = k

        # metadynamics coefficients
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
        self.theta = np.empty(0)
        self.mus = np.empty(0)
        self.sigmas = np.empty(0)
        self.time_steps = 0

        # boolean array telling us if the algorithm succeeded or not for each sample
        self.succ = np.empty(self.num_samples, dtype=bool)

        # metadynamics algorythm for different samples
        for i in np.arange(self.num_samples):
            self.metadynamics_per_sample(i)

        # normalize
        self.theta /= self.num_samples

        # stop timer
        self.stop_timer()

    def metadynamics_per_sample(self, i):
        '''
        '''
        # reset sampling
        sample = self.sample
        sample.is_drifted = False
        sample.xzero = np.full(sample.M, self.xzero)

        # maximal number of updates
        updates = self.N_lim // sample.N_lim

        # set the weights of the bias functions
        #omegas = 1 * np.ones(updates)
        omegas = 0.99 * np.ones(updates)
        omegas = np.array([w**(i+1) for i, w in enumerate(omegas)])

        # preallocate means and standard deviation of the gaussians bias functions
        mus = np.empty(updates)
        sigmas = np.empty(updates)

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(updates):
            # plot after update
            if self.do_updates_plots:
                sample_stamp = '_i_{:d}'.format(i)
                bias_stamp = '_j_{:d}'.format(j)
                stamp = sample_stamp + bias_stamp
                sample.plot_tilted_potential('tilted_potential' + stamp, self.updates_dir_path)
                #sample.plot_tilted_drift('tilted_drift' + stamp, self.updates_dir_path)

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
        self.theta = np.concatenate((self.theta, sample.theta))
        self.mus = np.concatenate((self.mus, sample.ansatz.mus))
        self.sigmas = np.concatenate((self.sigmas, sample.ansatz.sigmas))
        self.time_steps += time_steps

    def save_bias_potential(self):
        file_path = os.path.join(self.dir_path, 'bias_potential.npz')
        np.savez(
            file_path,
            theta=self.theta,
            mus=self.mus,
            sigmas=self.sigmas,
        )

    def write_report(self):
        sample = self.sample
        sample.N_lim = self.N_lim
        sample.xzero = self.xzero

        k_steps_stamp = '_k{:d}'.format(self.k)
        file_name = 'report' + k_steps_stamp + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        sample.write_sde_parameters(f)
        sample.write_euler_maruyama_parameters(f)
        sample.write_sampling_parameters(f)

        f.write('Metadynamics parameters and statistics\n')
        if self.seed:
            f.write('seed: {:d}\n'.format(self.seed))
        f.write('number of samples: {:d}\n'.format(self.num_samples))
        f.write('k: {:d}\n\n'.format(self.k))

        f.write('samples succeeded: {:2.2f} %\n'
                ''.format(100 * np.sum(self.succ) / self.num_samples))
        f.write('m: {:d}\n'.format(self.theta.shape[0]))
        f.write('used time steps: {:,d}\n\n'.format(self.time_steps))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))
        f.close()