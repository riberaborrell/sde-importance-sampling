from mds.utils import get_metadynamics_dir_path, make_dir_path, empty_dir, get_time_in_hms

import time
import numpy as np
import os

class Metadynamics:
    '''
    '''

    def __init__(self, sample, N, xzero, N_lim, k, sigma_i, seed=None, do_updates_plots=False):

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
        self.ms = None
        self.thetas = None
        self.means = None
        self.sigma_i = sigma_i
        self.cov = None
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

        # plots per trajectory
        self.do_updates_plots = do_updates_plots

    def set_dir_path(self):
        self.dir_path = get_metadynamics_dir_path(
            self.sample.sde.example_dir_path,
            self.sigma_i,
            self.k,
            self.N,
        )
        #self.updates_dir_path = os.path.join(self.dir_path, 'updates')
        #make_dir_path(self.updates_dir_path)
        #empty_dir(self.updates_dir_path)

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
        self.means = np.empty((self.N, self.updates_lim, self.sample.sde.n))
        self.cov = self.sigma_i * np.eye(self.sample.sde.n)
        self.time_steps = np.empty(self.N)

        # boolean array telling us if the algorithm succeeded or not for each sample
        self.succ = np.empty(self.N, dtype=bool)

        # metadynamics algorythm for different samples
        for i in np.arange(self.N):
            self.metadynamics_per_trajectory(i)

        # stop timer
        self.stop_timer()

    def metadynamics_per_trajectory(self, i):
        '''
        '''
        # reset sampling
        sample = self.sample
        sample.is_drifted = False
        sample.xzero = np.full((sample.N, self.sample.sde.n), self.xzero)

        # preallocate means and cov matrix of the gaussiansbias functions
        means = np.empty((self.updates_lim, self.sample.sde.n))

        # set the weights of the bias functions
        #omegas = 1 * np.ones(updates)
        omegas = 0.99 * np.ones(self.updates_lim)
        omegas = np.array([w**(i+1) for i, w in enumerate(omegas)])

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(self.updates_lim):
            if self.do_updates_plots:
                N_ext = '_i_{:d}'.format(i)
                update_ext = '_j_{:d}'.format(j)
                ext = N_ext + update_ext
                if sample.is_drifted:
                    pass

            # sample with the given weights
            succ, xtemp = sample.sample_meta()

            if succ:
                self.succ[i] = succ
                # update used time stemps
                time_steps += xtemp.shape[0]
                break

            # add new bias ansatz function
            means[j] = np.mean(xtemp, axis=(0, 1))
            #print('({:2.3f}, {:2.3f})'.format(means[j, 0], means[j, 1]))

            sample.is_drifted = True
            sample.theta = omegas[:j+1] / 2
            sample.ansatz.set_given_ansatz_functions(means[:j+1], self.cov)
            sample.xzero = np.full((sample.N, self.sample.sde.n), np.mean(xtemp[-1]))

            # update used time steps
            time_steps += sample.k_lim

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.thetas[i, :j] = sample.theta
        self.means[i, :j] = sample.ansatz.means
        self.time_steps[:j] = time_steps

    def save_bias_potential(self):
        file_path = os.path.join(self.dir_path, 'bias-potential.npz')
        np.savez(
            file_path,
            ms=self.ms,
            thetas=self.thetas,
            means=self.means,
            cov=self.cov,
        )

    def write_means(self, f):
        f.write('Center of the Gaussians\n')
        f.write('i: trajectory index, j: gaussian index\n')
        for i in np.arange(self.N):
            for j in np.arange(self.ms[i]):
                mean_str = '('
                for x_i in range(self.sample.sde.n):
                    if x_i == 0:
                        mean_str += '{:2.1f}'.format(self.means[i, j, x_i])
                    else:
                        mean_str += ', {:2.1f}'.format(self.means[i, j, x_i])
                mean_str += ')'
                f.write('i={:d}, j={:d}, mu_j={}\n'.format(i, j, mean_str))
        f.write('\n')

    def write_report(self):
        sample = self.sample
        sample.N_lim = self.N_lim
        sample.xzero = np.full((sample.N, sample.sde.n), self.xzero)
        sample.xzero = self.xzero

        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, "w")

        sample.sde.write_setting(f)
        sample.write_euler_maruyama_parameters(f)
        sample.write_sampling_parameters(f)

        f.write('Metadynamics parameters and statistics\n')
        if self.seed:
            f.write('seed: {:d}\n'.format(self.seed))
        f.write('number of trajectories: {:d}\n'.format(self.N))
        f.write('k: {:d}\n'.format(self.k))
        f.write('sigma_i: {:2.2f}\n\n'.format(self.sigma_i))

        f.write('samples succeeded: {:2.2f} %\n'
                ''.format(100 * np.sum(self.succ) / self.N))
        f.write('total m: {:d}\n'.format(int(np.sum(self.ms))))
        f.write('total time steps: {:,d}\n\n'.format(int(np.sum(self.time_steps))))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        self.write_means(f)
        f.close()
