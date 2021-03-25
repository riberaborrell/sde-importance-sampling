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
            self.sample.settings_dir_path,
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
        self.thetas = np.empty(0)
        self.means = np.empty((0, self.sample.n))
        self.cov = self.sigma_i * np.eye(self.sample.n)
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
        sample.is_controlled = False
        sample.xzero = np.full((sample.N, self.sample.n), self.xzero)

        # preallocate means and cov matrix of the gaussiansbias functions
        means = np.empty((0, self.sample.n))

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
            means = np.vstack((means, np.mean(xtemp, axis=(0, 1))))
            #print('({:2.3f}, {:2.3f})'.format(means[j, 0], means[j, 1]))

            sample.is_controlled = True
            sample.ansatz.set_given_ansatz_functions(means, self.cov)
            sample.ansatz.theta = omegas[:j+1] / 2
            sample.xzero = np.full((sample.N, self.sample.n), np.mean(xtemp[-1]))

            # update used time steps
            time_steps += sample.k_lim

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.thetas = np.append(self.thetas, sample.ansatz.theta)
        self.means = np.vstack((self.means, sample.ansatz.means))
        self.time_steps[:j] = time_steps

    def save_bias_potential(self):
        np.savez(
            os.path.join(self.dir_path, 'bias-potential.npz'),
            succ=self.succ,
            ms=self.ms,
            thetas=self.thetas,
            means=self.means,
            cov=self.cov,
            time_steps=self.time_steps,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_bias_potential(self):
        try:
            bias_pot = np.load(
                os.path.join(self.dir_path, 'bias-potential.npz'),
                allow_pickle=True,
            )
            self.succ = bias_pot['succ']
            self.ms = bias_pot['ms']
            self.thetas = bias_pot['thetas']
            self.means = bias_pot['means']
            self.cov = bias_pot['cov']
            self.time_steps = bias_pot['time_steps']
            self.t_initial = bias_pot['t_initial']
            self.t_final = bias_pot['t_final']
            return True

        except:
            print('no bias potential found')
            return False

    def write_means(self, f):
        f.write('Center of the Gaussians\n')
        f.write('i: trajectory index, j: gaussian index\n')
        for i in np.arange(self.N):
            idx_i = slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + self.ms[i])
            means_i = self.means[idx_i]
            for j in np.arange(self.ms[i]):
                mean_str = '('
                for x_i in range(self.sample.n):
                    if x_i == 0:
                        mean_str += '{:2.1f}'.format(means_i[j, x_i])
                    else:
                        mean_str += ', {:2.1f}'.format(means_i[j, x_i])
                mean_str += ')'
                f.write('i={:d}, j={:d}, mu_j={}\n'.format(i, j, mean_str))
        f.write('\n')

    def write_report(self):
        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write in file
        f = open(file_path, 'w')

        self.sample.N_lim = self.N_lim
        #self.sample.xzero = np.full((self.sample.N, self.sample.n), self.xzero)
        self.sample.xzero = self.xzero

        self.sample.write_setting(f)
        self.sample.write_euler_maruyama_parameters(f)
        self.sample.write_sampling_parameters(f)

        f.write('Metadynamics parameters and statistics\n')
        f.write('seed: {:d}\n'.format(self.seed))
        f.write('sigma_i_meta: {:2.2f}\n'.format(self.sigma_i))
        f.write('k: {:d}\n'.format(self.k))
        f.write('N_meta: {:d}\n\n'.format(self.N))

        f.write('traj succeeded: {:2.2f} %\n'
                ''.format(100 * np.sum(self.succ) / self.N))
        f.write('total m: {:d}\n'.format(int(np.sum(self.ms))))
        f.write('total time steps: {:,d}\n\n'.format(int(np.sum(self.time_steps))))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        self.write_means(f)
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def get_updates_to_show(self, i=0, num_updates_to_show=5):
        num_updates = self.ms[i]
        k = num_updates // num_updates_to_show
        updates = np.arange(num_updates)
        updates_to_show = np.where(updates % k == 0)[0]
        if updates[-1] != updates_to_show[-1]:
            updates_to_show = np.append(updates_to_show, updates[-1])
        return updates_to_show

    def plot_1d_updates(self, i=0):
        # get sampling object
        sample = self.sample

        # discretize domain and evaluate in grid
        sample.discretize_domain(h=0.001)
        x = sample.domain_h[:, 0]

        # filter updates to show
        updates_to_show = self.get_updates_to_show()

        # preallocate functions
        labels = []
        frees = np.zeros((updates_to_show.shape[0] + 1, x.shape[0]))
        controls = np.zeros((updates_to_show.shape[0] + 1, x.shape[0]))
        controlled_potentials = np.zeros((updates_to_show.shape[0] + 1, x.shape[0]))

        # not controlled case
        labels.append(r'not controlled')
        sample.is_controlled = False
        sample.get_grid_value_function_and_control()
        controlled_potentials[0, :] = sample.grid_controlled_potential
        frees[0, :] = sample.grid_value_function
        controls[0, :] = sample.grid_control[:, 0]

        sample.is_controlled = True
        idx_i = slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + self.ms[i])
        means = self.means[idx_i]
        thetas = self.thetas[idx_i]
        for index, update in enumerate(updates_to_show):
            labels.append(r'update = {:d}'.format(update+1))

            # set theta
            sample.ansatz.set_given_ansatz_functions(
                means[:update+1],
                self.cov,
            )
            sample.ansatz.theta = thetas[:update+1]
            sample.get_grid_value_function_and_control()

            # update functions
            controlled_potentials[index+1, :] = sample.grid_controlled_potential
            frees[index+1, :] = sample.grid_value_function
            controls[index+1, :] = sample.grid_control[:, 0]

        # get hjb solution
        sol = sample.get_hjb_solver(h=0.001)
        sol.get_controlled_potential_and_drift()

        # file extension
        ext = '_i_{}'.format(i)

        sample.plot_1d_free_energies(frees, F_hjb=sol.F, labels=labels[:],
                                     dir_path=self.dir_path, ext=ext)
        sample.plot_1d_controls(controls, u_hjb=sol.u_opt[:, 0], labels=labels[:],
                                dir_path=self.dir_path, ext=ext)
        sample.plot_1d_controlled_potentials(controlled_potentials,
                                             controlledV_hjb=sol.controlled_potential,
                                             labels=labels[:], dir_path=self.dir_path,
                                             ext=ext)


