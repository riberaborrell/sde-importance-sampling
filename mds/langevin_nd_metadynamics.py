from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils import get_metadynamics_dir_path, make_dir_path, empty_dir, get_time_in_hms
from mds.numeric_utils import slice_1d_array

import time
import numpy as np
import os

class Metadynamics:
    '''
    '''

    def __init__(self, sample, k, N, sigma_i, seed=None, do_updates_plots=False):

        # sampling object
        self.sample = sample

        # seed
        self.seed = seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.k_lim = None
        self.k = k
        self.updates_lim = None
        self.N = N
        self.xzero = None

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

        # plots per trajectory
        self.do_updates_plots = do_updates_plots

    def set_dir_path(self):
        self.dir_path = get_metadynamics_dir_path(
            self.sample.settings_dir_path,
            self.sample.dt,
            self.sigma_i,
            self.k,
            self.N,
        )

    def set_updates_dir_path(self):
        self.updates_dir_path = os.path.join(self.dir_path, 'updates')
        make_dir_path(self.updates_dir_path)
        empty_dir(self.updates_dir_path)

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def set_sampling_parameters(self, k_lim, dt, xzero):
        '''set k-steps sampling and Euler-Marujama parameters
        '''
        # limit number of time steps and limit number of updates
        self.k_lim = k_lim
        self.updates_lim = k_lim // self.k

        # initial position
        self.xzero = xzero

        # sampling parameters
        self.sample.set_sampling_parameters(
            dt=dt,
            k_lim=self.k,
            xzero=xzero,
            N=1,
        )

    def metadynamics_algorithm(self):
        # start timer
        self.start_timer()

        # initialize bias potentials coefficients
        self.ms = np.empty(self.N, dtype=np.intc)
        self.thetas = np.empty(0)
        self.means = np.empty((0, self.sample.n))
        self.cov = self.sigma_i * np.eye(self.sample.n)
        self.time_steps = np.empty(self.N, dtype=np.int32)

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
        self.time_steps[i] = time_steps

    def get_trajectory_indices(self, i):
        assert i in range(self.N), ''

        return slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + self.ms[i])

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
            msg = 'no meta bias potential found with dt={:.4f}, sigma_i={:.2f}, k={:d}, ' \
                  'N={:.0e}'.format(self.sample.dt, self.sigma_i, self.k, self.N)
            print(msg)
            return False

    def set_ansatz_single_trajectory(self, i, update):

        # get means and weights of i trajectory
        idx_i = self.get_trajectory_indices(i)
        means_i = self.means[idx_i]
        thetas_i = self.thetas[idx_i]

        # set ansatz and theta
        self.sample.ansatz.set_given_ansatz_functions(
            means_i[:update+1],
            self.cov,
        )
        self.sample.ansatz.theta = thetas_i[:update+1]
        self.sample.ansatz.set_value_function_constant_corner()

    def set_ansatz_all_trajectories(self):

        self.sample.ansatz.set_given_ansatz_functions(
            means=self.means,
            cov=self.cov,
        )

        m = np.sum(self.ms)
        thetas = np.empty((self.N, m))

        for i in np.arange(self.N):
            # get means and thetas for each trajectory
            idx_i = self.get_trajectory_indices(i)
            meta_means_i = self.means[idx_i]
            meta_thetas_i = self.thetas[idx_i]

            # create ansatz functions corresponding to the ith metadynamics trajectory
            meta_ansatz_i = GaussianAnsatz(n=self.sample.n)
            meta_ansatz_i.set_given_ansatz_functions(
                means=meta_means_i,
                cov=self.cov,
            )

            # meta value function evaluated at the grid
            x = self.means
            meta_ansatz_i.set_value_function_constant_corner(meta_thetas_i)
            value_f_meta = meta_ansatz_i.value_function(x, meta_thetas_i)

            # ansatz functions evaluated at the grid
            v = self.sample.ansatz.basis_value_f(x)

            # solve theta V = \Phi
            thetas[i], _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)

        self.sample.ansatz.theta = np.mean(thetas, axis=0)

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

        self.sample.k_lim = self.k_lim
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
        f.write('total m: {:d}\n'.format(np.sum(self.ms)))
        f.write('total time steps: {:,d}\n\n'.format(int(np.sum(self.time_steps))))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        self.write_means(f)
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def plot_1d_updates(self, i=0):
        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        x = self.sample.domain_h[:, 0]

        # filter updates to show
        n_updates = self.ms[i]
        updates = np.arange(n_updates)
        sliced_updates = slice_1d_array(updates, n_elements=5)

        # preallocate functions
        labels = []
        frees = np.zeros((sliced_updates.shape[0] + 1, x.shape[0]))
        controls = np.zeros((sliced_updates.shape[0] + 1, x.shape[0]))
        controlled_potentials = np.zeros((sliced_updates.shape[0] + 1, x.shape[0]))

        # not controlled case
        labels.append(r'not controlled')
        self.sample.is_controlled = False
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()
        controlled_potentials[0, :] = self.sample.grid_controlled_potential
        frees[0, :] = self.sample.grid_value_function
        controls[0, :] = self.sample.grid_control[:, 0]

        self.sample.is_controlled = True
        for index, update in enumerate(sliced_updates):
            labels.append(r'update = {:d}'.format(update+1))

            self.set_ansatz_single_trajectory(i, update)

            self.sample.get_grid_value_function()
            self.sample.get_grid_control()

            # update functions
            controlled_potentials[index+1, :] = self.sample.grid_controlled_potential
            frees[index+1, :] = self.sample.grid_value_function
            controls[index+1, :] = self.sample.grid_control[:, 0]

        # get hjb solution
        sol = self.sample.get_hjb_solver(h=0.001)
        sol.get_controlled_potential_and_drift()

        # file extension
        ext = '_i_{}'.format(i)

        self.sample.plot_1d_free_energies(frees, F_hjb=sol.F, labels=labels[:],
                                          dir_path=self.dir_path, ext=ext)
        self.sample.plot_1d_controls(controls, u_hjb=sol.u_opt[:, 0], labels=labels[:],
                                     dir_path=self.dir_path, ext=ext)
        self.sample.plot_1d_controlled_potentials(controlled_potentials,
                                                  controlledV_hjb=sol.controlled_potential,
                                                  labels=labels[:], dir_path=self.dir_path,
                                                  ext=ext)

    def plot_1d_update(self, i=None, update=None):

        # plot given update for the chosen trajectory
        if i is not None:
            # number of updates of i meta trajectory
            n_updates = self.ms[i]
            updates = np.arange(n_updates)

            # if update not given choose last update
            if update is None:
                update = updates[-1]

            assert update in updates, ''

            # set plot dir path and file extension
            self.set_updates_dir_path()
            plot_dir_path = self.updates_dir_path
            ext = '_update{}'.format(update)

            # set ansatz
            self.set_ansatz_single_trajectory(i, update)

        # plot averaged bias potential
        else:
            self.set_ansatz_all_trajectories()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        self.sample.plot_1d_controlled_potential(self.sample.grid_controlled_potential,
                                                 dir_path=plot_dir_path, ext=ext)
        self.sample.plot_1d_control(self.sample.grid_control[:, 0],
                                    dir_path=plot_dir_path, ext=ext)
        self.sample.plot_1d_controlled_drift(self.sample.grid_controlled_drift[:, 0],
                                             dir_path=plot_dir_path, ext=ext)

    def plot_2d_update(self, i=None, update=None):

        # plot given update for the chosen trajectory
        if i is not None:
            # number of updates of i meta trajectory
            n_updates = self.ms[i]
            updates = np.arange(n_updates)

            # if update not given choose last update
            if update is None:
                update = updates[-1]

            assert update in updates, ''

            # set plot dir path and file extension
            self.set_updates_dir_path()
            plot_dir_path = self.updates_dir_path
            ext = '_update{}'.format(update)

            # set ansatz
            self.set_ansatz_single_trajectory(i, update)

        # plot averaged bias potential
        else:
            self.set_ansatz_all_trajectories()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.05)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        self.sample.plot_2d_controlled_potential(self.sample.grid_controlled_potential,
                                                 plot_dir_path, ext)
        self.sample.plot_2d_control(self.sample.grid_control, plot_dir_path, ext)
        self.sample.plot_2d_controlled_drift(self.sample.grid_controlled_drift,
                                             plot_dir_path, ext)
