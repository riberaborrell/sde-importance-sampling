from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils import get_metadynamics_dir_path, make_dir_path, empty_dir, get_time_in_hms
from mds.numeric_utils import slice_1d_array
from mds.plots import Plot

import time
import numpy as np
import os

class Metadynamics:
    '''
    '''

    def __init__(self, sample, k, N, sigma_i, seed=None,
                 is_cumulative=False, do_updates_plots=False):

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
        self.is_cumulative = is_cumulative
        self.ms = None
        self.thetas = None
        self.means = None
        self.sigma_i = sigma_i
        self.cov = None
        self.time_steps = None

        # succeeded
        self.succ = None

        # computational time
        self.ct_initial = None
        self.ct_final = None
        self.ct = None

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
            self.is_cumulative,
            self.k,
            self.N,
        )

    def set_updates_dir_path(self):
        self.updates_dir_path = os.path.join(self.dir_path, 'updates')
        make_dir_path(self.updates_dir_path)
        empty_dir(self.updates_dir_path)

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

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

    def preallocate_metadynamics_coefficients(self):
        # bias potentials coefficients
        self.ms = np.empty(self.N, dtype=np.intc)
        self.thetas = np.empty(0)
        self.means = np.empty((0, self.sample.n))
        self.cov = self.sigma_i * np.eye(self.sample.n)
        self.time_steps = np.empty(self.N, dtype=np.int32)

        # boolean array telling us if the algorithm succeeded or not for each sample
        self.succ = np.empty(self.N, dtype=bool)

    def get_weights_trajectory(self):
        # constant weights
        #omegas = 1 * np.ones(updates)

        # exponential
        omegas = 0.99 * np.ones(self.updates_lim)
        omegas = np.array([w**(i+1) for i, w in enumerate(omegas)])

        return omegas

    def independent_metadynamics_algorithm(self, i):
        '''
        '''
        sample = self.sample

        # reset initial position
        sample.xzero = np.full((sample.N, self.sample.n), self.xzero)

        # preallocate means and cov matrix of the gaussiansbias functions
        means = np.empty((0, self.sample.n))

        # set the weights of the bias functions for this trajectory
        omegas = self.get_weights_trajectory()

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(self.updates_lim):

            # set controlled flag
            if j == 0:
                sample.is_controlled = False
            else:
                sample.is_controlled = True

            # sample with the given weights
            self.succ[i], xtemp = sample.sample_meta()

            # if trajectory arrived update used time stemps
            if self.succ[i]:
                time_steps += xtemp.shape[0]
                break

            # add new bias ansatz function and weight
            means = np.vstack((means, np.mean(xtemp, axis=(0, 1))))
            #print('({:2.3f}, {:2.3f})'.format(means[j, 0], means[j, 1]))

            sample.ansatz.set_given_ansatz_functions(means, self.cov)
            sample.ansatz.theta = omegas[:j+1] / 2
            sample.xzero = np.full((sample.N, self.sample.n), np.mean(xtemp[-1]))

            # update used time steps
            time_steps += sample.k_lim

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.time_steps[i] = time_steps
        if j > 0:
            self.thetas = np.append(self.thetas, sample.ansatz.theta)
            self.means = np.vstack((self.means, sample.ansatz.means))

    def cumulative_metadynamics_algorithm(self, i):
        '''
        '''
        sample = self.sample

        # reset initial position
        sample.xzero = np.full((sample.N, self.sample.n), self.xzero)

        # set the weights of the bias functions for this trajectory
        omegas = self.get_weights_trajectory()

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(self.updates_lim):

            # set controlled flag
            if self.means.shape[0] == 0:
                sample.is_controlled = False
            else:
                sample.is_controlled = True
                sample.ansatz.set_given_ansatz_functions(self.means, self.cov)
                sample.ansatz.theta = self.thetas

            # sample with the given weights
            self.succ[i], xtemp = sample.sample_meta()

            # if trajectory arrived update used time stemps
            if self.succ[i]:
                time_steps += xtemp.shape[0]
                break

            # add new bias ansatz function and weight
            self.means = np.vstack((self.means, np.mean(xtemp, axis=(0, 1))))
            self.thetas = np.append(self.thetas, omegas[j+1] / 2)

            # update initial point
            sample.xzero = np.full((sample.N, self.sample.n), np.mean(xtemp[-1]))

            # update used time steps
            time_steps += sample.k_lim

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.time_steps[i] = time_steps

    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        np.savez(
            os.path.join(self.dir_path, 'bias-potential.npz'),
            dt=self.sample.dt,
            succ=self.succ,
            is_cumulative=self.is_cumulative,
            ms=self.ms,
            thetas=self.thetas,
            means=self.means,
            cov=self.cov,
            time_steps=self.time_steps,
            ct = self.ct,
        )

    def load(self):
        ''' loads the saved arrays and sets them as attributes back
        '''
        try:
            data = np.load(
                os.path.join(self.dir_path, 'bias-potential.npz'),
                allow_pickle=True,
            )
            for file_name in data.files:
                if data[file_name].ndim == 0:
                    setattr(self, file_name, data[file_name][()])
                else:
                    setattr(self, file_name, data[file_name])
            return True

        except:
            msg = 'no meta bias potential found with dt={:.4f}, sigma_i={:.2f}, k={:d}, ' \
                  'N={:.0e}'.format(self.sample.dt, self.sigma_i, self.k, self.N)
            print(msg)
            return False

    def get_trajectory_indices(self, i, update=None):
        ''' returns the indices of the ansatz functions used for each trajectory up to the given update
        '''
        assert i in range(self.N), ''
        if update is not None:
            assert update in range(self.ms[i] + 1), ''

        if not self.is_cumulative and update is None:
            return slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + self.ms[i])
        elif not self.is_cumulative and update is not None:
            return slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + update)
        elif self.is_cumulative and update is None:
            return slice(0, np.sum(self.ms[:i]) + self.ms[i])
        else:
            return slice(0, np.sum(self.ms[:i]) + update)

    def set_ansatz_trajectory(self, i, update):
        '''
        '''
        # get ansatz indices
        idx = self.get_trajectory_indices(i, update)
        means = self.means[idx]
        thetas = self.thetas[idx]

        # set ansatz and theta
        self.sample.ansatz.set_given_ansatz_functions(
            means=means,
            cov=self.cov,
        )
        self.sample.ansatz.theta = thetas
        self.sample.ansatz.set_value_function_constant_corner()

    def set_ansatz_cumulative(self):
        self.sample.ansatz.set_given_ansatz_functions(
            means=self.means,
            cov=self.cov,
        )
        self.sample.ansatz.theta = self.thetas
        self.sample.ansatz.set_value_function_constant_corner()

    def set_ansatz_averaged(self):

        # set all means used for each trajectory
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
        self.sample.ansatz.set_value_function_constant_corner()

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
        f.write('cumulative: {}\n'.format(self.is_cumulative))
        f.write('seed: {:d}\n'.format(self.seed))
        f.write('sigma_i_meta: {:2.2f}\n'.format(self.sigma_i))
        f.write('k: {:d}\n'.format(self.k))
        f.write('N_meta: {:d}\n\n'.format(self.N))

        f.write('traj succeeded: {:2.2f} %\n'
                ''.format(100 * np.sum(self.succ) / self.N))
        f.write('total m: {:d}\n'.format(np.sum(self.ms)))
        f.write('total time steps: {:,d}\n\n'.format(int(np.sum(self.time_steps))))

        h, m, s = get_time_in_hms(self.ct)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        #self.write_means(f)
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def plot_n_gaussians(self):
        trajectories = np.arange(self.N)
        plt = Plot(self.dir_path, 'n_gaussians')
        plt.one_line_plot(trajectories, self.ms)

    def plot_1d_updates(self, i=0):
        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        x = self.sample.domain_h[:, 0]

        # filter updates to show
        n_sliced_updates = 5
        n_updates = self.ms[i]
        updates = np.arange(n_updates)

        if n_updates < n_sliced_updates:
            sliced_updates = updates
        else:
            sliced_updates = slice_1d_array(updates, n_elements=n_sliced_updates)

        # preallocate functions
        labels = []
        frees = np.zeros((sliced_updates.shape[0] + 1, x.shape[0]))
        controls = np.zeros((sliced_updates.shape[0] + 1, x.shape[0]))
        controlled_potentials = np.zeros((sliced_updates.shape[0] + 1, x.shape[0]))

        # the initial bias potential is the not controlled potential
        if not self.is_cumulative:
            labels.append(r'not controlled potential')
            self.sample.is_controlled = False

        # the initial bias potential is given by the previous trajectory
        else:
            labels.append(r'initial bias potential')
            self.sample.is_controlled = True
            self.set_ansatz_trajectory(i, update=0)

        # evaluate at the grid
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()
        controlled_potentials[0, :] = self.sample.grid_controlled_potential
        frees[0, :] = self.sample.grid_value_function
        controls[0, :] = self.sample.grid_control[:, 0]

        self.sample.is_controlled = True
        for index, update in enumerate(sliced_updates):
            labels.append(r'update = {:d}'.format(update + 1))

            self.set_ansatz_trajectory(i, update + 1)

            self.sample.get_grid_value_function()
            self.sample.get_grid_control()

            # update functions
            controlled_potentials[index+1, :] = self.sample.grid_controlled_potential
            frees[index+1, :] = self.sample.grid_value_function
            controls[index+1, :] = self.sample.grid_control[:, 0]

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.discretize_domain()
        sol_hjb.get_controlled_potential_and_drift()

        # file extension
        ext = '_i_{}'.format(i)

        self.sample.plot_1d_free_energies(frees, F_hjb=sol_hjb.F, labels=labels[:],
                                          dir_path=self.dir_path, ext=ext)
        self.sample.plot_1d_controls(controls, u_hjb=sol_hjb.u_opt[:, 0], labels=labels[:],
                                     dir_path=self.dir_path, ext=ext)
        self.sample.plot_1d_controlled_potentials(controlled_potentials,
                                                  controlledV_hjb=sol_hjb.controlled_potential,
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
            self.set_ansatz_trajectory(i, update)

        # plot averaged bias potential
        else:
            if not self.is_cumulative:
                self.set_ansatz_averaged()
            else:
                self.set_ansatz_cumulative()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        controlled_potentials = self.sample.grid_controlled_potential
        frees = self.sample.grid_value_function
        controls = self.sample.grid_control

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.discretize_domain()
        sol_hjb.get_controlled_potential_and_drift()


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
            self.set_ansatz_trajectory(i, update)

        # plot averaged bias potential
        elif not self.is_cumulative:
            self.set_ansatz_averaged()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        else:
            self.set_ansatz_cumulative()

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

    def plot_2d_means(self):

        #self.set_ansatz_all_trajectories()
        x, y = np.moveaxis(self.means, -1, 0)
        plt = Plot(self.dir_path, 'means')
        plt.plt.scatter(x, y)
        plt.plt.xlim(-2, 2)
        plt.plt.ylim(-2, 2)
        plt.plt.savefig(plt.file_path)
        plt.plt.close()



