from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils_path import get_metadynamics_dir_path, make_dir_path, empty_dir, get_time_in_hms
from mds.utils_numeric import slice_1d_array

import matplotlib.pyplot as plt
import numpy as np
import os
import time

META_TYPES = [
    'cum', # cumulative, bias potential with gaussian ansatz
    'cum-nn', # cumulative, control with nn
    'ind', # independent
]

WEIGHTS_TYPES = [
    'const', # constant
    'geom', # geometric succession
]

class Metadynamics:
    '''
    '''

    def __init__(self, sample, k, N, seed=None, meta_type='cum',
                 weights_type='geom', omega_0=1., do_updates_plots=False):

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
        self.meta_type = meta_type
        self.ms = None
        self.time_steps = None

        # succeeded
        self.succ = None

        # gaussian ansatz
        self.weights_type = weights_type
        self.omega_0 = omega_0
        self.omegas = None
        self.means = None
        self.sigma_i = 0.5
        self.cov = self.sigma_i * np.eye(sample.n)

        # nn
        self.thetas = None

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
        meta_rel_path = get_metadynamics_dir_path(
            self.meta_type,
            self.weights_type,
            self.omega_0,
            self.k,
            self.N,
        )
        self.dir_path = os.path.join(
            self.sample.settings_dir_path,
            meta_rel_path,
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
        self.ms = np.zeros(self.N, dtype=np.int)
        self.omegas = np.empty(0)
        self.means = np.empty((0, self.sample.n))
        self.time_steps = np.empty(self.N, dtype=np.int32)

        # nn parameters
        if self.meta_type == 'cum-nn':
            self.thetas = np.empty(0)

        # boolean array telling us if the algorithm succeeded or not for each sample
        self.succ = np.empty(self.N, dtype=bool)

    def set_weights(self):
        # constant weights
        if self.weights_type == 'const' and self.meta_type == 'ind':
            self.weights = self.omega_0 * np.ones(self.updates_lim)
        elif self.weights_type == 'const' and self.meta_type == 'cum':
            self.weights = self.omega_0 * np.ones(self.N)

        # geometric decay
        elif self.weights_type == 'geometric' and self.meta_type == 'ind':
            r = 0.95
            self.weights = np.array([self.omega_0 * (r**i) for i in np.arange(self.updates_lim)])
        elif self.weights_type == 'geometric' and self.meta_type == 'cum':
            r = 0.95
            self.weights = np.array([self.omega_0 * (r**i) for i in np.arange(self.N)])

    def independent_metadynamics_algorithm(self, i):
        '''
        '''
        sample = self.sample

        # reset initial position
        sample.xzero = np.full((sample.N, self.sample.n), self.xzero)

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(self.updates_lim):

            # set controlled flag
            if j == 0:
                sample.is_controlled = False
            else:
                sample.is_controlled = True
                idx = slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + j)
                sample.ansatz.set_given_ansatz_functions(self.means[idx], self.cov)
                sample.ansatz.theta = self.weights[:j] / 2

            # sample with the given weights
            self.succ[i], xtemp = sample.sample_meta()

            # if trajectory arrived update used time stemps
            if self.succ[i]:
                time_steps += xtemp.shape[0]
                break

            # add new bias ansatz function and weight
            self.means = np.vstack((self.means, np.mean(xtemp, axis=(0, 1))))

            # update initial point
            sample.xzero = np.full((sample.N, self.sample.n), np.mean(xtemp[-1]))

            # update used time steps
            time_steps += sample.k_lim

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.omegas = np.append(self.omegas, self.weights[:j])
        self.time_steps[i] = time_steps

    def cumulative_metadynamics_algorithm(self, i):
        '''
        '''
        sample = self.sample

        # reset initial position
        sample.xzero = np.full((sample.N, self.sample.n), self.xzero)

        # time steps of the sampled meta trajectory
        time_steps = 0

        for j in np.arange(self.updates_lim):

            # set controlled flag
            if self.means.shape[0] == 0:
                sample.is_controlled = False
            else:
                sample.is_controlled = True
                sample.ansatz.set_given_ansatz_functions(self.means, self.cov)
                sample.ansatz.theta = self.omegas / 2

            # sample with the given weights
            self.succ[i], xtemp = sample.sample_meta()

            # if trajectory arrived update used time stemps
            if self.succ[i]:
                time_steps += xtemp.shape[0]
                break

            # add new bias ansatz function and weight
            self.omegas = np.append(self.omegas, self.weights[i])
            self.means = np.vstack((self.means, np.mean(xtemp, axis=(0, 1))))

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
        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save npz file
        np.savez(
            os.path.join(self.dir_path, 'bias-potential.npz'),
            meta_type=self.meta_type,
            weights_type=self.weights_type,
            omega_0=self.omega_0,
            omegas=self.omegas,
            k=self.k,
            N=self.N,
            succ=self.succ,
            time_steps=self.time_steps,
            ms=self.ms,
            means=self.means,
            thetas=self.thetas,
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
        # trajectory
        assert i in range(self.N), ''

        # if None assign last update 
        if update is None:
            update = self.ms[i]
        else:
            assert update in range(self.ms[i] + 1), ''

        # independent trajectories
        if self.meta_type == 'ind':
            return slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + update)

        # cumulative trajectories
        elif self.meta_type == 'cum':
            return slice(0, np.sum(self.ms[:i]) + update)

    def set_ansatz_trajectory(self, i, update=None):
        '''
        '''
        # get ansatz indices
        idx = self.get_trajectory_indices(i, update)

        # set ansatz and theta
        self.sample.ansatz.set_given_ansatz_functions(
            means=self.means[idx],
            cov=self.cov,
        )
        self.sample.ansatz.theta = self.omegas[idx] / 2
        self.sample.ansatz.set_value_function_constant_corner()

    def set_ansatz_cumulative(self):
        self.set_ansatz_trajectory(i=self.N -1)

    def set_ansatz_averaged(self):
        ''' average omegas such that each trajectory is equally important
        '''

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
            meta_thetas_i = self.omegas[idx_i] / 2

            # create ansatz functions corresponding to the ith metadynamics trajectory
            meta_ansatz_i = GaussianAnsatz(n=self.sample.n, normalized=False)
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
        idx = 0
        for i in np.arange(self.N):
            for j in np.arange(self.ms[i]):

                # get center added at the trajectory i update j
                mean = self.means[idx]
                idx += 1

                # get string
                mean_str = '('
                for x_i in range(self.sample.n):
                    if x_i == 0:
                        mean_str += '{:2.2f}'.format(mean[x_i])
                    else:
                        mean_str += ', {:2.2[}'.format(mean[x_i])
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

        f.write('\nMetadynamics parameters and statistics\n')
        f.write('metadynamics algorithm: {}\n'.format(self.meta_type))
        f.write('k: {:d}\n'.format(self.k))
        f.write('N_meta: {:d}\n\n'.format(self.N))


        f.write('\nGaussians\n')
        f.write('sigma_i_meta: {:2.2f}\n'.format(self.sigma_i))
        f.write('weights decay: {}\n'.format(self.weights_type))
        f.write('omega_0: {}\n'.format(self.omega_0))

        f.write('\nseed: {:d}\n'.format(self.seed))
        f.write('traj succeeded: {:2.2f} %\n'
                ''.format(100 * np.sum(self.succ) / self.N))
        f.write('total m: {:d}\n'.format(np.sum(self.ms)))
        f.write('total time steps: {:,d}\n\n'.format(int(np.sum(self.time_steps))))

        h, m, s = get_time_in_hms(self.ct)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        self.write_means(f)
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def plot_1d_updates(self, i=0, n_sliced_updates=5):
        from figures.myfigure import MyFigure

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        x = self.sample.domain_h[:, 0]

        # filter updates to show
        n_updates = self.ms[i]
        updates = np.arange(n_updates)

        if n_updates < n_sliced_updates:
            sliced_updates = updates
            n_sliced_updates = n_updates
        else:
            sliced_updates = slice_1d_array(updates, n_elements=n_sliced_updates)
            n_sliced_updates = sliced_updates.shape[0]

        # preallocate arrays
        frees = np.zeros((n_sliced_updates + 2, x.shape[0]))
        controls = np.zeros((n_sliced_updates + 2, x.shape[0]))
        controlled_potentials = np.zeros((n_sliced_updates + 2, x.shape[0]))

        # preallocate functions
        labels = []
        colors = [None for i in range(n_sliced_updates + 2)]

        # the initial bias potential is the not controlled potential
        if self.meta_type == 'ind':
            labels.append(r'not controlled potential')
            self.sample.is_controlled = False

        # the initial bias potential is given by the previous trajectory
        elif self.meta_type == 'cum':
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
        frees[-1, :] = sol_hjb.F
        controlled_potentials[-1, :] = sol_hjb.controlled_potential
        controls[-1, :] = sol_hjb.u_opt[:, 0]
        labels.append('num sol HJB PDE')
        colors[-1] = 'tab:cyan'

        # file extension
        ext = '_i_{}'.format(i)

        # plot free energy
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='free-energy' + ext,
        )
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        fig.plot(x, frees, labels=labels, colors=colors)

        # plot controlled potential
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='controlled-potential' + ext,
        )
        controlled_potentials[-1, :] = sol_hjb.controlled_potential
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        fig.set_ylim(0, 20)
        fig.plot(x, controlled_potentials, labels=labels, colors=colors)

        # plot control
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='control' + ext,
        )
        controls[-1, :] = sol_hjb.u_opt[:, 0]
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        fig.plot(x, controls, labels=labels, colors=colors)

    def plot_1d_update(self, i=None, update=None):
        from figures.myfigure import MyFigure

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

        # plot bias potential
        else:
            if self.meta_type == 'ind':
                self.set_ansatz_averaged()
            elif self.meta_type == 'cum':
                self.set_ansatz_cumulative()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.001)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        # get hjb solution
        sol_hjb = self.sample.get_hjb_solver(h=0.001)
        sol_hjb.discretize_domain()
        sol_hjb.get_controlled_potential_and_drift()

        # colors and labels
        labels = [r'meta (trajectory: {}, update: {})'.format(i, update), 'num sol HJB PDE']
        colors = ['tab:purple', 'tab:cyan']

        # domain
        x = self.sample.domain_h[:, 0]

        # plot free energy
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=plot_dir_path,
            file_name='free-energy' + ext,
        )
        y = np.vstack((
            self.sample.grid_value_function,
            sol_hjb.F,
        ))
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        fig.plot(x, y, labels=labels, colors=colors)

        # plot controlled potential
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=plot_dir_path,
            file_name='controlled-potential' + ext,
        )
        y = np.vstack((
            self.sample.grid_controlled_potential,
            sol_hjb.controlled_potential,
        ))
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        fig.plot(x, y, labels=labels, colors=colors)

        # plot control
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=plot_dir_path,
            file_name='control' + ext,
        )
        y = np.vstack((
            self.sample.grid_control[:, 0],
            sol_hjb.u_opt[:, 0],
        ))
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        fig.plot(x, y, labels=labels, colors=colors)


    def plot_2d_update(self, i=None, update=None):
        from figures.myfigure import MyFigure

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
        elif self.meta_type == 'ind':
            self.set_ansatz_averaged()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        elif self.meta_type == 'cum':
            self.set_ansatz_cumulative()

            # set plot dir path and file extension
            plot_dir_path = self.dir_path
            ext = ''

        # discretize domain and evaluate in grid
        self.sample.discretize_domain(h=0.05)
        self.sample.get_grid_value_function()
        self.sample.get_grid_control()

        # domain
        X = self.sample.domain_h[:, :, 0]
        Y = self.sample.domain_h[:, :, 1]

        # plot free energy
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=plot_dir_path,
            file_name='free-energy' + ext,
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_contour_levels_scale('log')
        fig.contour(X, Y, self.sample.grid_value_function)

        # plot controlled potential
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=plot_dir_path,
            file_name='controlled-potential' + ext,
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_contour_levels_scale('log')
        fig.contour(X, Y, self.sample.grid_controlled_potential)

        # plot control
        U = self.sample.grid_control[:, :, 0]
        V = self.sample.grid_control[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=plot_dir_path,
            file_name='control' + ext,
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.vector_field(X, Y, U, V, scale=30)

    def plot_n_gaussians(self, dir_path=None):
        ''' plot number of gaussians added at each trajectory
        '''
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='n-gaussians',
        )
        x = np.arange(self.N)
        fig.set_xlabel('trajectories')
        fig.set_plot_scale('semilogx')
        fig.plot(x, self.ms)
        plt.savefig(fig.file_path)

    def plot_fht(self):
        ''' plot fht for each meta trajectory
        '''
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='fht',
        )
        x = np.arange(self.N)
        y = np.vstack((
                self.time_steps,
                np.full(self.N, self.k),
        ))
        labels = ['cumulative metadynamics', 'k']
        fig.set_xlabel('trajectories')
        fig.plot(x, y, labels=labels)
        plt.savefig(fig.file_path)


    def plot_2d_means(self):
        from figures.myfigure import MyFigure
        assert self.means.shape[1] == 2, ''
        x, y = np.moveaxis(self.means, -1, 0)
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='means',
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(x, y)
        plt.savefig(fig.file_path)

    def plot_3d_means(self):
        from figures.myfigure import MyFigure

        assert self.means.shape[1] == 3, ''
        x, y, z = np.moveaxis(self.means, -1, 0)

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='means',
        )
        plt.scatter(x, y, z)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig(plt.file_path)
        plt.close()

    def plot_projected_means(self, i, j):
        from figures.myfigure import MyFigure

        m = np.sum(self.ms)
        proj_means = np.empty((m, 2))
        for k in range(m):
            proj_means[k, 0] = self.means[k, i]
            proj_means[k, 1] = self.means[k, j]
        x, y = np.moveaxis(proj_means, -1, 0)

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='means-{}-{}'.format(i, j),
        )
        plt.scatter(x, y)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig(fig.file_path)
        plt.close()

    def plot_nd_means(self):
        n = self.sample.n
        idx_dim_pairs = [(i, j) for i in range(n) for j in range(n) if j > i]
        for i, j in idx_dim_pairs:
            self.plot_projected_means(i, j)
