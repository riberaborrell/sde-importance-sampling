from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils import get_metadynamics_nn_dir_path, make_dir_path, empty_dir, get_time_in_hms
from mds.numeric_utils import slice_1d_array
from mds.plots import Plot

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import copy
import time
import os

META_TYPES = [
    'cum', # cumulative
    'ind', # independent
]

class MetadynamicsNN:
    '''
    '''

    def __init__(self, sample, k, N, sigma_i, seed=None,
                 meta_type='cum', do_updates_plots=False):

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
        self.dir_path = get_metadynamics_nn_dir_path(
            self.sample.settings_dir_path,
            self.sample.dt,
            self.sigma_i,
            self.meta_type,
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
        # flat dimension of the nn
        m = self.sample.nn_func_appr.model.d_flat

        # bias potentials coefficients
        self.ms = np.empty(self.N, dtype=np.intc)
        self.omegas = np.empty(0)
        self.means = np.empty((0, self.sample.n))
        self.cov = self.sigma_i * np.eye(self.sample.n)
        self.thetas = np.empty((0, m))
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
                self.train_nn_with_new_gaussian()

            # sample with the given weights
            self.succ[i], xtemp = sample.sample_meta()

            # if trajectory arrived update used time stemps
            if self.succ[i]:
                time_steps += xtemp.shape[0]
                break

            # add weight and center of the gaussian function used to refit the bias potential
            self.omegas = np.append(self.omegas, omegas[j])
            self.means = np.vstack((self.means, np.mean(xtemp, axis=(0, 1))))
            print(self.means[-1])

            # update initial point
            sample.xzero = np.full((sample.N, self.sample.n), np.mean(xtemp[-1]))

            # update used time steps
            time_steps += sample.k_lim

        # save bias functions added for this trajectory
        self.ms[i] = j
        self.time_steps[i] = time_steps

    def train_nn_with_new_gaussian(self):

        # freezed model and model
        model = self.sample.nn_func_appr.model
        freezed_model = copy.deepcopy(self.sample.nn_func_appr.model)

        # define optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.01,
        )

        # define loss
        loss = nn.MSELoss()

        # training parameters
        self.n_iterations_lim = 10**4
        self.N_train = 10**3
        self.epsilon = 0.01

        # initialize Gaussian Ansatz
        ansatz = GaussianAnsatz(self.sample.n, normalized=False)

        for i in np.arange(self.n_iterations_lim):

            # sample training data
            x = self.sample.sample_domain_uniformly(self.N_train)
            #x = self.sample.sample_multivariate_normal(self.means[-1], self.cov, self.N_train)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # bias potential + gaussian 
            if self.sample.nn_func_appr.target_function == 'value-f':
                pass
            elif self.sample.nn_func_appr.target_function == 'control':
                freezed_control = freezed_model.forward(x_tensor).detach()
                grad_mv = np.squeeze(ansatz.vec_grad_mv_normal_pdf(x, self.means[-1:], self.cov))
                gaussian_control = - self.omegas[-1] * grad_mv / np.sqrt(2)
                gaussian_control_tensor = torch.tensor(gaussian_control, requires_grad=False,
                                                       dtype=torch.float32)
                target_tensor = freezed_control + gaussian_control_tensor

            # compute loss
            inputs = model.forward(x_tensor)
            output = loss(inputs, target_tensor)

            if i % 100 == 0:
                print('{:d}, {:2.3f}'.format(i, output))

            # stop if we have reached enough accuracy
            if output <= self.epsilon:
                break

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        # save flattened parameters of the nn
        self.thetas = np.vstack((self.thetas, model.get_parameters()))
        return

        self.sample.discretize_domain(h=0.05)
        self.sample.get_grid_control()

        # coordinates
        X = self.sample.domain_h[:, :, 0]
        Y = self.sample.domain_h[:, :, 1]

        # control
        U = self.sample.grid_control[:, :, 0]
        V = self.sample.grid_control[:, :, 1]

        # plot control
        from mds.plots import Plot
        plt = Plot(self.dir_path, 'control')
        plt.set_xlim(-2, 2)
        plt.set_ylim(-2, 2)
        plt.vector_field(X, Y, U, V, scale=5)

    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        np.savez(
            os.path.join(self.dir_path, 'bias-potential.npz'),
            dt=self.sample.dt,
            succ=self.succ,
            is_cumulative=self.meta_type,
            ms=self.ms,
            omegas=self.omegas,
            means=self.means,
            cov=self.cov,
            thetas=self.thetas,
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

        if self.meta_type == 'ind' and update is None:
            return slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + self.ms[i])
        elif self.meta_type == 'ind' and update is not None:
            return slice(np.sum(self.ms[:i]), np.sum(self.ms[:i]) + update)
        elif self.meta_type == 'cum' and update is None:
            return slice(0, np.sum(self.ms[:i]) + self.ms[i])
        else:
            return slice(0, np.sum(self.ms[:i]) + update)

    def set_parameters_trajectory(self, i, update):
        '''
        '''
        # get trajectory indices after a given update
        idx = self.get_trajectory_indices(i, update)
        thetas = self.thetas[idx]

        # set parameters
        self.sample.nn_func_appr.model.load_parameters(thetas)


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

        self.sample.nn_func_appr.write_parameters(f)

        f.write('\nMetadynamics parameters and statistics\n')
        f.write('metadynamics algorithm type: {}\n'.format(self.meta_type))
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

    def plot_n_gaussians(self, dir_path=None):
        '''
        '''

        # set directory path
        if dir_path is None:
            dir_path = self.dir_path

        trajectories = np.arange(self.N)
        plt = Plot(dir_path, 'n_gaussians')
        plt.plt.semilogx(trajectories, meta_cum.ms)
        plt.plt.savefig(plt.file_path)
        plt.plt.close()

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
        assert self.means.shape[1] == 2, ''

        x, y = np.moveaxis(self.means, -1, 0)
        plt = Plot(self.dir_path, 'means')
        plt.plt.scatter(x, y)
        plt.plt.xlim(-2, 2)
        plt.plt.ylim(-2, 2)
        plt.plt.savefig(plt.file_path)
        plt.plt.close()

    def plot_3d_means(self):
        assert self.means.shape[1] == 3, ''

        x, y, z = np.moveaxis(self.means, -1, 0)
        plt = Plot(self.dir_path, 'means')
        plt.plt.scatter(x, y, z)
        plt.plt.xlim(-2, 2)
        plt.plt.ylim(-2, 2)
        plt.plt.savefig(plt.file_path)
        plt.plt.close()

    def plot_projected_means(self, i, j):
        m = np.sum(self.ms)
        proj_means = np.empty((m, 2))
        for k in range(m):
            proj_means[k, 0] = self.means[k, i]
            proj_means[k, 1] = self.means[k, j]
        x, y = np.moveaxis(proj_means, -1, 0)
        plt = Plot(self.dir_path, 'means-{}-{}'.format(i, j))
        plt.plt.scatter(x, y)
        plt.plt.xlim(-2, 2)
        plt.plt.ylim(-2, 2)
        plt.plt.savefig(plt.file_path)
        plt.plt.close()

    def plot_nd_means(self):
        n = self.sample.n
        idx_dim_pairs = [(i, j) for i in range(n) for j in range(n) if j > i]
        for i, j in idx_dim_pairs:
            self.plot_projected_means(i, j)
