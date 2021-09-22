from mds.potentials_and_gradients_nd import get_potential_and_gradient
from mds.plots import Plot
from mds.utils import get_settings_dir_path, \
                      get_metadynamics_dir_path

import numpy as np

import os
import sys

class LangevinSDE(object):
    '''
    '''

    def __init__(self, potential_name, n, alpha, beta,
                 target_set=None, domain=None, h=None):
        '''
        '''
        # get potential and gradient functions
        potential, gradient, _ = get_potential_and_gradient(potential_name, n, alpha)

        # domain and target set
        if domain is None:
            domain = np.full((n, 2), [-3, 3])
        if target_set is None:
            target_set = np.full((n, 2), [1, 3])

        # sde parameters
        self.potential_name = potential_name
        self.n = n
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.target_set = target_set
        self.domain = domain

        # domain discretization
        self.h = h
        self.domain_h = None
        self.Nx = None
        self.Nh = None

        # dir_path
        self.settings_dir_path = None
        self.set_settings_dir_path()

    @classmethod
    def new_from(cls, obj):
        if issubclass(obj.__class__, LangevinSDE):
            _new = cls(obj.potential_name, obj.n, obj.alpha, obj.beta,
                       obj.target_set, obj.domain, obj.h)
            return _new
        else:
            msg = 'Expected subclass of <class LangevinSDE>, got {}.'.format(type(obj))
            raise TypeError(msg)

    def set_settings_dir_path(self):

        if self.potential_name == 'nd_2well':
            assert (self.alpha == self.alpha[0]).all(), ''

        elif self.potential_name == 'nd_2well_asym':
            assert self.n > 1, ''
            assert self.alpha[0] != self.alpha[1], ''

        self.settings_dir_path = get_settings_dir_path(self.potential_name, self.n,
                                                      self.alpha, self.beta, 'hypercube')
    def discretize_domain(self, h=None):
        ''' this method discretizes the hyper-rectangular domain uniformly with step-size h
        '''
        if h is not None:
            self.h = h
        assert self.h is not None, ''

        # construct not sparse nd grid
        try:
            mgrid_input = []
            for i in range(self.n):
                mgrid_input.append(
                    slice(self.domain[i, 0], self.domain[i, 1] + self.h, self.h)
                )
            self.domain_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)
        except MemoryError as e:
            print('MemoryError: {}'.format(e))
            sys.exit()

        # check shape
        assert self.domain_h.shape[-1] == self.n, ''

        # save number of indices per axis
        self.Nx = self.domain_h.shape[:-1]

        # save number of flattened indices
        N = 1
        for i in range(self.n):
            N *= self.Nx[i]
        self.Nh = N

    def sample_domain_uniformly(self, N):
        x = np.random.uniform(
            self.domain[:, 0],
            self.domain[:, 1],
            (N, self.n),
        )
        return x


    def sample_domain_boundary_uniformly(self):
        x = np.empty(self.n)
        i = np.random.randint(self.n)
        for j in range(self.n):
            if j == i:
                k = np.random.randint(2)
                x[j] = self.domain[j, k]
            else:
                x[j] = np.random.uniform(
                    self.domain[j, 0],
                    self.domain[j, 1],
                    1
                )
        return x

    def sample_domain_boundary_uniformly_vec(self, N):
        x = self.sample_domain_uniformly(N)
        i = np.random.randint(self.n, size=N)
        k = np.random.randint(2, size=N)
        for j in np.arange(N):
            x[j, i[j]] = self.domain[i[j], k[j]]
        return x

    def sample_S_uniformly(self, N):
        x = np.empty((N, self.n))
        for j in range(N):
            is_in_target_set = True
            while is_in_target_set:
                x_j = self.sample_domain_uniformly(N=1)
                is_in_target_set = (
                    (x_j >= self.target_set[:, 0]) &
                    (x_j <= self.target_set[:, 1])
                ).all()
            x[j] = x_j
        return x

    def get_flattened_domain_h(self):
        ''' this method returns the flattened discretized domain
        '''
        return self.domain_h.reshape(self.Nh, self.n)

    def get_index(self, x):
        ''' returns the index of the point of the grid closest to x
        '''
        assert x.ndim == 1, ''
        assert x.shape[0] == self.n, ''

        idx = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], self.Nx[i])
            idx[i] = np.argmin(np.abs(axis_i - x[i]))

        return tuple(idx)

    def get_index_vectorized(self, x):
        ''' returns the index of the point of the grid closest to x
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        # get index of xzero
        idx = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], self.Nx[i])
            idx[i] = tuple(np.argmin(np.abs(axis_i - x[:, i].reshape(x.shape[0], 1)), axis=1))

        idx = tuple(idx)
        return idx

    def get_x(self, idx):
        ''' returns the coordinates of the point determined by the axis indices idx
        '''
        return self.domain_h[idx]

    def get_idx_target_set(self):
        assert self.domain_h is not None, ''

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.n)

        # boolean array telling us if x is in the target set
        is_in_target_set = (
            (x >= self.target_set[:, 0]) &
            (x <= self.target_set[:, 1])
        ).all(axis=1).reshape(self.Nh, 1)

        # get index
        return np.where(is_in_target_set == True)[0]

    def get_hjb_solver(self, h=None):
        from mds.langevin_nd_hjb_solver import SolverHJB

        if h is None and self.n == 1:
            h = 0.001
        elif h is None and self.n == 2:
            h = 0.005
        elif h is None and self.n == 3:
            h = 0.1
        elif h is None:
            return

        # initialize hjb solver
        sol_hjb = SolverHJB(
            potential_name=self.potential_name,
            n=self.n,
            alpha=self.alpha,
            beta=self.beta,
            h=h,
        )

        # load already computed solution
        sol_hjb.load()
        return sol_hjb

    def get_not_controlled_sampling(self, dt, N):
        from mds.langevin_nd_importance_sampling import Sampling

        # initialize not controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = False

        # set Euler-Marujama discretiztion step and number of trajectories
        sample.dt = dt
        sample.N = N

        # set path
        sample.set_not_controlled_dir_path()

        # load already sampled statistics
        sample.load()
        return sample

    def get_metadynamics_sampling(self, dt, sigma_i, is_cumulative, k, N):
        from mds.langevin_nd_importance_sampling import Sampling
        from mds.langevin_nd_metadynamics import Metadynamics

        # initialize controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = True
        sample.dt = dt

        # initialize meta nd object
        meta = Metadynamics(
            sample=sample,
            k=k,
            N=N,
            sigma_i=sigma_i,
        )
        meta.is_cumulative = is_cumulative

        # set path
        meta.set_dir_path()

        # load already sampled trajectories
        meta.load()
        return meta

    def get_flat_bias_sampling(self, dt, k_lim, N):
        from mds.langevin_nd_importance_sampling import Sampling
        from mds.langevin_nd_flat_bias_potential import GetFlatBiasPotential

        # initialize sampling object
        sample = Sampling.new_from(self)
        sample.dt = dt
        sample.k_lim = k_lim
        sample.N = N

        # initialize flatbias object
        flatbias = GetFlatBiasPotential(sample)

        # set path
        flatbias.set_dir_path()

        # load already sampled trajectories
        flatbias.load()
        return flatbias

    def write_setting(self, f):
        '''
        '''
        f.write('\nSDE Setting\n')
        f.write('potential: {}\n'.format(self.potential_name))
        f.write('alpha: {}\n'.format(self.alpha))
        f.write('beta: {:2.1f}\n'.format(self.beta))

        target_set = 'target set: ['
        for i in range(self.n):
            i_axis_str = '[{:2.1f}, {:2.1f}]'.format(self.target_set[i, 0], self.target_set[i, 1])
            if i == 0:
                target_set += i_axis_str
            else:
                target_set += ', ' + i_axis_str
        target_set += ']\n'
        f.write(target_set)

    def print_report(self, dir_path):
        try:
            with open(os.path.join(dir_path, 'report.txt'), 'r') as f:
                print(f.read())
        except:
            print('no report file found with path: {}'.format(dir_path))

    def plot_1d_psi(self, psi, psi_hjb=None, color='m', label='', dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'psi' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(0, 2 * self.alpha)

        if psi_hjb is None:
            plt.one_line_plot(x, psi, label=label)
        else:
            ys = np.vstack((psi, psi_hjb))
            colors = [color, 'c']
            labels = [label, 'num sol HJB PDE']

            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_free_energy(self, free, free_hjb=None, color=None,
                            label=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'free-energy' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(0, 2 * self.alpha)

        if free_hjb is None:
            plt.one_line_plot(x, free, color=color, label=label)
        else:
            ys = np.vstack((free, free_hjb))
            colors = [color, 'c']
            labels = [label, 'num sol HJB PDE']
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_controlled_potential(self, Vcontrolled, Vcontrolled_hjb=None, color=None,
                                     label=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'controlled-potential' + ext)
        plt.xlabel = 'x'
        plt.set_xlim(-2, 2)
        plt.set_ylim(0, 10 * self.alpha)

        if Vcontrolled_hjb is None:
            plt.one_line_plot(x, Vcontrolled, color=color, label=label)
        else:
            ys = np.vstack((Vcontrolled, Vcontrolled_hjb))
            colors = [color, 'c']
            labels = [label, 'num sol HJB PDE']
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_control(self, control, control_hjb=None, color=None,
                        label=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'control' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(- 5 * self.alpha, 5 * self.alpha)
        if control_hjb is None:
            plt.one_line_plot(x, control, color=color, label=label)
        else:
            ys = np.vstack((control, control_hjb))
            colors = [color, 'c']
            labels = [label, 'num sol HJB PDE']
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_controlled_drift(self, dVcontrolled, dVcontrolled_hjb=None, color=None,
                                 label=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'controlled-drift' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(- 5 * self.alpha, 5 * self.alpha)

        if dVcontrolled_hjb is None:
            plt.one_line_plot(x, dVcontrolled, color=color, label=label)
        else:
            ys = np.vstack((dVcontrolled, dVcontrolled_hjb))
            colors = [color, 'c']
            labels = [label, 'num sol HJB PDE']
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_free_energies(self, Fs, F_hjb=None,
                              labels=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'free-energy' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(0, 2.5 * self.alpha)

        if F_hjb is None:
            plt.multiple_lines_plot(x, Fs, labels=labels)
        else:
            ys = np.vstack((Fs, F_hjb))
            colors = [None for i in range(ys.shape[0])]
            colors[-1] = 'c'
            labels.append('num sol HJB PDE')
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_controls(self, us, u_hjb=None,
                         labels=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'control' + ext)
        plt.xlabel = 'x'
        plt.set_xlim(-2, 2)
        plt.set_ylim(-2, 5) # alpha=1, beta=1
        #plt.set_ylim(-5, 10) # alpha=4, beta=1
        #plt.set_ylim(-15, 25) # alpha=10, beta=1

        if u_hjb is None:
            plt.multiple_lines_plot(x, us, labels=labels)
        else:
            ys = np.vstack((us, u_hjb))
            colors = [None for i in range(ys.shape[0])]
            colors[-1] = 'c'
            labels.append('num sol HJB PDE')
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_1d_controlled_potentials(self, controlledVs, controlledV_hjb=None,
                                      labels=None, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        x = self.domain_h[:, 0]

        plt = Plot(dir_path, 'controlled-potential' + ext)
        plt.xlabel = 'x'
        plt.set_xlim(-2, 2)
        plt.set_ylim(0, 10) # alpha=1, beta=1
        #plt.set_ylim(0, 20) # alpha=4, beta=1
        #plt.set_ylim(0, 50) # alpha=4, beta=1

        if controlledV_hjb is None:
            plt.multiple_lines_plot(x, controlledVs, labels=labels)
        else:
            ys = np.vstack((controlledVs, controlledV_hjb))
            colors = [None for i in range(ys.shape[0])]
            colors[-1] = 'c'
            labels.append('num sol HJB PDE')
            plt.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_2d_psi(self, psi, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]

        # surface plot
        plt = Plot(dir_path, 'psi_surface' + ext)
        plt.set_title('hola')
        plt.surface(X, Y, psi)

        # contour plot
        plt = Plot(dir_path, 'psi_contour' + ext)
        plt.contour(X, Y, psi)

    def plot_2d_free_energy(self, free, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]

        # surface plot
        plt = Plot(dir_path, 'free_energy_surface' + ext)
        plt.surface(X, Y, free)

        # contour plot
        levels = np.linspace(-0.5, 3.5, 21)
        plt = Plot(dir_path, 'free_energy_contour' + ext)
        plt.set_zlim(0, 3)
        plt.contour(X, Y, free)

    def plot_2d_controlled_potential(self, Vcontrolled, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]

        # surface plot
        plt = Plot(dir_path, 'controlled_potential_surface' + ext)
        plt.set_xlim(-2, 2)
        plt.set_ylim(-2, 2)
        plt.set_zlim(0, 10)
        plt.surface(X, Y, Vcontrolled)

        # contour plot
        levels = np.logspace(-2, 1, 20, endpoint=True)
        plt = Plot(dir_path, 'controlled_potential_contour' + ext)
        plt.set_xlim(-2, 2)
        plt.set_ylim(-2, 2)
        plt.set_zlim(0, 10)
        plt.contour(X, Y, Vcontrolled, levels)

    def plot_2d_control(self, u, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = u[:, :, 0]
        V = u[:, :, 1]

        # gradient plot
        plt = Plot(dir_path, 'control' + ext)
        plt.vector_field(X, Y, U, V, scale=8)
        return

        # zoom gradient plot
        plt = Plot(dir_path, 'control_zoom_ts' + ext)
        plt.set_xlim(0, 2)
        plt.set_ylim(0, 2)
        plt.vector_field(X, Y, U, V, scale=30)

    def plot_2d_controlled_drift(self, dVcontrolled, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = dVcontrolled[:, :, 0]
        V = dVcontrolled[:, :, 1]

        plt = Plot(dir_path, 'controlled_drift' + ext)
        plt.set_xlim(-1.5, 1.5)
        plt.set_ylim(-1.5, 1.5)
        plt.vector_field(X, Y, U, V, scale=50)

    def plot_2d_control_vs_hjb(self, u, u_hjb, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = u[:, :, 0] - u_hjb[:, :, 0]
        V = u[:, :, 1] - u_hjb[:, :, 1]

        # gradient plot
        plt = Plot(dir_path, 'control_vs_hjb' + ext)
        plt.set_colormap(colormap='viridis')
        plt.vector_field(X, Y, U, V, scale=8)

    def plot_2d_controlled_drift_vs_hjb(self, dVcontrolled, dVcontrolled_hjb, dir_path=None, ext=''):
        if dir_path is None:
            dir_path = self.dir_path

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = dVcontrolled[:, :, 0] - dVcontrolled_hjb[:, :, 0]
        V = dVcontrolled[:, :, 1] - dVcontrolled_hjb[:, :, 1]

        # gradient plot
        plt = Plot(dir_path, 'controlled_drift_vs_hjb' + ext)
        plt.set_colormap(colormap='viridis')
        plt.vector_field(X, Y, U, V, scale=8)
