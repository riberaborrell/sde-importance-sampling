from mds.potentials_and_gradients_nd import get_potential_and_gradient
from mds.plots import Plot
from mds.utils import get_example_dir_path, \
                      get_metadynamics_dir_path

import numpy as np

import os

class LangevinSDE:
    '''
    '''

    def __init__(self, n, potential_name, alpha, beta,
                 target_set=None, domain=None, h=None):
        '''
        '''
        # get potential and gradient functions
        potential, gradient, _ = get_potential_and_gradient(n, potential_name, alpha)

        # domain and target set
        if domain is None:
            domain = np.full((n, 2), [-3, 3])
        if target_set is None:
            target_set = np.full((n, 2), [1, 3])

        # sde parameters
        self.n = n
        self.potential_name = potential_name
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
        self.example_dir_path = None
        self.set_example_dir_path()

    def set_example_dir_path(self):
        assert (self.alpha == self.alpha[0]).all(), ''
        self.example_dir_path = get_example_dir_path(self.potential_name, self.n,
                                                     self.alpha[0], self.beta, 'hypercube')
    def discretize_domain(self, h=None):
        ''' this method discretizes the hyper-rectangular domain uniformly with step-size h
        '''
        if h is not None:
            self.h = h
        assert self.h is not None, ''

        # construct not sparse nd grid
        mgrid_input = []
        for i in range(self.n):
            mgrid_input.append(
                slice(self.domain[i, 0], self.domain[i, 1] + self.h, self.h)
            )
        self.domain_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)

        # check shape
        assert self.domain_h.shape[-1] == self.n, ''

        # save number of indices per axis
        self.Nx = self.domain_h.shape[:-1]

        # save number of flatten indices
        N = 1
        for i in range(self.n):
            N *= self.Nx[i]
        self.Nh = N

    def get_flatted_domain_h(self):
        ''' this method returns the flatten discretized domain
        '''
        breakpoint()
        flat_shape = tuple(self.N, self.n)
        x = self.domain_h.reshape(self.N, 2)

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

    def get_hjb_solver(self, h):
        from mds.langevin_nd_hjb_solver import Solver
        # initialize hjb solver
        sol = Solver(
            n=self.n,
            potential_name=self.potential_name,
            alpha=self.alpha,
            beta=self.beta,
            h=h,
        )

        # load already computed solution
        sol.load_hjb_solution()
        return sol

    def get_meta_bias_potential(self, sigma_i_meta, k, N_meta):
        try:
            meta_dir_path = get_metadynamics_dir_path(
                self.example_dir_path,
                sigma_i_meta,
                k,
                N_meta,
            )
            file_path = os.path.join(meta_dir_path, 'bias-potential.npz')
            meta_bias_pot = np.load(file_path)
            return meta_bias_pot
        except:
            print('no metadynamics-sampling found with sigma_i_meta={:.0e}, k={}, meta_N:{}'
                  ''.format(sigma_i_meta, k, N_meta))

    def get_not_controlled(self, N):
        try:
            dir_path = os.path.join(
                self.example_dir_path,
                'mc-sampling',
                'N_{:.0e}'.format(N),
            )
            file_path = os.path.join(dir_path, 'mc-sampling.npz')
            mcs = np.load(file_path, allow_pickle=True)
            return mcs
        except:
            print('no mc-sampling found with N={:.0e}'.format(N))

    def write_setting(self, f):
        '''
        '''
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
        target_set += ']\n\n'
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

        plt = Plot(dir_path, 'free_energy' + ext)
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

        plt = Plot(dir_path, 'controlled_potential' + ext)
        plt.xlabel = 'x'
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

        plt = Plot(dir_path, 'controlled_drift' + ext)
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

        plt = Plot(dir_path, 'free_energies' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(0, self.alpha * 2)

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

        plt = Plot(dir_path, 'controls' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(- 5 * self.alpha, 5 * self.alpha)

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

        plt = Plot(dir_path, 'controlled_potentials' + ext)
        plt.xlabel = 'x'
        plt.set_ylim(0, 10 * self.alpha)

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
