from mds.potentials_and_gradients_nd import get_potential_and_gradient
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
    def discretize_domain(self):
        ''' this method discretizes the hyper-rectangular domain uniformly with step-size h
        '''
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

