from mds.functions import quadratic_one_well, double_well, double_well_gradient
from mds.langevin_nd_sde import LangevinSDE
from mds.utils_path import get_hjb_solution_dir_path, get_time_in_hms
from mds.utils_numeric import arange_generator

import functools
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.linalg import solve_banded
import time

import os


class SolverHJBDet(LangevinSDE):
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            \partial_t Ψ  = LΨ − f Ψ  for all x \in \R^n, t \in [t, T)
            Ψ(T, x) = exp(− g(x)) for all x \in \R^n
        where f = 0, g = quadratic one well and L is the infinitessimal generator
        of the not controlled n-dimensional overdamped langevin process:
            L = - ∇V·∇ + epsilon Δ
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
   '''

    def __init__(self, problem_name, potential_name, n, alpha,
                 beta, h, domain=None, target_set=None, T=None, nu=None, dt=None):

        super().__init__(problem_name, potential_name, n, alpha, beta,
                         domain, target_set, T, nu)

        # discretization step
        self.h = h

        # time discretization step
        self.dt = dt

        # set potential_i and gradient_i
        self.set_potential_and_gradient_i()

        # set g_i
        self.set_g_i()

        # dir_path
        self.dir_path = get_hjb_solution_dir_path(self.settings_dir_path, self.h)

    def set_potential_and_gradient_i(self):
        '''
        '''
        self.potential_i = functools.partial(double_well, alpha=self.alpha[:1])
        self.gradient_i = functools.partial(double_well_gradient, alpha=self.alpha[:1])

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def discretize_domain(self, h=None):
        ''' discretize 1d reduced domain
        '''
        if h is not None:
            self.h = h
        assert self.h is not None, ''

        self.x = np.arange(
            self.domain[0, 0],
            self.domain[0, 1] + self.h,
            self.h,
        )
        self.x = np.expand_dims(self.x, axis=1)
        self.Nh = self.x.shape[0]

    def set_g_i(self):
        '''
        '''
        self.g_i = functools.partial(quadratic_one_well, nu=3 * np.ones(1))

    def preallocate_psi_i(self):
        ''' preallocate psi_i at each time step
        '''
        self.K = int(self.T / self.dt)
        self.psi_i = np.empty((self.K + 1, + self.Nh))

    def preallocate_u_opt_i(self):
        ''' preallocate u_opt_i at each time step
        '''
        self.u_opt_i = np.zeros((self.K + 1, + self.Nh))

    def solve_bvp_det(self, i=0):
        ''' for the ith dimension
        '''

        # flat domain
        x = self.domain_h.reshape(self.Nh, self.n)

        # evaluate psi at T
        psi_T_flat = np.exp(- self.g(x))
        psi_T = psi_T_flat.reshape(self.Nx)
        self.Psi[K, :] = psi_T

        # reverse loop over the time step indices
        for l in range(K - 1, -1, -1):

            #if l // 10 == 1:
            print(l)

            # assemble linear system of equations: A \Psi = b.
            A = sparse.lil_matrix((self.Nh, self.Nh))
            b = np.zeros(self.Nh)

            for k in arange_generator(self.Nh):
                #print('k={}'.format(k))

                # get discretized domain index
                idx = self.get_bumpy_index(k)

                # classify type of node
                is_on_boundary = self.is_on_domain_boundary(idx)

                # assemble matrix A and vector b on the domain
                if not is_on_boundary:
                    x = self.get_x(idx)
                    grad = self.gradient(x)[0]
                    A[k, k] = 1 - self.dt * 2 / (self.beta * self.h**2) - self.dt * self.f(x)
                    A[k, k - 1] = self.dt / (self.beta * self.h**2) + self.dt * grad / (2 * self.h)
                    A[k, k + 1] = self.dt / (self.beta * self.h**2) - self.dt * grad / (2 * self.h)

                    psi_l_plus_flat = self.Psi[l + 1, :].reshape(self.Nh)
                    b[k] = psi_l_plus_flat[k]

                # assemble matrix A and vector b on the boundary
                else:
                    if k == 0:
                        #A[k, k + 1] = - 1
                        A[k, k + 1] = - self.dt
                    elif k == self.Nh -1:
                        #A[k, k - 1] = - 1
                        A[k, k - 1] = - self.dt
                    #b[k] = 0
                    psi_l_plus_flat = self.Psi[l + 1, :].reshape(self.Nh)
                    b[k] = psi_l_plus_flat[k]

            # solve linear system
            breakpoint()
            self.psi_i[l, :] = linalg.spsolve(A.tocsc(), b)

        self.solved = True

    def solve_bvp_eigenproblem(self):
        ''' solve bvp eigenproblem
        '''
        # lower bound
        lb = self.x[0, 0]

        # A
        A = np.zeros([self.Nh, self.Nh])
        for k in arange_generator(self.Nh):

            x = lb + (k + 0.5) * self.h
            if k > 0:
                x0 = lb + (k - 0.5) * self.h
                x1 = lb + k * self.h
                A[k, k - 1] = - np.exp(
                    self.beta * 0.5 * (
                        self.potential_i(x0) + self.potential_i(x) - 2 * self.potential_i(x1)
                    )
                ) / self.h ** 2
                A[k, k] = np.exp(
                    self.beta * (self.potential_i(x) - self.potential_i(x1))
                ) / self.h**2

            if k < self.Nh - 1:
                x0 = lb + (k + 1.5) * self.h
                x1 = lb + (k + 1) * self.h
                A[k, k + 1] = - np.exp(
                    self.beta * 0.5 * (
                        self.potential_i(x0) + self.potential_i(x) - 2 * self.potential_i(x1)
                    )
                ) / self.h ** 2
                A[k, k] = A[k, k] + np.exp(
                    self.beta * (self.potential_i(x) - self.potential_i(x1))
                ) / self.h ** 2

        A = - A / self.beta

        D = np.diag(np.exp(self.beta * self.potential_i(self.x) / 2))
        D_inv = np.diag(np.exp(-self.beta * self.potential_i(self.x) / 2))

        np.linalg.cond(np.eye(self.Nh) - self.dt * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.dt * A)

        self.psi_i[self.K, :] = np.exp(- self.g_i(self.x))

        for l in range(self.K - 1, -1, -1):
            band = - self.dt * np.vstack([
                np.append([0], np.diagonal(A, offset=1)),
                np.diagonal(A, offset=0) - self.K / self.T,
                np.append(np.diagonal(A, offset=1), [0])
            ])

            self.psi_i[l, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi_i[l + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - dt * A, D_inv.dot(psi[n + 1, :])));

        self.solved = True

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_i^*(t, x_i) = - √2 / beta^{-1} ∂/∂xi (− log (Ψ_i(t, x_i)))
        '''
        assert hasattr(self, 'psi_i'), ''
        assert self.psi_i.ndim == 2, ''
        #assert self.psi_i.shape[0] == self.Nh, ''

        for l in range(self.K + 1):
            for k in range(self.Nh - 1):
                self.u_opt_i[l, k] = - (np.sqrt(2) / self.beta) * (
                    - np.log(self.psi_i[l, k + 1])
                    + np.log(self.psi_i[l, k])
                ) / self.h
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)


    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save arrays in a npz file
        np.savez(
            os.path.join(self.dir_path, 'hjb-solution.npz'),
            h=self.h,
            Nh=self.Nh,
            x=self.x,
            T=self.T,
            dt=self.dt,
            K=self.K,
            psi_i=self.psi_i,
            u_opt_i=self.u_opt_i,
            ct=self.ct,
        )

    def load(self):
        ''' loads the saved arrays and sets them as attributes back
        '''
        try:
            data = np.load(
                os.path.join(self.dir_path, 'hjb-solution.npz'),
                allow_pickle=True,
            )
            for file_name in data.files:
                if hasattr(self, file_name):
                    assert getattr(self, file_name) == data[file_name]
                else:
                    setattr(self, file_name, data[file_name])
            return True

        except:
            print('no hjb-solution found with h={:.0e}'.format(self.h))
            return False

    def get_time_index(self, t):
        ''' returns time index for the given time t
        '''
        assert 0 <= t <= self.T, ''

        return int(np.ceil(t / self.dt))

    def get_space_index(self, x):
        ''' returns the index of the point of the grid closest to x
        '''
        assert x.ndim == 1, ''
        assert x.shape[0] == self.n, ''

        idx = [None for i in range(self.n)]
        axis_i = self.x[:, 0]
        for i in range(self.n):
            idx[i] = np.argmin(np.abs(axis_i - x[i]))
        return tuple(idx)

    def get_space_index_vectorized(self, x):
        ''' returns the index of the point of the grid closest to x
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        # get index of xzero
        idx = [None for i in range(self.n)]
        axis_i = self.x[:, 0]
        for i in range(self.n):
            idx[i] = tuple(np.argmin(np.abs(axis_i - x[:, i].reshape(x.shape[0], 1)), axis=1))
        return tuple(idx)

    def get_psi_t_x(self, t, x):
        # get time index
        l = self.get_time_index(t)

        # get index of x
        idx = self.get_space_index(x)

        # evaluate psi at idx
        psi = 1.
        for i in range(self.n):
            psi *= self.psi_i[l, idx[i]]

        return psi

    def get_f_t_x(self, t, x):
        # get time index
        l = self.get_time_index(t)

        # get index of x
        idx = self.get_space_index(x)

        # evaluate F at idx
        F = 0.
        for i in range(self.n):
            F += - np.log(self.psi_i[l, idx[i]])

        return F

    def get_u_opt_t_x(self, t, x):
        # get time index
        l = self.get_time_index(t)

        # get index of x
        idx = self.get_space_index(x)

        # evaluate F at idx
        u_opt = np.empty(self.n)
        for i in range(self.n):
            u_opt[i] = self.u_opt_i[l, idx[i]]

        return u_opt

    def write_report(self, t, x):

        # psi and F at x
        psi = self.get_psi_t_x(t, x)
        F = self.get_f_t_x(t, x)
        u = self.get_u_opt_t_x(t, x)

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write file
        f = open(file_path, 'w')

        f.write('\n space discretization\n')
        f.write('h = {:2.4f}\n'.format(self.h))
        f.write('N_h = {:d}\n'.format(self.Nh))

        f.write('\n time discretization\n')
        f.write('T = {:2.4f}\n'.format(self.T))
        f.write('dt = {:2.4f}\n'.format(self.dt))
        f.write('K = {:d}\n'.format(self.K))

        f.write('\n psi and value function at x\n')
        f.write('t = {:2.4f}\n'.format(t))
        x_str = 'x: ('
        for i in range(self.n):
            if i == 0:
                x_str += '{:2.1f}'.format(x[i])
            else:
                x_str += ', {:2.1f}'.format(x[i])
        x_str += ')\n'
        f.write(x_str)

        f.write('psi(t, x) = {:2.3e}\n'.format(psi))
        f.write('F(t, x) = {:2.3e}\n'.format(F))
        u_opt_str = 'u_opt(t, x) = ('
        for i in range(self.n):
            if i == 0:
                u_opt_str += '{:2.1f}'.format(u[i])
            else:
                u_opt_str += ', {:2.1f}'.format(u[i])
        u_opt_str += ')\n'
        f.write(u_opt_str)

        h, m, s = get_time_in_hms(self.ct)
        f.write('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def plot_1d_psi_i(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='psi_i',
        )
        x = self.x[:, 0]
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        t_0 = 0.
        idx_0 = self.get_time_index(0)
        t_1 = self.T
        idx_1 = self.get_time_index(t_1)
        y = np.vstack((
            self.psi_i[idx_0, :],
            self.psi_i[idx_1, :],
        ))
        colors = [
            'tab:cyan',
            'tab:orange',
        ]
        labels = [
            'HJB (t = {:2.2f})'.format(t_0),
            'HJB (t = {:2.2f})'.format(t_1),
        ]
        fig.plot(x, y, labels=labels, colors=colors)

    def plot_1d_free_energy(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='free-energy',
        )
        x = self.domain_h[:, 0]
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        fig.plot(x, self.F, labels='num sol HJB PDE', colors='tab:cyan')

    def plot_1d_controlled_potential(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='controlled-potential',
        )
        x = self.domain_h[:, 0]
        if self.controlled_potential is None:
            self.get_controlled_potential_and_drift()

        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        fig.plot(x, self.controlled_potential, labels='num sol HJB PDE', colors='tab:cyan')

    def plot_1d_control(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='control',
        )
        x = self.domain_h[:, 0]
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        fig.plot(x, self.u_opt[:, 0], labels='num sol HJB PDE', colors='tab:cyan')

    def plot_1d_controlled_drift(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='controlled-drift',
        )
        x = self.domain_h[:, 0]
        if self.controlled_drift is None:
            self.get_controlled_potential_and_drift()
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        fig.plot(x, self.controlled_drift[:, 0], labels='num sol HJB PDE', colors='tab:cyan')
