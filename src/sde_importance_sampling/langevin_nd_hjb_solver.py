from mds.langevin_nd_sde import LangevinSDE
from mds.utils_path import get_hjb_solution_dir_path, get_time_in_hms
from mds.utils_numeric import arange_generator

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import time

import os


class SolverHJB(LangevinSDE):
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f = 1, g = 0 and L is the infinitessimal generator
        of the not controlled n-dimensional overdamped langevin process:
            L = - ∇V·∇ + beta^{-1} Δ
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
   '''

    def __init__(self, problem_name, potential_name, n, alpha,
                 beta, h, domain=None, target_set=None, T=None):

        super().__init__(problem_name, potential_name, n, alpha, beta,
                         domain, target_set, T)

        # discretization step
        self.h = h

        # dir_path
        self.dir_path = get_hjb_solution_dir_path(self.settings_dir_path, self.h)

    def start_timer(self):
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def get_flatten_index(self, idx):
        ''' maps the bumpy index of the node (index of each axis) to
            the flatten index of the node, i.e. the node number.
        '''
        assert type(idx) == tuple, ''
        assert len(idx) == self.n, ''
        k = 0
        for i in range(self.n):
            assert 0 <= idx[i] <= self.Nx[i] - 1, ''
            Nx_prod = 1
            for j in range(i+1, self.n):
                Nx_prod *= self.Nx[j]
            k += idx[i] * Nx_prod

        return k

    def get_bumpy_index(self, k):
        ''' maps the flatten index of the node (node number) to
            the bumpy index of the node.
        '''
        assert type(k) == int, ''
        assert 0 <= k <= self.Nh -1, ''

        idx = [None for i in range(self.n)]
        for i in range(self.n):
            Nx_prod = 1
            for j in range(i+1, self.n):
                Nx_prod *= self.Nx[j]
            idx[i] = k // Nx_prod
            k -= idx[i] * Nx_prod
        return tuple(idx)

    def is_on_domain_boundary(self, idx):
        ''' returns True if the idx is on the
            boundary of the domain
        '''
        for i in range(self.n):
            if (idx[i] == 0 or
                idx[i] == self.Nx[i] - 1):
                return True
        return False

    def is_on_domain_boundary_i_axis(self, idx, i):
        ''' returns True if the idx is on the
            i axis boundary of the domain
        '''
        if (idx[i] == 0 or
            idx[i] == self.Nx[i] - 1):
            return True
        else:
            return False

    def is_on_domain_corner(self, idx):
        ''' returns True if the idx is on the corner of the rectangular boundary
        '''
        for i in range(self.n):
            if (idx[i] != 0 and
                idx[i] != self.Nx[i] - 1):
                return False
        return True

    def is_on_ts(self, idx):
        '''returns True if the idx is on the target set
        '''
        x = self.get_x(idx)
        for i in range(self.n):
            if (x[i] < self.target_set[i, 0] or
                x[i] > self.target_set[i, 1]):
                return False
        return True

    def get_flatten_idx_from_axis_neighbours(self, idx, i):

        # find flatten index of left neighbour wrt the i axis
        if idx[i] == 0:
            k_left = None
        else:
            idx_left = list(idx)
            idx_left[i] = idx[i] - 1
            k_left = self.get_flatten_index(tuple(idx_left))

        # find flatten index of right neighbour wrt the i axis
        if idx[i] == self.Nx[i] - 1:
            k_right = None
        else:
            idx_right = list(idx)
            idx_right[i] = idx[i] + 1
            k_right = self.get_flatten_index(tuple(idx_right))

        return (k_left, k_right)

    def get_flatten_idx_from_corner_neighbour(self, k):
        idx = self.get_bumpy_index(k)

        idx_inside = [None for i in range(self.n)]
        for i in range(self.n):
            if idx[i] == 0:
                idx_inside[i] = idx[i] + 1
            elif idx[i] == self.Nx[i] - 1 :
                idx_inside[i] = idx[i] - 1
        return self.get_flatten_index(tuple(idx_inside))

    def solve_bvp(self):
        # assemble linear system of equations: A \psi = b.
        A = sparse.lil_matrix((self.Nh, self.Nh))
        b = np.zeros(self.Nh)

        for k in arange_generator(self.Nh):

            # get discretized domain index
            idx = self.get_bumpy_index(k)

            # classify type of node
            is_on_ts = self.is_on_ts(idx)
            is_on_boundary = self.is_on_domain_boundary(idx)

            # assemble matrix A and vector b on S
            if not is_on_ts and not is_on_boundary:
                x = self.get_x(idx)
                grad = self.gradient(np.array([x]))[0]
                A[k, k] = - (2 * self.n) / (self.beta * self.h**2) - self.f(x)
                for i in range(self.n):
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i)
                    A[k, k_left] = 1 / (self.beta * self.h**2) + grad[i] / (2 * self.h)
                    A[k, k_right] = 1 / (self.beta * self.h**2) - grad[i] / (2 * self.h)

            # impose condition on ∂S
            elif is_on_ts and not is_on_boundary:
                A[k, k] = 1
                b[k] = np.exp(- self.g(x))

            # stability condition on the boundary
            elif is_on_boundary:
                neighbour_counter = 0
                for i in range(self.n):
                    if self.is_on_domain_boundary_i_axis(idx, i):

                        # update counter
                        neighbour_counter += 1

                        # add neighbour
                        k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i)
                        if k_left is not None:
                            A[k, k_left] = - 1
                        elif k_right is not None:
                            A[k, k_right] = - 1

                # normalize
                A[k, k] = neighbour_counter

        psi = linalg.spsolve(A.tocsc(), b)
        self.psi = psi.reshape(self.Nx)
        self.solved = True

    def compute_value_function(self):
        ''' this methos computes the value function
                value_f = - log (psi)
        '''
        assert hasattr(self, 'psi'), ''
        assert self.psi.ndim == self.n, ''
        assert self.psi.shape == self.Nx, ''

        self.value_f =  - np.log(self.psi)

    def get_idx_value_f_type(self, i):
        return [slice(self.Nx[i]) for i in range(self.n)]

    def get_idx_u_type(self, i):
        idx_u = [None for i in range(self.n + 1)]
        idx_u[-1] = i
        for j in range(self.n):
            idx_u[j] = slice(self.Nx[j])
        return idx_u

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_opt = - (√2 / beta) ∇ value_f
        '''
        assert hasattr(self, 'value_f'), ''
        assert self.value_f.ndim == self.n, ''
        assert self.value_f.shape == self.Nx, ''

        u_opt = np.zeros(self.Nx + (self.n, ))

        for i in range(self.n):

            idx_value_f_k_plus = self.get_idx_value_f_type(i)
            idx_value_f_k_minus = self.get_idx_value_f_type(i)
            idx_u_k = self.get_idx_u_type(i)
            idx_u_0 = self.get_idx_u_type(i)
            idx_u_1 = self.get_idx_u_type(i)
            idx_u_N_minus = self.get_idx_u_type(i)
            idx_u_N = self.get_idx_u_type(i)

            for j in range(self.n):
                if j == i:
                    idx_value_f_k_plus[j] = slice(2, self.Nx[j])
                    idx_value_f_k_minus[j] = slice(0, self.Nx[j] - 2)
                    idx_u_k[j] = slice(1, self.Nx[j] - 1)
                    idx_u_0[j] = 0
                    idx_u_1[j] = 1
                    idx_u_N_minus[j] = self.Nx[j] - 2
                    idx_u_N[j] = self.Nx[j] - 1
                    break

            u_opt[tuple(idx_u_k)] = - np.sqrt(2) * (1 / self.beta) *(
                self.value_f[tuple(idx_value_f_k_plus)] - self.value_f[tuple(idx_value_f_k_minus)]
            ) / (2 * self.h)
            u_opt[tuple(idx_u_0)] = u_opt[tuple(idx_u_1)]
            u_opt[tuple(idx_u_N)] = u_opt[tuple(idx_u_N_minus)]

        self.u_opt = u_opt

    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save arrays in a npz file
        np.savez(
            os.path.join(self.dir_path, 'hjb-solution.npz'),
            domain_h=self.domain_h,
            Nx=self.Nx,
            Nh=self.Nh,
            psi=self.psi,
            value_f=self.value_f,
            u_opt=self.u_opt,
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
            for attr_name in data.files:

                # get attribute from data
                if data[attr_name].ndim == 0:
                    attr = data[attr_name][()]
                else:
                    attr = data[attr_name]

                # if attribute exists check if they are the same
                if hasattr(self, attr_name):
                    assert getattr(self, attr_name) == attr

                # if attribute does not exist save attribute
                else:
                    setattr(self, attr_name, attr)
            return True

        except:
            print('no hjb-solution found with h={:.0e}'.format(self.h))
            return False

    def get_psi_at_x(self, x):
        # get index of x
        idx = self.get_index(x)

        # evaluate psi at idx
        return self.psi[idx] if hasattr(self, 'psi') else None

    def get_value_function_at_x(self, x):
        # get index of x
        idx = self.get_index(x)

        # evaluate psi at idx
        return self.value_f[idx] if hasattr(self, 'value_f') else None

    def get_u_opt_at_x(self, x):
        # get index of x
        idx = self.get_index(x)

        # evaluate psi at idx
        return self.u_opt[idx] if hasattr(self, 'u_opt') else None

    def get_controlled_potential_and_drift(self):

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.n)

        # potential, bias potential and tilted potential
        V = self.potential(x).reshape(self.Nx)
        self.bias_potential = 2 * self.value_f / self.beta
        self.controlled_potential = V + self.bias_potential

        # gradient and tilted drift
        dV = self.gradient(x).reshape(self.domain_h.shape)
        self.controlled_drift = - dV + np.sqrt(2) * self.u_opt

    def write_report(self, x):

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write file
        f = open(file_path, 'w')

        # space discretization
        f.write('\n space discretization\n')
        f.write('h = {:2.4f}\n'.format(self.h))
        f.write('N_h = {:d}\n'.format(self.Nh))

        # psi, value function and control
        f.write('\n psi, value function and optimal control at x\n')

        psi = self.get_psi_at_x(x)
        value_f = self.get_value_function_at_x(x)
        u_opt = self.get_u_opt_at_x(x)

        x_str = 'x: ('
        for i in range(self.n):
            if i == 0:
                x_str += '{:2.1f}'.format(x[i])
            else:
                x_str += ', {:2.1f}'.format(x[i])
        x_str += ')\n'

        u_opt_str = 'u_opt(x) = ('
        for i in range(self.n):
            if i == 0:
                u_opt_str += '{:2.1f}'.format(u_opt[i])
            else:
                u_opt_str += ', {:2.1f}'.format(u_opt[i])
        u_opt_str += ')\n'

        f.write(x_str)
        if psi is not None:
            f.write('psi(x) = {:2.3e}\n'.format(psi))

        if value_f is not None:
            f.write('value_f(x) = {:2.3e}\n'.format(value_f))

        # TODO! write general method in utils to write np.array into string
        f.write(u_opt_str)

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        f.write('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def plot_1d_psi(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='psi',
        )
        x = self.domain_h[:, 0]
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        fig.plot(x, self.psi, labels='num sol HJB PDE', colors='tab:cyan')

    def plot_1d_value_function(self, ylim=None):
        from figures.myfigure import MyFigure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='value-function',
        )
        x = self.domain_h[:, 0]
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])
        fig.plot(x, self.value_f, labels='num sol HJB PDE', colors='tab:cyan')

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

    def plot_2d_psi(self):
        from figures.myfigure import MyFigure

        # contour plot
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='psi',
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        #fig.set_contour_levels_scale('log')
        fig.contour(X, Y, self.psi)

        # surface plot


    def plot_2d_value_function(self):
        from figures.myfigure import MyFigure

        # contour plot
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='value-function',
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_contour_levels_scale('log')
        fig.contour(X, Y, self.value_f)

        # surface plot


    def plot_2d_controlled_potential(self):
        from figures.myfigure import MyFigure

        # contour plot
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='controlled-potential',
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_contour_levels_scale('log')
        fig.contour(X, Y, self.controlled_potential)

        # surface plot


    def plot_2d_control(self, scale=None, width=0.005):
        from figures.myfigure import MyFigure

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = self.u_opt[:, :, 0]
        V = self.u_opt[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='control',
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.vector_field(X, Y, U, V, scale=scale, width=width)


    def plot_2d_controlled_drift(self, scale=None, width=0.005):
        from figures.myfigure import MyFigure

        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = self.controlled_drift[:, :, 0]
        V = self.controlled_drift[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='controlled-drift',
        )
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        fig.vector_field(X, Y, U, V, scale=scale, width=width)
