from mds.potentials_and_gradients_nd import get_potential_and_gradient
from mds.utils import get_example_dir_path, get_time_in_hms

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import time

from inspect import isfunction

import os

class Solver():
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not drifted 2d overdamped langevin process:
            L = - ∇V·∇ + epsilon Δ
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
   '''

    def __init__(self, n, f, g, potential_name, alpha, beta, h, target_set=None, domain=None):

        # get potential and gradient functions
        potential, gradient, _ = get_potential_and_gradient(n, potential_name, alpha)

        # domain and target set
        if domain is None:
            domain = np.full((n, 2), [-3, 3])
        if target_set is None:
            target_set = np.full((n, 2), [1, 3])

        self.n = n
        self.f = f
        self.g = g
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.domain = domain
        self.target_set = target_set

        # discretized domain
        self.h = h
        self.domain_h = None
        self.Nx = None
        self.Nh = None

        # discretized solution
        self.Psi = None
        self.F = None
        self.u_opt = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # dir_path
        self.dir_path = get_example_dir_path(potential_name, n, alpha, beta,
                                             'hypercube', 'hjb-solution')


    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def discretize_domain(self):
        ''' this method discretizes the rectangular domain uniformly with step-size h
        '''
        assert self.h is not None, ''

        # construct not sparse nd grid
        mgrid_input = []
        for i in range(self.n):
            mgrid_input.append(
                slice(self.domain[i, 0], self.domain[i, 1] + self.h, self.h)
            )
        self.domain_h = np.mgrid[mgrid_input]

        # check shape
        assert self.domain_h.shape[0] == self.n, ''

        # save number of indices per axis
        self.Nx = self.domain_h.shape[1:]

        # save number of flatten indices
        N = 1
        for i in range(self.n):
            N *= self.Nx[i]
        self.Nh = N

    def get_index(self, x):
        assert x.ndim == 1, ''
        assert x.shape[0] == self.n, ''

        idx = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], self.Nx[i])
            idx[i] = np.argmin(np.abs(axis_i - x[i]))

        return tuple(idx)

    def get_flatten_index(self, idx):
        assert type(idx) == tuple, ''
        assert len(idx) == self.n, ''
        k = 0
        for i in range(self.n):
            assert idx[i] in np.arange(self.Nx[i]), ''
            Nx_prod = 1
            for j in range(i+1, self.n):
                Nx_prod *= self.Nx[j]
            k += idx[i] * Nx_prod

        return k

    def get_bumpy_index(self, k):
        assert k in np.arange(self.Nh), ''

        idx = [None for i in range(self.n)]
        for i in range(self.n):
            Nx_prod = 1
            for j in range(i+1, self.n):
                Nx_prod *= self.Nx[j]
            idx[i] = k // Nx_prod
            k -= idx[i] * Nx_prod
        return tuple(idx)

    def get_x(self, k):
        ''' returns the coordinates of the node k
        '''
        idx = self.get_bumpy_index(k)
        idx_domain_h = [np.arange(self.n)]
        for i in range(self.n):
            idx_domain_h.append(np.array(idx[i]))
        idx_domain_h = tuple(idx_domain_h)

        return self.domain_h[idx_domain_h]

    def is_on_domain_boundary(self, k):
        ''' returns True if the node k is on the
            boundary of the domain
        '''
        idx = self.get_bumpy_index(k)
        for i in range(self.n):
            if (idx[i] == 0 or
                idx[i] == self.Nx[i] - 1):
                return True
        return False

    def is_on_domain_corner(self, k):
        ''' returns True if the node k is on the corner of the rectangular boundary
        '''
        idx = self.get_bumpy_index(k)
        for i in range(self.n):
            if (idx[i] != 0 and
                idx[i] != self.Nx[i] - 1):
                return False
        return True

    def is_on_ts(self, k):
        '''returns True if the node k is on the target set
        '''
        x = self.get_x(k)
        for i in range(self.n):
            if (x[i] < self.target_set[i, 0] or
                x[i] > self.target_set[i, 1]):
                return False
        return True

    def get_flatten_idx_from_axis_neighbours(self, k, i):
        idx = self.get_bumpy_index(k)

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
        # assemble linear system of equations: A \Psi = b.
        A = sparse.lil_matrix((self.Nh, self.Nh))
        b = np.zeros(self.Nh)

        # nodes in boundary, boundary corner and target set
        idx_boundary = np.array([k for k in np.arange(self.Nh) if self.is_on_domain_boundary(k)])
        idx_ts = np.array([k for k in np.arange(self.Nh) if self.is_on_ts(k)])

        for k in np.arange(self.Nh):
            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)
                grad = self.gradient(np.array([x]))[0]
                A[k, k] = - (2 * self.n) / (self.beta * self.h**2) - self.f(x)
                for i in range(self.n):
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(k, i)
                    A[k, k_left] = 1 / (self.beta * self.h**2) + grad[i] / (2 * self.h)
                    A[k, k_right] = 1 / (self.beta * self.h**2) - grad[i] / (2 * self.h)

            # impose condition on ∂S
            elif k in idx_ts and k not in idx_boundary:
                A[k, k] = 1
                b[k] = np.exp(- self.g(x))

            # stability condition on the boundary
            for i in range(self.n):
                for k in idx_boundary:
                    # index on the boundary of the i hyperplane
                    idx_bumpy = self.get_bumpy_index(k)
                    if not (idx_bumpy[i] == 0 or
                            idx_bumpy[i] == self.Nx[i] - 1):
                        continue
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(k, i)
                    if k_left is not None:
                        A[k, k] = 1
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k] = 1
                        A[k, k_right] = - 1

        Psi = linalg.spsolve(A.tocsc(), b)
        self.Psi = Psi.reshape(self.Nx)

    def compute_free_energy(self):
        ''' this methos computes the free energy
                F = - epsilon log (Psi)
        '''
        assert self.Psi is not None, ''
        assert self.Psi.ndim == self.n, ''
        assert self.Psi.shape == self.Nx, ''

        self.F =  - np.log(self.Psi)

    def get_idx_u_type(self, i):
        idx_u = [None for i in range(self.n + 1)]
        idx_u[0] = i
        for j in range(self.n):
            idx_u[j+1] = slice(self.Nx[j])
        return idx_u

    def get_idx_F_type(self, i):
        return [slice(self.Nx[i]) for i in range(self.n)]

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_opt = - √2 ∇F
        '''
        assert self.F is not None, ''
        assert self.F.ndim == self.n, ''
        assert self.F.shape == self.Nx, ''

        u_opt = np.zeros((self.n, ) + self.Nx)

        for i in range(self.n):

            idx_F_k_plus = self.get_idx_F_type(i)
            idx_F_k_minus = self.get_idx_F_type(i)
            idx_u_k = self.get_idx_u_type(i)
            idx_u_0 = self.get_idx_u_type(i)
            idx_u_1 = self.get_idx_u_type(i)
            idx_u_N_minus = self.get_idx_u_type(i)
            idx_u_N = self.get_idx_u_type(i)

            for j in range(self.n):
                if j == i:
                    idx_F_k_plus[j] = slice(2, self.Nx[j])
                    idx_F_k_minus[j] = slice(0, self.Nx[j] - 2)
                    idx_u_k[j+1] = slice(1, self.Nx[j] - 1)
                    idx_u_0[j+1] = 0
                    idx_u_1[j+1] = 1
                    idx_u_N_minus[j+1] = self.Nx[j] - 2
                    idx_u_N[j+1] = self.Nx[j] - 1
                    break

            u_opt[tuple(idx_u_k)] = - np.sqrt(2) * (
                self.F[tuple(idx_F_k_plus)] - self.F[tuple(idx_F_k_minus)]
            ) / (2 * self.h)
            u_opt[tuple(idx_u_0)] = u_opt[tuple(idx_u_1)]
            u_opt[tuple(idx_u_N)] = u_opt[tuple(idx_u_N_minus)]

        self.u_opt = u_opt

    def save_reference_solution(self):
        # file name
        h_ext = '_h{:.0e}'.format(self.h)
        file_name = 'reference_solution' + h_ext + '.npz'

        np.savez(
            os.path.join(self.dir_path, file_name),
            domain_h=self.domain_h,
            Psi=self.Psi,
            F=self.F,
            u_opt=self.u_opt,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_reference_solution(self):
        # file name
        h_ext = '_h{:.0e}'.format(self.h)
        file_name = 'reference_solution' + h_ext + '.npz'

        ref_sol = np.load(
            os.path.join(self.dir_path, file_name),
            allow_pickle=True,
        )
        self.domain_h = hjb_sol['domain_h']
        self.Psi = hjb_sol['Psi']
        self.F = hjb_sol['F']
        self.u_opt = hjb_sol['u_opt']
        self.t_initial = hjb_sol['t_initial']
        self.t_final = hjb_sol['t_final']

    def write_report(self, x):
        # get index of x
        idx = self.get_index(x)
        idx_Psi = tuple([idx[i] for i in range(self.n)])

        # Psi at x
        Psi = self.Psi[idx_Psi] if self.Psi is not None else np.nan
        F = self.F[idx_Psi] if self.F is not None else np.nan

        # file name
        h_ext = '_h{:.0e}'.format(self.h)
        file_name = 'report' + h_ext + '.txt'

        # write file
        f = open(os.path.join(self.dir_path, file_name), "w")

        f.write('h = {:2.4f}\n'.format(self.h))
        f.write('N_h = {:d}\n'.format(self.Nh))

        posicion = 'x: ('
        for i in range(self.n):
            if i == 0:
                posicion += '{:2.1f}'.format(x[i])
            else:
                posicion += ', {:2.1f}'.format(x[i])
        posicion += ')\n'
        f.write(posicion)

        f.write('Psi at x = {:2.3e}\n'.format(Psi))
        f.write('F at x = {:2.3e}\n'.format(F))
        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))
        f.close()
