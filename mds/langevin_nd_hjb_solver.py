from mds.potentials_and_gradients_nd import get_potential_and_gradient
from mds.utils import get_example_dir_path, get_hjb_solution_dir_path, get_time_in_hms

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

    def __init__(self, sde, f, g):

        # langevin sde
        self.sde = sde

        # work functional
        self.f = f
        self.g = g

        # discretized solution
        self.Psi = None
        self.F = None
        self.u_opt = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # dir_path
        self.dir_path = get_hjb_solution_dir_path(self.sde.example_dir_path, self.sde.h)

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def get_flatten_index(self, idx):
        ''' maps the bumpy index of the node (index of each axis) to
            the flatten index of the node, i.e. the node number.
        '''
        assert type(idx) == tuple, ''
        assert len(idx) == self.sde.n, ''
        k = 0
        for i in range(self.sde.n):
            assert idx[i] in np.arange(self.sde.Nx[i]), ''
            Nx_prod = 1
            for j in range(i+1, self.sde.n):
                Nx_prod *= self.sde.Nx[j]
            k += idx[i] * Nx_prod

        return k

    def get_bumpy_index(self, k):
        ''' maps the flatten index of the node (node number) to
            the bumpy index of the node.
        '''
        assert k in np.arange(self.sde.Nh), ''

        idx = [None for i in range(self.sde.n)]
        for i in range(self.sde.n):
            Nx_prod = 1
            for j in range(i+1, self.sde.n):
                Nx_prod *= self.sde.Nx[j]
            idx[i] = k // Nx_prod
            k -= idx[i] * Nx_prod
        return tuple(idx)

    def is_on_domain_boundary(self, k):
        ''' returns True if the node k is on the
            boundary of the domain
        '''
        idx = self.get_bumpy_index(k)
        for i in range(self.sde.n):
            if (idx[i] == 0 or
                idx[i] == self.sde.Nx[i] - 1):
                return True
        return False

    def is_on_domain_corner(self, k):
        ''' returns True if the node k is on the corner of the rectangular boundary
        '''
        idx = self.get_bumpy_index(k)
        for i in range(self.sde.n):
            if (idx[i] != 0 and
                idx[i] != self.sde.Nx[i] - 1):
                return False
        return True

    def is_on_ts(self, k):
        '''returns True if the node k is on the target set
        '''
        idx = self.get_bumpy_index(k)
        x = self.sde.get_x(idx)
        for i in range(self.sde.n):
            if (x[i] < self.sde.target_set[i, 0] or
                x[i] > self.sde.target_set[i, 1]):
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
        if idx[i] == self.sde.Nx[i] - 1:
            k_right = None
        else:
            idx_right = list(idx)
            idx_right[i] = idx[i] + 1
            k_right = self.get_flatten_index(tuple(idx_right))

        return (k_left, k_right)

    def get_flatten_idx_from_corner_neighbour(self, k):
        idx = self.get_bumpy_index(k)

        idx_inside = [None for i in range(self.sde.n)]
        for i in range(self.sde.n):
            if idx[i] == 0:
                idx_inside[i] = idx[i] + 1
            elif idx[i] == self.sde.Nx[i] - 1 :
                idx_inside[i] = idx[i] - 1
        return self.get_flatten_index(tuple(idx_inside))

    def solve_bvp(self):
        # assemble linear system of equations: A \Psi = b.
        A = sparse.lil_matrix((self.sde.Nh, self.sde.Nh))
        b = np.zeros(self.sde.Nh)

        # nodes in boundary, boundary corner and target set
        idx_boundary = np.array(
            [k for k in np.arange(self.sde.Nh) if self.is_on_domain_boundary(k)]
        )
        idx_ts = np.array([k for k in np.arange(self.sde.Nh) if self.is_on_ts(k)])

        for k in np.arange(self.sde.Nh):
            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                idx = self.get_bumpy_index(k)
                x = self.sde.get_x(idx)
                grad = self.sde.gradient(np.array([x]))[0]
                A[k, k] = - (2 * self.sde.n) / (self.sde.beta * self.sde.h**2) - self.f(x)
                for i in range(self.sde.n):
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(k, i)
                    A[k, k_left] = 1 / (self.sde.beta * self.sde.h**2) + grad[i] / (2 * self.sde.h)
                    A[k, k_right] = 1 / (self.sde.beta * self.sde.h**2) - grad[i] / (2 * self.sde.h)

            # impose condition on ∂S
            elif k in idx_ts and k not in idx_boundary:
                A[k, k] = 1
                b[k] = np.exp(- self.g(x))

            # stability condition on the boundary
            for i in range(self.sde.n):
                for k in idx_boundary:
                    # index on the boundary of the i hyperplane
                    idx = self.get_bumpy_index(k)
                    if not (idx[i] == 0 or
                            idx[i] == self.sde.Nx[i] - 1):
                        continue
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(k, i)
                    if k_left is not None:
                        A[k, k] = 1
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k] = 1
                        A[k, k_right] = - 1

        Psi = linalg.spsolve(A.tocsc(), b)
        self.Psi = Psi.reshape(self.sde.Nx)

    def compute_free_energy(self):
        ''' this methos computes the free energy
                F = - epsilon log (Psi)
        '''
        assert self.Psi is not None, ''
        assert self.Psi.ndim == self.sde.n, ''
        assert self.Psi.shape == self.sde.Nx, ''

        self.F =  - np.log(self.Psi)

    def get_idx_F_type(self, i):
        return [slice(self.sde.Nx[i]) for i in range(self.sde.n)]

    def get_idx_u_type(self, i):
        idx_u = [None for i in range(self.sde.n + 1)]
        idx_u[-1] = i
        for j in range(self.sde.n):
            idx_u[j] = slice(self.sde.Nx[j])
        return idx_u

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_opt = - √2 ∇F
        '''
        assert self.F is not None, ''
        assert self.F.ndim == self.sde.n, ''
        assert self.F.shape == self.sde.Nx, ''

        u_opt = np.zeros(self.sde.Nx + (self.sde.n, ))

        for i in range(self.sde.n):

            idx_F_k_plus = self.get_idx_F_type(i)
            idx_F_k_minus = self.get_idx_F_type(i)
            idx_u_k = self.get_idx_u_type(i)
            idx_u_0 = self.get_idx_u_type(i)
            idx_u_1 = self.get_idx_u_type(i)
            idx_u_N_minus = self.get_idx_u_type(i)
            idx_u_N = self.get_idx_u_type(i)

            for j in range(self.sde.n):
                if j == i:
                    idx_F_k_plus[j] = slice(2, self.sde.Nx[j])
                    idx_F_k_minus[j] = slice(0, self.sde.Nx[j] - 2)
                    idx_u_k[j] = slice(1, self.sde.Nx[j] - 1)
                    idx_u_0[j] = 0
                    idx_u_1[j] = 1
                    idx_u_N_minus[j] = self.sde.Nx[j] - 2
                    idx_u_N[j] = self.sde.Nx[j] - 1
                    break

            u_opt[tuple(idx_u_k)] = - np.sqrt(2) * (
                self.F[tuple(idx_F_k_plus)] - self.F[tuple(idx_F_k_minus)]
            ) / (2 * self.sde.h)
            u_opt[tuple(idx_u_0)] = u_opt[tuple(idx_u_1)]
            u_opt[tuple(idx_u_N)] = u_opt[tuple(idx_u_N_minus)]

        self.u_opt = u_opt

    def save_hjb_solution(self):
        np.savez(
            os.path.join(self.dir_path, 'hjb-solution.npz'),
            domain_h=self.sde.domain_h,
            Nx=self.sde.Nx,
            Nh=self.sde.Nh,
            Psi=self.Psi,
            F=self.F,
            u_opt=self.u_opt,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load_hjb_solution(self):
        hjb_sol = np.load(
            os.path.join(self.dir_path, 'hjb-solution.npz'),
            allow_pickle=True,
        )
        self.sde.domain_h = hjb_sol['domain_h']
        self.sde.Nx = hjb_sol['Nx']
        self.sde.Nh = hjb_sol['Nh']
        self.Psi = hjb_sol['Psi']
        self.F = hjb_sol['F']
        self.u_opt = hjb_sol['u_opt']
        self.t_initial = hjb_sol['t_initial']
        self.t_final = hjb_sol['t_final']

    def write_report(self, x):
        # get index of x
        idx = self.sde.get_index(x)

        # Psi at x
        Psi = self.Psi[idx] if self.Psi is not None else np.nan
        F = self.F[idx] if self.F is not None else np.nan

        # write file
        f = open(os.path.join(self.dir_path, 'report.txt'), "w")

        f.write('h = {:2.4f}\n'.format(self.sde.h))
        f.write('N_h = {:d}\n'.format(self.sde.Nh))

        posicion = 'x: ('
        for i in range(self.sde.n):
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
