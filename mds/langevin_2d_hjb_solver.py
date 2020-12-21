from mds.plots_2d import Plot2d
from mds.potentials_and_gradients_2d import get_potential_and_gradient
from mds.utils import get_example_data_path, get_time_in_hms
from mds.numpy_utils import coarse_matrix
from mds.validation import is_2d_valid_interval, is_2d_valid_target_set

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    def __init__(self, f, g, potential_name, alpha, beta, target_set, h, domain=None):

        # validate domain and target set
        if domain is None:
            domain = np.array([[-3, 3], [-3, 3]])
        is_2d_valid_interval(domain)
        is_2d_valid_target_set(domain, target_set)

        # dir_path
        self.dir_path = get_example_data_path(potential_name, alpha, beta,
                                              target_set, 'reference_solution')

        # get potential and gradient functions
        potential, gradient, _, _, _ = get_potential_and_gradient(potential_name, alpha)

        self.f = f
        self.g = g
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.domain = domain
        self.target_set = target_set
        self.h = h

        self.domain_h = None
        self.N = None
        self.Nx = None
        self.Ny = None
        self.Psi = None
        self.F = None
        self.u_opt_x = None
        self.u_opt_y = None

        # computational time
        self.t_initial = None
        self.t_final = None

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def discretize_domain(self):
        ''' this method discretizes the rectangular domain uniformly with step-size h
        '''
        assert self.h is not None, ''
        d_xmin, d_xmax = self.domain[0]
        d_ymin, d_ymax = self.domain[1]
        h = self.h
        x = np.around(np.arange(d_xmin, d_xmax + h, h), decimals=3)
        y = np.around(np.arange(d_ymin, d_ymax + h, h), decimals=3)
        X, Y = np.meshgrid(x, y, sparse=False, indexing='ij')
        self.domain_h = np.dstack((X, Y))
        self.Nx = x.shape[0]
        self.Ny = y.shape[0]
        self.N = self.Nx * self.Ny

    def get_x(self, k):
        ''' returns the x-coordinate of the node k
        '''
        assert k in np.arange(self.N), ''

        X = self.domain_h[:, :, 0]
        return X[np.mod(k, self.Nx), 0]

    def get_y(self, k):
        ''' returns the y-coordinate of the node k
        '''
        assert k in np.arange(self.N), ''

        Y = self.domain_h[:, :, 1]
        return Y[0, np.floor_divide(k, self.Ny)]

    def is_on_domain_boundary(self, k):
        ''' returns True if the node k is on the rectangular
            boundary of the domain
        '''
        assert k in np.arange(self.N), ''

        if np.mod(k + 1, self.Nx) == 1:
            return True
        elif np.mod(k + 1, self.Nx) == 0:
            return True
        elif k + 1 <= self.Nx:
            return True
        elif k >= self.Nx * (self.Ny -1):
            return True
        return False

    def is_on_domain_corner(self, k):
        ''' returns True if the node k is on the corner of the rectangular boundary
        '''
        assert k in np.arange(self.N), ''

        if k in [0, self.Nx -1, self.Nx * (self.Ny -1), self.N-1]:
            return True
        else:
            return False

    def is_on_ts(self, k):
        '''returns True if the node k is on the target set
        '''
        assert k in np.arange(self.N), ''

        x = self.get_x(k)
        y = self.get_y(k)

        if (x <= self.target_set[0, 1] and
            x >= self.target_set[0, 0] and
            y <= self.target_set[1, 1] and
            y >= self.target_set[1, 0]):
            return True
        return False

    def solve_bvp(self):
        # assemble linear system of equations: A \Psi = b.
        A = sparse.lil_matrix((self.N, self.N))
        b = np.zeros(self.N)

        # nodes in boundary, boundary corner and target set
        idx_boundary = np.array([k for k in np.arange(self.N) if self.is_on_domain_boundary(k)])
        idx_corner = np.array([k for k in idx_boundary if self.is_on_domain_corner(k)])
        idx_ts = np.array([k for k in np.arange(self.N) if self.is_on_ts(k)])

        for k in np.arange(self.N):
            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)
                y = self.get_y(k)
                z = np.array([[x, y]])
                dVx = self.gradient(z)[0, 0]
                dVy = self.gradient(z)[0, 1]
                A[k, k] = - 4 / (self.beta * self.h**2) - 1
                A[k, k -1] = 1 / (self.beta * self.h**2) + dVx / (2 * self.h)
                A[k, k +1] = 1 / (self.beta * self.h**2) - dVx / (2 * self.h)
                A[k, k - self.Nx] = 1 / (self.beta * self.h**2) + dVy / (2 * self.h)
                A[k, k + self.Nx] = 1 / (self.beta * self.h**2) - dVy / (2 * self.h)

            # impose condition on ∂S
            elif k in idx_ts:
                A[k, k] = 1
                b[k] = 1

            # stability condition on the boundary
            elif k in idx_boundary and k not in idx_corner:
                if k + 1 <= self.Nx:
                    A[k, k] = 1
                    A[k, k + self.Nx] = -1
                    b[k] = 0
                elif k >= self.Nx * (self.Ny -1):
                    A[k, k] = 1
                    A[k, k - self.Nx] = -1
                    b[k] = 0
                elif np.mod(k + 1, self.Nx) == 1:
                    A[k, k] = 1
                    A[k, k +1] = -1
                    b[k] = 0
                elif np.mod(k + 1, self.Nx) == 0:
                    A[k, k] = 1
                    A[k, k -1] = -1
                    b[k] = 0

        # stability condition on the corner of the boundary
        A[0, 0] = 1
        A[0, self.Nx + 1] = -1
        b[0] = 0
        A[self.Nx -1, self.Nx -1] = 1
        A[self.Nx -1, 2 * self.Nx -2] = -1
        b[self.Nx - 1] = 0
        A[self.Nx * (self.Ny -1), self.Nx * (self.Ny -1)] = 1
        A[self.Nx * (self.Ny -1), self.Nx * (self.Ny -2) + 1] = -1
        b[self.Nx * (self.Ny -1)] = 0
        A[self.N -1, self.N - 1] = 1
        A[self.N -1, self.N - self.Nx - 2] = -1
        b[self.N -1] = 0

        # solve the linear system
        self.Psi = linalg.spsolve(A.tocsc(), b).reshape((self.Nx, self.Ny)).T

    def compute_free_energy(self):
        ''' this methos computes the free energy
                F = - epsilon log (Psi)
        '''
        self.F =  - np.log(self.Psi)

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_opt = - √2 ∇F
        '''
        assert self.F is not None, ''
        assert self.F.ndim == 2, ''
        assert self.F.shape == (self.Nx, self.Ny), ''

        u_opt = np.zeros((self.Nx, self.Ny, 2))
        u_opt_x = np.zeros((self.Nx, self.Ny))
        u_opt_y = np.zeros((self.Nx, self.Ny))

        u_opt_x[1: -1, :] = - np.sqrt(2) * (self.F[2:, :] - self.F[:-2, :]) / (2 * self.h)
        u_opt_x[0, :] = u_opt_x[1, :]
        u_opt_x[self.Nx - 1, :] = u_opt_x[self.Nx - 2, :]
        u_opt[:, :, 0] = u_opt_x

        u_opt_y[:, 1: -1] = - np.sqrt(2) * (self.F[:, 2:] - self.F[:, :-2]) / (2 * self.h)
        u_opt_y[:, 0] = u_opt_y[:, 1]
        u_opt_y[:, self.Ny - 1] = u_opt_y[:, self.Ny - 2]
        u_opt[:, :, 1] = u_opt_y

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
        self.domain_h = ref_sol['domain_h']
        self.Psi = ref_sol['Psi']
        self.F = ref_sol['F']
        self.u_opt = ref_sol['u_opt']
        self.t_initial = ref_sol['t_initial']
        self.t_final = ref_sol['t_final']

    def write_report(self, x):
        # exp fht and mgf at x
        idx_x = np.where(
            (self.domain_h[:, :, 0] == x[0]) &
            (self.domain_h[:, :, 1] == x[1])
        )
        if idx_x[0].shape[0] != 1 or idx_x[1].shape[0] != 1:
            return
        idx_x1 = idx_x[0][0]
        idx_x2 = idx_x[1][0]
        Psi = self.Psi[idx_x1, idx_x2] if self.Psi is not None else np.nan
        F = self.F[idx_x1, idx_x2] if self.F is not None else np.nan

        # file name
        h_ext = '_h{:.0e}'.format(self.h)
        file_name = 'report' + h_ext + '.txt'

        # write file
        f = open(os.path.join(self.dir_path, file_name), "w")
        f.write('h = {:2.4f}\n'.format(self.h))
        f.write('N_h = {:d}\n'.format(self.N))
        f.write('(x_1, x_2) = ({:2.1f}, {:2.1f})\n'.format(x[0], x[1]))
        f.write('Psi at x = {:2.3e}\n'.format(Psi))
        f.write('F at x = {:2.3e}\n'.format(F))
        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))
        f.close()

    def plot_psi(self):
        Psi = self.Psi
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]

        # surface plot
        plt2d = Plot2d(self.dir_path, 'psi_surface')
        #plt2d.set_title(r'$\Psi(x_1, x_2)$')
        plt2d.surface(X, Y, Psi)

        # contour plot
        plt2d = Plot2d(self.dir_path, 'psi_contour')
        #plt2d.set_title(r'$\Psi(x_1, x_2)$')
        plt2d.contour(X, Y, Psi)

    def plot_free_energy(self):
        F = self.F
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]

        # surface plot
        plt2d = Plot2d(self.dir_path, 'free_energy_surface')
        #plt2d.set_title(r'$F(x_1, x_2)$')
        plt2d.surface(X, Y, F)

        # contour plot
        levels = np.linspace(-0.5, 3.5, 21)
        plt2d = Plot2d(self.dir_path, 'free_energy_contour')
        #plt2d.set_title(r'$F(x_1, x_2)$')
        plt2d.set_zlim(0, 3)
        plt2d.contour(X, Y, F)

    def plot_optimal_tilted_potential(self):
        F = self.F
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        Nx = self.domain_h.shape[0]
        Ny = self.domain_h.shape[1]
        N = Nx * Ny
        x = self.domain_h.reshape(N, 2)
        V = self.potential(x).reshape(Nx, Ny)
        Vb = 2 * F

        # surface plot
        plt2d = Plot2d(self.dir_path, 'tilted_potential_surface')
        #plt2d.set_title(r'$\tilde{V}(x_1, x_2)$')
        plt2d.set_xlim(-2, 2)
        plt2d.set_ylim(-2, 2)
        plt2d.set_zlim(0, 10)
        plt2d.surface(X, Y, V + Vb)

        # contour plot
        levels = np.logspace(-2, 1, 20, endpoint=True)
        plt2d = Plot2d(self.dir_path, 'tilted_potential_contour')
        #plt2d.set_title(r'$\tilde{V}(x_1, x_2)$')
        plt2d.set_xlim(-2, 2)
        plt2d.set_ylim(-2, 2)
        plt2d.set_zlim(0, 10)
        plt2d.contour(X, Y, V + Vb, levels)

    def plot_optimal_control(self):
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        U = self.u_opt[:, :, 0]
        V = self.u_opt[:, :, 1]

        #gradient plot
        plt2d = Plot2d(self.dir_path, 'control')
        #plt2d.set_title(r'$u_{opt}(x_1, x_2)$')
        plt2d.vector_field(X, Y, U, V, scale=8)

        #zoom gradient plot
        plt2d = Plot2d(self.dir_path, 'control_zoom_ts')
        #plt2d.set_title(r'$u_{opt}(x_1, x_2)$')
        plt2d.set_xlim(0, 2)
        plt2d.set_ylim(0, 2)
        plt2d.vector_field(X, Y, U, V, scale=30)

    def plot_optimal_tilted_drift(self):
        X = self.domain_h[:, :, 0]
        Y = self.domain_h[:, :, 1]
        Nx = self.domain_h.shape[0]
        Ny = self.domain_h.shape[1]
        N = Nx * Ny
        x = self.domain_h.reshape(N, 2)
        dV = self.gradient(x).reshape(Nx, Ny, 2)
        U = - dV[:, :, 0] + np.sqrt(2) * self.u_opt[:, :, 0]
        V = - dV[:, :, 1] + np.sqrt(2) * self.u_opt[:, :, 1]

        plt2d = Plot2d(self.dir_path, 'tilted_drift')
        #plt2d.set_title(r'$-\nabla \tilde{V}(x_1, x_2)$')
        plt2d.set_xlim(-1.5, 1.5)
        plt2d.set_ylim(-1.5, 1.5)
        plt2d.vector_field(X, Y, U, V, scale=50)
