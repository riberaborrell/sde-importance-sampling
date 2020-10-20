from potentials_and_gradients import get_potential_and_gradient
from utils import get_example_data_path
from numpy_utils import coarse_matrix

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from inspect import isfunction

import os

class Solver():
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            0 = LΨ − β f Ψ in S
            Ψ = exp(−βg) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not drifted 2d overdamped langevin process:
            L = - ∇V·∇ + epsilon Δ
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
   '''

    def __init__(self, potential_name, alpha, beta, target_set, domain=None, h=0.1):

        # validate domain
        if domain is None:
            domain = np.array([[-3, 3], [-3, 3]])
        #is_2d_valid_domain(domain)

        # validate target set

        # dir_path
        self.dir_path = get_example_data_path(potential_name, alpha, beta,
                                              target_set, 'reference_solution')

        # get potential and gradient functions
        potential, gradient = get_potential_and_gradient(potential_name, alpha)

        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.domain = domain
        self.target_set = target_set
        self.h = h

        self.meshgrid = None
        self.N = None
        self.Nx = None
        self.Ny = None
        self.Psi = None
        self.F = None
        self.u_opt_x = None
        self.u_opt_y = None

    def discretize_domain(self):
        ''' this method discretizes the rectangular domain uniformly with step-size h
        '''
        D_xmin, D_xmax = self.domain[0]
        D_ymin, D_ymax = self.domain[1]
        h = self.h
        assert h is not None, ''

        x = np.arange(D_xmin, D_xmax + h, h)
        y = np.arange(D_ymin, D_ymax + h, h)
        self.meshgrid = np.meshgrid(x, y, sparse=True, indexing='ij')
        self.Nx = x.shape[0]
        self.Ny = y.shape[0]
        self.N = x.shape[0] * y.shape[0]

    def get_x(self, k):
        ''' returns the x-coordinate of the node k
        '''
        N = self.N
        Nx = self.Nx
        xx, _ = self.meshgrid
        assert k in np.arange(N), ''

        return xx[np.mod(k, Nx), 0]

    def get_y(self, k):
        ''' returns the y-coordinate of the node k
        '''
        N = self.N
        Ny = self.Ny
        _ , yy = self.meshgrid
        assert k in np.arange(N), ''

        return yy[0, np.floor_divide(k, Ny)]

    def is_on_domain_boundary(self, k):
        ''' returns True if the node k is on the rectangular
            boundary of the domain
        '''
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
        assert k in np.arange(N), ''

        if np.mod(k + 1, Nx) == 1:
            return True
        elif np.mod(k + 1, Nx) == 0:
            return True
        elif k + 1 <= Nx:
            return True
        elif k >= Nx * (Ny -1):
            return True
        return False

    def is_on_domain_corner(self, k):
        ''' returns True if the node k is on the corner of the rectangular boundary
        '''
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
        assert k in np.arange(N), ''

        if k in [0, Nx -1, Nx * (Ny -1), N-1]:
            return True
        else:
            return False

    def is_on_ts(self, k):
        '''returns True if the node k is on the target set
        '''
        N = self.N
        assert k in np.arange(N), ''

        target_set_x_min, target_set_x_max = self.target_set[0]
        target_set_y_min, target_set_y_max = self.target_set[1]
        x = self.get_x(k)
        y = self.get_y(k)

        if (x <= target_set_x_max and
            x >= target_set_x_min and
            y <= target_set_y_max and
            y >= target_set_y_min):
            return True
        return False

    def solve_bvp(self):
        gradient = self.gradient
        beta = self.beta
        target_set_x_min = self.target_set[0][0]
        target_set_x_max = self.target_set[0][1]
        target_set_y_min = self.target_set[1][0]
        target_set_y_max = self.target_set[1][1]
        h = self.h

        Nx = self.Nx
        Ny = self.Ny
        N = self.N

        # assemble linear system of equations: A \Psi = b.
        A = np.zeros((N, N))
        b = np.zeros(N)

        # nodes in boundary, boundary corner and target set
        idx_boundary = np.array([k for k in np.arange(N) if self.is_on_domain_boundary(k)])
        idx_corner = np.array([k for k in idx_boundary if self.is_on_domain_corner(k)])
        idx_ts = np.array([k for k in np.arange(N) if self.is_on_ts(k)])

        for k in np.arange(N):
            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)
                y = self.get_y(k)
                #dVx, dVy = gradient(x, y)
                z = np.array([[x, y]])
                dVx = gradient(z)[0, 0]
                dVy = gradient(z)[0, 1]
                A[k, k] = - 4 / (beta**2 * h**2) - 1
                A[k, k -1] = 1 / (beta**2 * h**2) + dVx / (beta * 2 * h)
                A[k, k +1] = 1 / (beta**2 * h**2) - dVx / (beta * 2 * h)
                A[k, k - Nx] = 1 / (beta**2 * h**2) + dVy / (beta * 2 * h)
                A[k, k + Nx] = 1 / (beta**2 * h**2) - dVy / (beta * 2 * h)

            # impose condition on ∂S
            elif k in idx_ts:
                A[k, k] = 1
                b[k] = 1

            # stability condition on the boundary
            elif k in idx_boundary and k not in idx_corner:
                if k + 1 <= Nx:
                    A[k, k] = 1
                    A[k, k + Nx] = -1
                    b[k] = 0
                elif k >= Nx * (Ny -1):
                    A[k, k] = 1
                    A[k, k - Nx] = -1
                    b[k] = 0
                elif np.mod(k + 1, Nx) == 1:
                    A[k, k] = 1
                    A[k, k +1] = -1
                    b[k] = 0
                elif np.mod(k + 1, Nx) == 0:
                    A[k, k] = 1
                    A[k, k -1] = -1
                    b[k] = 0

        # stability condition on the corner of the boundary
        A[0, 0] = 1
        A[0, Nx+1] = -1
        b[0] = 0
        A[Nx -1, Nx -1] = 1
        A[Nx -1, 2 * Nx -2] = -1
        b[Nx - 1] = 0
        A[Nx * (Ny -1), Nx * (Ny -1)] = 1
        A[Nx * (Ny -1), Nx * (Ny -2) + 1] = -1
        b[Nx * (Ny -1)] = 0
        A[N -1, N -1] = 1
        A[N -1, N -Nx -2] = -1
        b[N -1] = 0

        # solve the linear system
        Psi = np.linalg.solve(A, b)
        #Psi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # compute the free energy
        F =  - np.log(Psi) / beta

        # save MGF and free energy
        self.Psi = Psi.reshape((Nx, Ny)).T
        self.F = F.reshape((Nx, Ny)).T

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_opt = - √2 ∇F
        '''
        F = self.F
        h = self.h
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
        assert F is not None, ''
        assert F.ndim == 2, ''
        assert F.shape == (Nx, Ny), ''

        u_opt_x = np.zeros((Nx, Ny))
        u_opt_y = np.zeros((Nx, Ny))

        u_opt_x[1: -1, :] = - np.sqrt(2) * (F[2:, :] - F[:-2, :]) / (2 * h)
        u_opt_x[0, :] = u_opt_x[1, :]
        u_opt_x[Nx-1, :] = u_opt_x[Nx-2, :]

        u_opt_y[:, 1: -1] = - np.sqrt(2) * (F[:, 2:] - F[:, :-2]) / (2 * h)
        u_opt_y[:, 0] = u_opt_y[:, 1]
        u_opt_y[:, Ny-1] = u_opt_y[:, Ny-2]

        self.u_opt_x = u_opt_x
        self.u_opt_y = u_opt_y

    def save_reference_solution(self):
        np.savez(
            os.path.join(self.dir_path, 'reference_solution.npz'),
            h=self.h,
            meshgrid=self.meshgrid,
            Psi=self.Psi,
            F=self.F,
            u_opt_x=self.u_opt_x,
            u_opt_y=self.u_opt_y,
        )

    def load_reference_solution(self):
        ref_sol = np.load(
            os.path.join(self.dir_path, 'reference_solution.npz'),
            allow_pickle=True,
        )
        self.h = ref_sol['h']
        self.meshgrid = ref_sol['meshgrid']
        self.Psi = ref_sol['Psi']
        self.F = ref_sol['F']
        self.u_opt_x = ref_sol['u_opt_x']
        self.u_opt_y = ref_sol['u_opt_y']

    def plot_psi(self):
        Psi = self.Psi
        xx, yy = self.meshgrid

        label = r'$\Psi(x, y), \, h = {}$'
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            xx,
            yy,
            Psi,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        file_path = os.path.join(self.dir_path, 'mgf.png')
        plt.savefig(file_path)
        plt.close()

    def plot_free_energy(self):
        F = self.F
        xx, yy = self.meshgrid

        label = r'$\F(x, y), \, h = {}$'
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            xx,
            yy,
            F,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        file_path = os.path.join(self.dir_path, 'free_energy_surface.png')
        plt.savefig(file_path)
        plt.close()

        fig, ax = plt.subplots()
        levels = np.linspace(-0.5, 3.5, 21)
        X, Y = np.meshgrid(xx[:, 0], yy[0, :], sparse=False, indexing='ij')
        cs = ax.contourf(
            X,
            Y,
            F,
            vmin=0,
            vmax=3,
            levels=levels,
            cmap=cm.coolwarm,
        )
        cbar = fig.colorbar(cs)
        file_path = os.path.join(self.dir_path, 'free_energy_contour.png')
        plt.savefig(file_path)
        plt.close()


    def plot_optimal_tilted_potential(self):
        F = self.F
        xx, yy = self.meshgrid

        label = r'$\tilde{V}(x, y), \, h = {}$'
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            xx,
            yy,
            2 * F,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        fig.colorbar(surf, fraction=0.15, shrink=0.7, aspect=20)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        file_path = os.path.join(self.dir_path, 'optimal_tilted_potential.png')
        plt.savefig(file_path)
        plt.close()

    def plot_optimal_control(self):
        h = self.h
        xx, yy = self.meshgrid
        u_opt_x = self.u_opt_x
        u_opt_y = self.u_opt_y

        fig, ax = plt.subplots()
        label = r'$\u_opt(x, y), \, h = {}$'

        # show every k arrow
        X, Y = np.meshgrid(xx[:, 0], yy[0, :], sparse=False, indexing='ij')
        k = int(X.shape[0] / 20)
        X = X[::k, ::k]
        Y = Y[::k, ::k]
        U = u_opt_x[::k, ::k]
        V = u_opt_y[::k, ::k]
        C = np.sqrt(U**2 + V**2)
        quiv = ax.quiver(X, Y, U, V, C, angles='xy', scale_units='xy')
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        file_path = os.path.join(self.dir_path, 'optimal_control')
        plt.savefig(file_path)
        plt.close()

        fig, ax = plt.subplots()
        # show every k arrow
        X, Y = np.meshgrid(xx[:, 0], yy[0, :], sparse=False, indexing='ij')
        k = 2
        X = X[::k, ::k]
        Y = Y[::k, ::k]
        U = u_opt_x[::k, ::k]
        V = u_opt_y[::k, ::k]
        C = np.sqrt(U**2 + V**2)
        quiv = ax.quiver(X, Y, U, V, C, angles='xy', scale_units='xy')
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        file_path = os.path.join(self.dir_path, 'optimal_control_zoom_ts')
        plt.savefig(file_path)
        plt.close()
