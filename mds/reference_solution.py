from potentials_and_gradients import get_potential_and_gradient
from plotting import Plot
from utils import get_data_path, get_sde_stamp, get_trajectories_stamp

import numpy as np

from inspect import isfunction

import os

#TODO: generalize method for f and g different to 1 and 0
class langevin_1d_reference_solution():
    def __init__(self, potential_name, alpha, beta, target_set, h=0.001):

        # get potential and gradient functions
        potential, gradient = get_potential_and_gradient(potential_name, alpha)

        # validate target set
        target_set_min, target_set_max = target_set
        assert target_set_min < target_set_max, \
            'The target set interval is not valid'

        # dir_path
        self.dir_path = get_data_path(potential_name, alpha, beta,
                                      target_set, 'reference_solution')

        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.target_set_min = target_set_min
        self.target_set_max = target_set_max
        self.h = h

    def compute_reference_solution(self):
        ''' This method computes by using finite differences the solution
            of the following BVP:
                0 = LΨ − β f Ψ in S
                Ψ = exp(−βg) in ∂S,
            where f = 1, g = 1 and L is the infinitessimal generator
            of the not drifted 1d overdamped langevin process:
                L = - dV/dx d/dx + epsilon d^2/dx^2
            Its solution is the moment generating function associated
            to the overdamped langevin sde.
        '''
        gradient = self.gradient
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max

        # discretization of the state space
        omega_min = -3
        omega_max = 3
        h = self.h
        N = int((omega_max - omega_min) / h) + 1
        omega_h = np.around(np.linspace(omega_min, omega_max, N), decimals=3)

        # index of the target set T and of the domain S = Omega \ T
        idx_ts = np.where((omega_h >= target_set_min) & (omega_h <= target_set_max))[0]
        idx_S = np.where((omega_h < target_set_min) | (omega_h > target_set_max))[0]

        # After discreting the operator (epsion L -1), the boundary conditions
        # and stability conditions we obtain a linear system of equations: A \Psi = b.
        a = np.zeros((N, N))
        b = np.zeros(N)

        # discretization of (epsilon L - 1) on S
        for k in idx_S[1:-1]:
            dV = gradient(omega_h[k])
            a[k, k - 1] = 1 / (beta**2 * h**2) + dV / (beta * 2 * h)
            a[k, k] = - 2 / (beta**2 * h**2) - 1
            a[k, k + 1] = 1 / (beta**2 * h**2) - dV / (beta * 2 * h)

        # boundary condition. Psi = 1 in ∂S but also in the target set
        for i in idx_ts:
            a[i, i] = 1
        b[idx_ts] = 1

        # stability condition: Psi should be flat on the boundary of the
        # discretized state space, i.e Psi(x_0)=Psi(x_1) and Psi(x_{N-1})=Psi(x_N)
        a[0, 0] = 1
        a[0, 1] = -1
        b[0] = 0

        a[N - 1 , N - 1] = 1
        a[N - 1 , N - 2] = -1
        b[N - 1] = 0

        # solve the linear system 
        Psi = np.linalg.solve(a, b)
        #Psi, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

        # compute the free energy which corresponds to the value function at the optimal control
        F =  - np.log(Psi) / beta

        # compute the optimal control u_opt = -sqrt(2) grad F by finite differences:
        # u_opt(x_k) = -sqrt(2) (F(x_{k+1}) - F(x_{k+1})) / 2h 
        u_opt = np.zeros(N)
        u_opt[1: -1] = - np.sqrt(2) * (F[2:] - F[:-2]) / (2 * h)

        # stability condition: u_opt should be flat on the boundary
        u_opt[0] = u_opt[1]
        u_opt[N-1] = u_opt[N-2]

        # save variables
        self.h = h
        self.omega_h = omega_h
        self.Psi = Psi
        self.F = F
        self.u_opt = u_opt

    def save_reference_solution(self):
        np.savez(
            os.path.join(self.dir_path, 'reference_solution.npz'),
            omega_h=self.omega_h,
            Psi=self.Psi,
            F=self.F,
            u_opt=self.u_opt,
        )

    def plot_free_energy(self):
        X = self.omega_h
        F = self.F
        pl = Plot(self.dir_path, 'free_energy')
        pl.set_ylim(bottom=0, top=self.alpha * 2.5)
        pl.free_energy(X, F)

    def plot_optimal_tilted_potential(self):
        X = self.omega_h
        V = self.potential(X)
        Vb = 2 * self.F
        pl = Plot(self.dir_path, 'optimal_tilted_potential')
        pl.set_ylim(bottom=0, top=self.alpha * 10)
        pl.potential_and_tilted_potential(X, V, Vb)

    def plot_optimal_tilted_drift(self):
        X = self.omega_h
        dV = self.gradient(X)
        dVb = - np.sqrt(2) * self.u_opt
        pl = Plot(self.dir_path, 'optimal_tilted_drift')
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        pl.drift_and_tilted_drift(X, dV, dVb)

    def plot_optimal_control(self):
        X = self.omega_h
        u = self.u_opt
        pl = Plot(self.dir_path, 'optimal_control')
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        pl.control(X, u)
