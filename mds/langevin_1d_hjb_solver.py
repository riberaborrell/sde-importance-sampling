from mds.potentials_and_gradients_1d import get_potential_and_gradient
from mds.plots_1d import Plot1d
from mds.utils import get_example_data_path, get_time_in_hms
from mds.validation import is_1d_valid_interval, is_1d_valid_target_set

import functools
import numpy as np
import time

import os

class Solver():
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not drifted 1d overdamped langevin process:
            L = - dV/dx d/dx + epsilon d^2/dx^2
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
    '''

    def __init__(self, f, g, potential_name, alpha, beta, target_set, h, domain=None):
        # validate domain and target set
        if domain is None:
            domain = np.array([-3, 3])
        is_1d_valid_interval(domain)
        is_1d_valid_target_set(domain, target_set)

        # dir_path
        self.dir_path = get_example_data_path(potential_name, alpha, beta,
                                              target_set, 'reference_solution')

        # get potential and gradient functions
        potential, gradient, _, _, _ = get_potential_and_gradient(potential_name, alpha)

        self.f = f
        self.g = g
        self.potential_name = potential_name
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.domain = domain
        self.target_set = target_set
        self.h = h

        self.N = None
        self.domain_h = None
        self.Psi = None
        self.F = None
        self.u_opt = None
        self.exp_fht = None

        # computational time
        self.t_initial = None
        self.t_final = None

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def discretize_domain(self):
        ''' this method discretizes the domain interval uniformly with step-size h
        '''
        assert self.h is not None, ''

        d_min, d_max = self.domain
        self.N = int((d_max - d_min) / self.h) + 1
        self.domain_h = np.around(np.linspace(d_min, d_max, self.N), decimals=3)

    def get_x(self, k):
        ''' returns the x-coordinate of the node k
        '''
        assert k in np.arange(self.N), ''

        return self.domain_h[k]

    def solve_bvp(self):
        # assemble linear system of equations: A \Psi = b.
        A = np.zeros((self.N, self.N))
        b = np.zeros(self.N)

        # nodes in boundary and in the target set T
        target_set_min, target_set_max = self.target_set
        idx_boundary = np.array([0, self.N - 1])
        idx_ts = np.where(
            (self.domain_h >= target_set_min) & (self.domain_h <= target_set_max)
        )[0]

        for k in np.arange(self.N):
            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)
                dV = self.gradient(x)
                A[k, k] = - 2 / (self.beta * self.h**2) - self.f(x)
                A[k, k - 1] = 1 / (self.beta * self.h**2) + dV / (2 * self.h)
                A[k, k + 1] = 1 / (self.beta * self.h**2) - dV / (2 * self.h)

            # impose condition on ∂S
            elif k in idx_ts:
                x = self.get_x(k)
                A[k, k] = 1
                b[k] = np.exp(- self.g(x))

        # stability condition on the boundary: Psi should be flat
        # i.e Psi(x_0)=Psi(x_1) and Psi(x_{N-1})=Psi(x_N)
        A[0, 0] = 1
        A[0, 1] = -1
        b[0] = 0
        A[self.N - 1 , self.N - 1] = 1
        A[self.N - 1 , self.N - 2] = -1
        b[self.N - 1] = 0

        # solve the linear system and save the mgf
        self.Psi = np.linalg.solve(A, b)

    def compute_free_energy(self):
        ''' this methos computes the free energy
                F = - epsilon log (Psi)
        '''
        self.F =  - np.log(self.Psi)

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control
                u_opt = - √2 dF/dx
        '''
        # central difference quotients: for any k in {1, ..., N-2}
        # u_opt(x_k) = -sqrt(2) (F(x_{k+1}) - F(x_{k-1})) / 2h 
        u_opt = np.zeros(self.N)
        u_opt[1: self.N - 1] = - np.sqrt(2) * (self.F[2:] - self.F[:self.N - 2]) / (2 * self.h)

        # stability condition on the boundary: u_opt should be flat
        u_opt[0] = u_opt[1]
        u_opt[self.N - 1] = u_opt[self.N - 2]

        self.u_opt = u_opt

    def compute_exp_fht(self):
        def f(x, c):
            return c

        l = 0.001
        sol_plus = Solver(
            f=functools.partial(f, c=l),
            g=self.g,
            potential_name=self.potential_name,
            alpha=self.alpha,
            beta=self.beta,
            target_set=self.target_set,
            domain=self.domain,
            h=self.h,
        )
        sol_plus.discretize_domain()
        sol_plus.solve_bvp()
        sol_minus = Solver(
            f=functools.partial(f, c=-l),
            g=self.g,
            potential_name=self.potential_name,
            alpha=self.alpha,
            beta=self.beta,
            target_set=self.target_set,
            domain=self.domain,
            h=self.h,
        )
        sol_minus.discretize_domain()
        sol_minus.solve_bvp()

        self.exp_fht = - (sol_plus.Psi - sol_minus.Psi) / (self.beta * 2 * l)

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
            #exp_fht=self.exp_fht,
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
        #self.exp_fht = ref_sol['exp_fht']
        self.t_initial = ref_sol['t_initial']
        self.t_final = ref_sol['t_final']

    def write_report(self, x):
        # exp fht and mgf at x
        idx_x = np.where(self.domain_h == x)[0]
        if idx_x.shape[0] != 1:
            return
        #exp_fht = self.exp_fht[idx_x[0]] if self.exp_fht is not None else np.nan
        Psi_at_x = self.Psi[idx_x[0]] if self.Psi is not None else np.nan
        F_at_x = self.F[idx_x[0]] if self.F is not None else np.nan

        # file name
        h_ext = '_h{:.0e}'.format(self.h)
        file_name = 'report' + h_ext + '.txt'

        # write file
        f = open(os.path.join(self.dir_path, file_name), "w")
        f.write('h = {:2.4f}\n'.format(self.h))
        f.write('N_h = {:d}\n'.format(self.N))
        f.write('x = {:2.1f}\n'.format(x))
        #f.write('E[fht] at x = {:2.3f}\n'.format(exp_fht))
        f.write('Psi at x = {:2.3e}\n'.format(Psi_at_x))
        f.write('F at x = {:2.3e}\n'.format(F_at_x))
        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))
        f.close()

    def plot_psi(self):
        x = self.domain_h
        Psi = self.Psi
        plt1d = Plot1d(self.dir_path, 'psi')
        plt1d.set_ylim(0, 2 * self.alpha)
        plt1d.one_line_plot(x, Psi, color='c', label=r'$\Psi(x)$')

    def plot_free_energy(self):
        x = self.domain_h
        F = self.F
        plt1d = Plot1d(self.dir_path, 'free_energy')
        plt1d.set_ylim(0, 3 * self.alpha)
        plt1d.one_line_plot(x, F, color='c', label=r'$F(x)$')

    def plot_optimal_control(self):
        x = self.domain_h
        u = self.u_opt
        plt1d = Plot1d(self.dir_path, 'control')
        plt1d.set_ylim(- 5 * self.alpha, 5 * self.alpha)
        plt1d.one_line_plot(x, u, color='c', label=r'$u(x)$')

    def plot_optimal_tilted_potential(self):
        x = self.domain_h
        V = self.potential(x)
        Vb_opt = 2 * self.F
        ys = np.vstack((V, Vb_opt, V + Vb_opt))
        colors = ['b', 'y', 'c']
        labels = [r'$V(x)$', r'$V_b(x)$', r'$\tilde{V}(x)$']
        plt1d = Plot1d(self.dir_path, 'tilted_potential')
        plt1d.set_ylim(0, 10 * self.alpha)
        plt1d.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_optimal_tilted_drift(self):
        x = self.domain_h
        dV = self.gradient(x)
        dVb_opt = - np.sqrt(2) * self.u_opt
        ys = np.vstack((-dV, -dVb_opt, -dV - dVb_opt))
        colors = ['b', 'y', 'c']
        labels = [r'$- \nabla V(x)$', r'$ - \nabla V_b(x)$', r'$ - \nabla \tilde{V}(x)$']
        plt1d = Plot1d(self.dir_path, 'tilted_drift')
        plt1d.set_ylim(- 5 * self.alpha, 5 * self.alpha)
        plt1d.multiple_lines_plot(x, ys, colors=colors, labels=labels)

    def plot_exp_fht(self):
        x = self.domain_h
        exp_fht = self.exp_fht
        plt1d = Plot1d(self.dir_path, 'exp_fht')
        plt1d.set_ylim(0, self.alpha * 5)
        plt1d.one_line_plot(x, exp_fht)
