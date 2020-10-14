from potentials_and_gradients import get_potential_and_gradient
from plotting import Plot
from utils import get_example_data_path
from validation import is_1d_valid_domain, is_1d_valid_target_set

import functools
import numpy as np

import os

class langevin_hjb_1d_solver():
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            0 = LΨ − β f Ψ in S
            Ψ = exp(−βg) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not drifted 1d overdamped langevin process:
            L = - dV/dx d/dx + epsilon d^2/dx^2
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
    '''

    def __init__(self, f, g, potential_name, alpha, beta, target_set, domain=None, h=0.001):
        # validate domain and target set
        if domain is None:
            domain = np.array([-3, 3])
        is_1d_valid_domain(domain)
        is_1d_valid_target_set(domain, target_set)

        # dir_path
        self.dir_path = get_example_data_path(potential_name, alpha, beta,
                                              target_set, 'reference_solution')

        # get potential and gradient functions
        potential, gradient = get_potential_and_gradient(potential_name, alpha)

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

    def discretize_domain(self):
        ''' this method discretizes the domain interval uniformly with step-size h
        '''
        D_min, D_max = self.domain
        h = self.h
        assert h is not None, ''

        self.N = int((D_max - D_min) / h) + 1
        self.domain_h = np.around(np.linspace(D_min, D_max, self.N), decimals=3)

    def get_x(self, k):
        ''' returns the x-coordinate of the node k
        '''
        N = self.N
        domain_h = self.domain_h
        assert k in np.arange(N), ''
        return domain_h[k]

    def solve_bvp(self):
        f = self.f
        g = self.g
        gradient = self.gradient
        beta = self.beta
        domain_h = self.domain_h
        target_set_min, target_set_max = self.target_set
        h = self.h
        N = self.N

        # nodes in boundary and in the target set T
        idx_boundary = np.array([0, N-1])
        idx_ts = np.where((domain_h >= target_set_min) & (domain_h <= target_set_max))[0]

        # assemble linear system of equations: A \Psi = b.
        A = np.zeros((N, N))
        b = np.zeros(N)

        for k in np.arange(N):
            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)
                dV = gradient(x)
                A[k, k] = - 2 / (beta**2 * h**2) - f(x)
                A[k, k - 1] = 1 / (beta**2 * h**2) + dV / (beta * 2 * h)
                A[k, k + 1] = 1 / (beta**2 * h**2) - dV / (beta * 2 * h)

            # impose condition on ∂S
            elif k in idx_ts:
                x = self.get_x(k)
                A[k, k] = 1
                b[k] = np.exp(-beta * g(x))

        # stability condition on the boundary: Psi should be flat
        # i.e Psi(x_0)=Psi(x_1) and Psi(x_{N-1})=Psi(x_N)
        A[0, 0] = 1
        A[0, 1] = -1
        b[0] = 0
        A[N - 1 , N - 1] = 1
        A[N - 1 , N - 2] = -1
        b[N - 1] = 0

        # solve the linear system and save the mgf
        self.Psi = np.linalg.solve(A, b)
        #self.Psi, _, _, _ =  np.linalg.lstsq(A, b, rcond=None)

    def compute_free_energy(self):
        ''' this methos computes the free energy
                F = - epsilon log (Psi)
        '''
        beta = self.beta
        Psi = self.Psi

        self.F =  - np.log(Psi) / beta

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control
                u_opt = - √2 dF/dx
        '''
        F = self.F
        h = self.h
        N = self.N

        # central difference quotients: for any k in {1, ..., N-2}
        # u_opt(x_k) = -sqrt(2) (F(x_{k+1}) - F(x_{k-1})) / 2h 
        u_opt = np.zeros(N)
        u_opt[1: N-1] = - np.sqrt(2) * (F[2:] - F[:N-2]) / (2 * h)

        # stability condition on the boundary: u_opt should be flat
        u_opt[0] = u_opt[1]
        u_opt[N-1] = u_opt[N-2]

        self.u_opt = u_opt

    def compute_exp_fht(self):
        def f(x, c):
            return c

        l = 0.001
        sol_plus = langevin_hjb_1d_solver(
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
        sol_minus = langevin_hjb_1d_solver(
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
        file_path = os.path.join(self.dir_path, 'reference_solution.npz')
        np.savez(
            file_path,
            domain_h=self.domain_h,
            Psi=self.Psi,
            F=self.F,
            u_opt=self.u_opt,
            exp_fht=self.exp_fht,
        )

    def load_reference_solution(self):
        file_path = os.path.join(self.dir_path, 'reference_solution.npz')
        ref_sol = np.load(file_path)
        self.domain_h = ref_sol['domain_h']
        self.Psi = ref_sol['Psi']
        self.F = ref_sol['F']
        self.u_opt = ref_sol['u_opt']
        self.exp_fht = ref_sol['exp_fht']

    def write_report(self, x):
        domain_h = self.domain_h

        # exp fht and mgf at x
        idx_x = np.where(domain_h == x)[0]
        if idx_x.shape[0] != 1:
            return
        exp_fht = self.exp_fht[idx_x[0]] if self.exp_fht is not None else np.nan
        Psi = self.Psi[idx_x[0]] if self.Psi is not None else np.nan
        F = self.F[idx_x[0]] if self.F is not None else np.nan

        # write file
        file_path = os.path.join(self.dir_path, 'report.txt')
        f = open(file_path, "w")
        f.write('x = {:2.1f}\n'.format(x))
        f.write('E[fht] at x = {:2.3f}\n'.format(exp_fht))
        f.write('Psi at x = {:2.3e}\n'.format(Psi))
        f.write('F at x = {:2.3e}\n'.format(F))
        f.close()

    def plot_mgf(self):
        x = self.domain_h
        Psi = self.Psi
        pl = Plot(self.dir_path, 'mgf')
        pl.set_ylim(bottom=0, top=self.alpha * 2)
        pl.mgf(x, Psi)

    def plot_free_energy(self):
        x = self.domain_h
        F = self.F
        pl = Plot(self.dir_path, 'free_energy')
        pl.set_ylim(bottom=0, top=self.alpha * 3)
        pl.free_energy(x, F)

    def plot_optimal_control(self):
        x = self.domain_h
        u = self.u_opt
        pl = Plot(self.dir_path, 'optimal_control')
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        pl.control(x, u)

    def plot_optimal_tilted_potential(self):
        x = self.domain_h
        V = self.potential(x)
        Vb_opt = 2 * self.F
        pl = Plot(self.dir_path, 'optimal_tilted_potential')
        pl.set_ylim(bottom=0, top=self.alpha * 10)
        pl.potential_and_tilted_potential(x, V, Vbias_opt=Vb_opt)

    def plot_optimal_tilted_drift(self):
        x = self.domain_h
        dV = self.gradient(x)
        dVb_opt = - np.sqrt(2) * self.u_opt
        pl = Plot(self.dir_path, 'optimal_tilted_drift')
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        pl.drift_and_tilted_drift(x, dV, dVbias_opt=dVb_opt)

    def plot_exp_fht(self):
        x = self.domain_h
        exp_fht = self.exp_fht
        pl = Plot(self.dir_path, 'exp_fht')
        pl.set_ylim(bottom=0, top=self.alpha * 5)
        pl.exp_fht(x, exp_fht)
