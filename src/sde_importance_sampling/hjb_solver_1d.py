from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.utils_path import get_hjb_solution_dir_path, get_time_in_hms
from sde_importance_sampling.utils_numeric import arange_generator, from_1dndarray_to_string

import functools
import numpy as np
import time

import os

class SolverHJB1d(LangevinSDE):
    ''' This class provides a solver of the following BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not controlled 1d overdamped langevin process:
            L = - dV/dx d/dx + beta^{-1} d^2/dx^2
        Its solution is the moment generating function associated
        to the overdamped langevin sde.
    '''

    def __init__(self, problem_name, potential_name, n, alpha,
                 beta, h, domain=None, target_set=None, T=None):

        assert n == 1, ''
        #TODO raise error method

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

    def get_x(self, k):
        ''' returns the x-coordinate of the node k
        '''
        assert k in np.arange(self.Nh), ''

        return self.domain_h[k]

    def solve_bvp(self):
        ''' solve bvp by using finite difference
        '''
        # assemble linear system of equations: A \Psi = b.
        A = np.zeros((self.Nh, self.Nh))
        b = np.zeros(self.Nh)

        # nodes in boundary
        idx_boundary = np.array([0, self.Nh - 1])

        # nodes in target set
        target_set_lb, target_set_ub = self.target_set[0, :]
        idx_ts = np.where(
            (self.domain_h >= target_set_lb) & (self.domain_h <= target_set_ub)
        )[0]

        for k in np.arange(self.Nh):

            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)
                dV = self.gradient(x)
                A[k, k] = - 2 / (self.beta * self.h**2) - self.f(x)
                A[k, k - 1] = 1 / (self.beta * self.h**2) + dV / (2 * self.h)
                A[k, k + 1] = 1 / (self.beta * self.h**2) - dV / (2 * self.h)
                b[k] = 0

            # impose condition on ∂S
            elif k in idx_ts:
                x = self.get_x(k)
                A[k, k] = 1
                b[k] = np.exp(- self.g(x))

        # stability condition on the boundary: Psi should be flat

        # Psi_0 = Psi_1
        A[0, 0] = 1
        A[0, 1] = -1
        b[0] = 0

        # psi_{Nh-1} = Psi_N)
        A[-1, -1] = 1
        A[-1, -2] = -1
        b[-1] = 0

        # solve linear system and save
        psi = np.linalg.solve(A, b)
        self.psi = psi.reshape(self.Nx)
        self.solved = True

    def compute_value_function(self):
        ''' this methos computes the value function
                value_f = - log (psi)
        '''
        self.value_f =  - np.log(self.psi)

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control
                u_opt = (- √2  / beta) ∇_x value_f
        '''
        assert hasattr(self, 'value_f'), ''
        assert self.value_f.ndim == self.n, ''
        assert self.value_f.shape == self.Nx, ''

        self.u_opt = np.zeros((self.Nh, 1))

        # forwad difference approximation
        # for any k in {0, ..., Nh-2}
        # u_opt(x_k) = - (sqrt(2) / beta) (Phi_{k+1} - Phi_{k}) / h 

        #self.u_opt[:-1, 0] = - np.sqrt(2) \
        #                   * (self.value_f[1:] - self.value_f[:-1]) / self.h
        #self.u_opt[-1, 0] = self.u_opt[-2, 0]

        # central difference approximation
        # for any k in {1, ..., Nh-2}
        # u_opt(x_k) = - (sqrt(2) / beta) (Phi_{k+1} - Phi_{k-1}) / 2h 

        self.u_opt[1:-1, 0] = - np.sqrt(2) \
                            * (self.value_f[2:] - self.value_f[:-2]) \
                            / (2 * self.h)
        self.u_opt[0, 0] = self.u_opt[1, 0]
        self.u_opt[-1, 0] = self.u_opt[-2, 0]

    def compute_exp_fht(self):
        #TODO! debug
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

    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save arrays in a npz file
        np.savez(
            os.path.join(self.dir_path, 'hjb-solution-1d.npz'),
            domain_h=self.domain_h,
            Nx=self.Nx,
            Nh=self.Nh,
            psi=self.psi,
            value_f=self.value_f,
            u_opt=self.u_opt,
            #exp_fht=self.exp_fht,
            ct=self.ct,
        )

    def load(self):
        ''' loads the saved arrays and sets them as attributes back
        '''
        try:
            data = np.load(
                os.path.join(self.dir_path, 'hjb-solution-1d.npz'),
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

        x_str = 'x: ' + from_1dndarray_to_string(x)
        psi = self.get_psi_at_x(x)
        value_f = self.get_value_function_at_x(x)
        u_opt = self.get_u_opt_at_x(x)
        u_opt_str = 'u_opt(x): ' + from_1dndarray_to_string(u_opt)

        f.write(x_str)
        if psi is not None:
            f.write('psi(x) = {:2.4e}\n'.format(psi))

        if value_f is not None:
            f.write('value_f(x) = {:2.4e}\n'.format(value_f))

        f.write(u_opt_str)

        # maximum value of the control
        f.write('\n maximum value of the optimal control\n')

        idx_u_max = np.argmax(self.u_opt)
        x_u_max = self.get_x(idx_u_max)
        u_opt_max = self.u_opt[idx_u_max]
        x_u_max_str = 'x: ' + from_1dndarray_to_string(x_u_max)
        u_opt_max_str = 'max u_opt(x): = ' + from_1dndarray_to_string(u_opt_max)

        f.write(x_u_max_str)
        f.write(u_opt_max_str)

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

    def plot_1d_exp_fht(self):
        #TODO! debug
        x = self.domain_h
        exp_fht = self.exp_fht
        plt1d = Plot1d(self.dir_path, 'exp_fht')
        plt1d.set_ylim(0, self.alpha * 5)
        plt1d.one_line_plot(x, exp_fht)
