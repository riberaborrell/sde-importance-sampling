import numpy as np

from inspect import isfunction

#TODO: generalize method for f and g different to 1 and 0
class langevin_1d_reference_solution():
    def __init__(self, gradient, beta, target_set):

        assert isfunction(gradient), 'the gradient must be a function'

        target_set_min, target_set_max = target_set
        assert target_set_min < target_set_max, \
            'The target set interval is not valid'

        self.gradient = gradient
        self.beta = beta
        self.target_set_min = target_set_min
        self.target_set_max = target_set_max

    def compute_reference_solution(self):
        ''' This method computes the solution of the following BVP
            0 = LΨ − β f Ψ in D
            Ψ = exp(−βg) in ∂D,
            where f = 1 and g =1, by using the Shortley-Welley method.
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
        h = 0.001
        N = int((omega_max - omega_min) / h) + 1
        omega_h = np.around(np.linspace(omega_min, omega_max, N), decimals=3)

        # get indexes for nodes in/out the target set
        idx_ts = np.where((omega_h >= target_set_min) & (omega_h <= target_set_max))
        target_set_h = omega_h[idx_ts]

        # discretization of the operator (epsionL -1), where L 
        # is the infinitessimal generator of the not drifted langevin 1d
        # L = - dV_bias d/dx + epsilon d^2/dx^2

        a = np.zeros((N, N))
        b = np.zeros(N)

        # weights of Psi
        for i in np.arange(1, N-1):
            dV_bias = gradient(omega_h[i])
            a[i, i - 1] = 1 / (beta**2 * h**2) + dV_bias / (beta * 2 * h)
            a[i, i] = - 2 / (beta**2 * h**2) - 1
            a[i, i + 1] = 1 / (beta**2 * h**2) - dV_bias / (beta * 2 * h)

        # boundary condition
        a[idx_ts, :] = 0
        for i in idx_ts:
            a[i, i] = 1
        b[idx_ts] = 1

        # stability: Psi shall be flat on the boundary
        a[0, :] = 0
        a[0, 0] = 1
        a[0, 1] = -1
        b[0] = 0

        a[N - 1, :] = 0
        a[N - 1 , N - 1] = 1
        a[N - 1 , N - 2] = -1
        b[N - 1] = 0

        # Apply the Shortley-Welley method. Psi solves the following linear system of equations
        Psi = np.linalg.solve(a, b)

        # compute the free energy which corresponds to the value function at the optimal control
        F =  - np.log(Psi) / beta

        # compute the optimal control u_opt = -sqrt(2) grad F with
        # finite differences : u_opt(x_k) = -sqrt(2) (F(x_{k+1}) - F(x_{k+1})) / 2h
        u_opt = np.zeros(N)
        u_opt[1: -1] = - np.sqrt(2) * (F[2:] - F[: -2]) / (2 * h)
        u_opt[0] = u_opt[1]
        u_opt[-1] = u_opt[-2]

        # save variables
        self.h = h
        self.omega_h = omega_h
        self.Psi = Psi
        self.F = F
        self.u_opt = u_opt
