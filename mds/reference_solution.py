from potentials_and_gradients import double_well_1d_gradient

import numpy as np

class langevin_1d_reference_solution():
    def __init__(self, beta, target_set_min, target_set_max):
        if target_set_min >= target_set_max:
            #TODO raise error
            print("The target set interval is not valid")

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
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max

        # discretization of the state space
        omega_min = -2
        omega_max = 2
        h = 0.01
        N = int((omega_max - omega_min) / h + 1)
        omega_h = np.linspace(omega_min, omega_max, N)
        target_set_h_idx = np.where((omega_h >= target_set_min) & (omega_h <= target_set_max))
        target_set_h = omega_h[target_set_h_idx]

        # discretization of the operator (epsionL -1), where L 
        # is the infinitessimal generator of the not drifted langevin 1d
        # L = - dV_bias d/dx + epsilon d^2/dx^2

        A = np.zeros((N, N))
        B = np.zeros(N)

        # weights of Psi
        for i in np.arange(1, N-1):
            dV_bias = double_well_1d_gradient(omega_h[i])
            A[i, i - 1] = 1 / (beta**2 * h**2) + dV_bias / (beta * 2 * h)
            A[i, i] = - 2 / (beta**2 * h**2) - 1
            A[i, i + 1] = 1 / (beta**2 * h**2) - dV_bias / (beta * 2 * h)

        # boundary condition
        A[target_set_h_idx, :] = 0
        for i in target_set_h_idx:
            A[i, i] = 1
        B[target_set_h_idx] = 1
        
        # stability: Psi shall be flat on the boundary
        A[0, :] = 0
        A[0, 0] = 1 
        A[0, 1] = -1 
        B[0] = 0
        
        A[N - 1, :] = 0
        A[N - 1 , N - 1] = 1 
        A[N - 1 , N - 2] = -1 
        B[N - 1] = 0
        
        # Apply the Shortley-Welley method. Psi solves the following linear system of equations
        Psi = np.linalg.solve(A, B)

        # compute the free energy which corresponds to the value function at the optimal control
        F =  - np.log(Psi) / beta

        # compute the optimal control: u_opt = -sqrt(2) grad F 
        u_opt = np.zeros(N)
        u_opt[1:] = - np.sqrt(2) * (F[1:] - F[:-1]) / h
        u_opt[0] = u_opt[1]

        # save variables
        self.h = h
        self.omega_h = omega_h
        self.Psi = Psi
        self.F = F
        self.u_opt = u_opt
