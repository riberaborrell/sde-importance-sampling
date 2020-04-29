from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient

import numpy as np

class langevin_1d_reference_solution():
    def __init__(self, beta, target_set_min, target_set_max):
        self.beta = beta
        self.target_set_min = target_set_min
        self.target_set_max = target_set_max

    def compute_reference_solution(self):
        ''' This method computes the solution of the BVP associated
            to the langevin equation by using the Shortley-Welley method.
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
        A[0, 0] = 1 
        A[0, 1] = -1 
        B[0] = 0
        
        A[N - 1 , N - 1] = 1 
        A[N - 1 , N - 2] = -1 
        B[N - 1] = 0

        Psi = np.linalg.solve(A, B)
        F =  - np.log(Psi)
        u_optimal = np.zeros(N)
        u_optimal[1:] = np.sqrt(2 / beta) * (F[1:] - F[:-1]) / h
        u_optimal[0] = u_optimal[1]

        self.h = h
        self.omega_h = omega_h
        self.Psi = Psi
        self.F = F
        self.u_optimal = u_optimal
