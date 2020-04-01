import numpy as np

from tools import double_well_1d_potential, \
                  gradient_double_well_1d_potential, \
                  normal_probability_density, \
                  derivative_normal_probability_density, \
                  bias_potential
from datetime import datetime
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

class langevin_1d:
    '''
    '''

    def __init__(self, seed, beta, xzero, is_drifted, target_set, num_trajectories):
        '''
        '''
        
        self._seed = seed

        # sde parameters
        self._beta = beta
        self._xzero = xzero
        self._is_drifted = is_drifted 

        # sampling
        self._M = num_trajectories
        if target_set[0] >= target_set[1]:
            #TODO raise error
            print("The target set interval is not valid")
        self._target_set_min = target_set[0]
        self._target_set_max = target_set[1]

        # Euler-Majurama
        self._tzero = 0
        self._T = 10**3
        self._N = 10**6
        self._dt = (self._T - self._tzero) / self._N
       
        # ansatz functions (gaussians) and coefficients
        self._a = None
        self._mus = None
        self._sigmas = None 
        
        # variables
        self._fhs = None
        self._fht = None
        self._I = None
        self._M_fht = None
       
        # mean, variance and re
        self._mean_fht = None
        self._var_fht = None
        self._re_fht = None
        self._mean_I = None
        self._var_I = None
        self._re_I = None
        self._mean_M = None
       
    def preallocate_variables(self):
        M = self._M

        # first hitting steps/times
        self._fhs = np.empty(M)
        self._fht = np.empty(M)
        self._fhs[:] = np.NaN
        self._fht[:] = np.NaN

        # observable of interest (sampling)
        self._I = np.empty(M)
        self._I[:] = np.NaN

        if self._is_drifted: 
            # preallocate Girsanov Martingale at fht
            self._M_fht = np.empty(M)
            self._M_fht[:] = np.NaN
       
        #TODO variables for the SOC
        # cost control (soc)
        #cost = np.ones(M)

        # gradient of S_h

        # sum partial of g
    
    def get_a_coefficients_metadynamcs(self, m, J_min, J_max):
        '''
        '''
        # validate input 
        if J_min >= J_max:
            #TODO raise error
            print("The interval J_h is not valid")
       
        # sampling parameters
        beta = self._beta
        
        # load metadynamics parameters
        #bias_pot_coeff = np.load(os.path.join(DATA_PATH, 'langevin1d_metadynamic.npz'))
        bias_pot_coeff = np.load(os.path.join(DATA_PATH, 'langevin1d_tilted_potential.npz'))
        meta_omegas = bias_pot_coeff['omegas']
        meta_mus = bias_pot_coeff['mus']
        meta_sigmas = bias_pot_coeff['sigmas']

        # define a coefficients 
        a = np.zeros(m)
        
        # grid on the interval J_h = [J_min, J_max].
        X = np.linspace(J_min, J_max, m)
        # step size
        h = (J_max - J_min) / m

        # the ansatz functions are gaussians with standard deviation sigma
        # and means uniformly spaced in J_h
        mus = X
        sigmas = 0.1 * np.ones(m)
        
        # ansatz functions evaluated at the grid
        ansatz_functions = np.zeros((m, m))
        for i in np.arange(m):
            for j in np.arange(m):
                ansatz_functions[i, j] = normal_probability_density(X[j], mus[i], sigmas[i])

        # value function evaluated at the grid
        phi = np.zeros(m)
        for i in np.arange(m):
            V_bias = bias_potential(X[i], meta_mus, meta_sigmas, meta_omegas)
            phi[i] = V_bias * beta / 2

        # solve a V = \Phi
        a = np.linalg.solve(ansatz_functions, phi)
        
        self._a = a
        self._mus = mus 
        self._sigmas = sigmas 

    def ansatz_functions(self, x):
        '''This method computes the ansatz functions evaluated at x

        Args:
            x (float) : position
        '''
        mus = self._mus
        sigmas = self._sigmas

        # preallocate ansatz functions
        m = len(mus)
        v = np.zeros(m)
        
        # evaluates ansatz function at x
        for j in np.arange(m):
            v[j] = normal_probability_density(x, mus[j], sigmas[j]) 

        return v

    def value_function(self, x):
        '''This method computes the value function evaluated at x

        Args:
            x (float) : position
        '''
        # a coefficients
        a = self._a

        # ansatz functions
        v = self.ansatz_functions(x)
        
        return np.dot(a, v)


    def control_basis_functions(self, x):
        '''This method computes the control basis functions evaluated at x

        Args:
            x (float) : position
        '''
        # sampling parameters
        beta = self._beta

        # ansatz functions
        mus = self._mus
        sigmas = self._sigmas

        # preallocate basis functions
        m = len(mus)
        b = np.zeros(m)
        
        for i in np.arange(m):
            b[i] = - np.sqrt(2 / beta) * derivative_normal_probability_density(x, mus[i], sigmas[i]) 

        return b

    def control(self, x):
        '''This method computes the control evaluated at x

        Args:
            x (float) : position
        '''
        # a coefficients
        a = self._a

        # control basis functions
        b = self.control_basis_functions(x)

        return np.dot(a, b)

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x (float) : position
        '''
        # sampling parameters
        beta = self._beta

        Vbias_at_x = (2 / beta) * self.value_function(x)
        
        return double_well_1d_potential(x) + Vbias_at_x

    def tilted_gradient(self, x):
        '''This method computes the tilted gradient at x

        Args:
            x (float) : position
        '''
        # sampling parameters
        beta = self._beta

        dVbias_at_x = - np.sqrt(2 / beta) * self.control(x)
        
        return gradient_double_well_1d_potential(x) + dVbias_at_x

    def sample(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        is_drifted = self._is_drifted
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max

        for i in np.arange(M):
            # initialize Xtemp
            Xtemp = xzero

            if is_drifted:
                # initialize martingale terms, M_t = e^(M1_t + M2_t)
                M1temp = 0
                M2temp = 0
            
            # Brownian increments
            dB = np.sqrt(dt) * np.random.normal(0, 1, N)
            
            for n in np.arange(1, N+1):
                # compute gradient
                if not is_drifted:
                    gradient_at_x = gradient_double_well_1d_potential(Xtemp)
                else:
                    gradient_at_x = self.tilted_gradient(Xtemp)

                # SDE iteration
                drift = - gradient_at_x * dt
                diffusion = np.sqrt(2 / beta) * dB[n-1]
                Xtemp = Xtemp + drift + diffusion

                if is_drifted:
                    # evaluate control at Xtemp
                    utemp = self.control(Xtemp)

                    # compute martingale terms
                    # M1temp = int_0^fht (-u_t dB_t)
                    # M2temp = int_0^fht (- 1/2 (u_t)^2 dt)
                    M1temp = M1temp - utemp * dB[n-1]
                    M2temp = M2temp - 0.5 * (utemp ** 2) * dt 

                # check if we have arrived to the target set
                if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                    fht = n * dt

                    # save first hitting time/step
                    self._fhs[i] = n 
                    self._fht[i] = fht

                    if not is_drifted:
                        # save quantity of interest at the fht
                        self._I[i] = np.exp(-beta * fht)
                    else:
                        # save quantity of interest at the fht
                        self._I[i] = np.exp(-beta * fht + M1temp + M2temp) 

                        # save Girsanov Martingale at fht
                        self._M_fht[i] = np.exp(M1temp + M2temp)
                    break


    def save_variables(self, i, n, x):
        # TODO: deprecated method!
        dt = self._dt
        beta = self._beta
        is_drifted = self._is_drifted


    def compute_statistics(self):
        is_drifted = self._is_drifted

        # sort out trajectories which have not arrived
        fht = np.array([t for t in self._fht if not np.isnan(t)])
        I = np.array([x for x in self._I if not np.isnan(x)])
        if is_drifted:
            M_fht = np.array([x for x in self._M_fht if not np.isnan(x)])
    
        # compute mean and variance of tau
        mean_fht = np.mean(fht)
        var_fht = np.var(fht)
        if mean_fht != 0:
            re_fht = np.sqrt(var_fht) / mean_fht
        else:
            re_fht = np.nan
        self._mean_fht = mean_fht
        self._var_fht = var_fht
        self._re_fht = re_fht

        # compute mean and variance of I
        mean_I = np.mean(I)
        var_I = np.var(I)
        if mean_I != 0:
            re_I = np.sqrt(var_I) / mean_I
        else:
            re_I = np.nan
        self._mean_I = mean_I
        self._var_I = var_I
        self._re_I = re_I

        if is_drifted:
            # compute mean of M_fht
            self._mean_M = np.mean(M_fht)


    def save_statistics(self):
        is_drifted = self._is_drifted

        # save output in a file
        time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(DATA_PATH, 'langevin_1d_2well_' + time_stamp + '.txt')
        f = open(file_path, "w")

        f.write('drifted process: {}\n'.format(self._is_drifted))
        f.write('beta: {:2.1f}\n'.format(self._beta))
        f.write('dt: {:2.4f}\n'.format(self._dt))
        f.write('Y_0: {:2.1f}\n'.format(self._xzero))
        f.write('target set: [{:2.1f}, {:2.1f}]\n\n'
                ''.format(self._target_set_min, self._target_set_max))

        f.write('sampled trajectories: {:d}\n\n'.format(self._M))
        f.write('% trajectories which have arrived: {:2.2f}\n\n'
                ''.format(len(self._fht) / self._M))

        f.write('Expectation of fht: {:2.4f}\n'.format(self._mean_fht))
        f.write('Variance of fht: {:2.4f}\n'.format(self._var_fht))
        f.write('Relative error of fht: {:2.4f}\n\n'.format(self._re_fht))
        
        f.write('Expectation of exp(-beta * fht): {:2.4e}\n'.format(self._mean_I))
        f.write('Variance of exp(-beta * fht): {:2.4e}\n'.format(self._var_I))
        f.write('Relative error of exp(-beta * fht): {:2.4e}\n\n'.format(self._re_I))

        if is_drifted:
            f.write('Expectation of M_fht: {:2.4e}\n'.format(self._mean_M))
    
        f.close()
