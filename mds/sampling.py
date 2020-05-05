from decorators import timer
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient, \
                                     one_well_1d_potential, \
                                     one_well_1d_gradient, \
                                     derivative_normal_pdf, \
                                     bias_potential
from plotting import Plot
from reference_solution import langevin_1d_reference_solution 

import numpy as np
from scipy import stats
from datetime import datetime
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

class langevin_1d:
    '''
    '''

    def __init__(self, beta, is_drifted=False):
        '''
        '''
        #seed
        self.seed = None

        # sde parameters
        self.beta = beta
        self.is_drifted = is_drifted 

        # sampling
        self.xzero = None
        self.M = None
        self.target_set_min = None
        self.target_set_max = None

        # Euler-Majurama
        self.dt = None
        self.N = None

        # ansatz functions (gaussians) and coefficients
        self.m = None
        self.a = None
        self.a_optimal = None
        self.mus = None
        self.sigmas = None 
        
        # variables
        self.fht = None

        # sampling problem
        self.is_sampling_problem = None
        self.Psi = None
        
        self.G_N = None
        self.G_fht = None
        
        self.Psi_rew = None

        # soc problem
        self.is_soc_problem = None
        self.cost = None
        self.J = None
        self.gradJ = None
        self.gradSh = None

        # mean, variance and re
        self.first_fht = None
        self.last_fht = None 
        self.mean_fht = None
        self.var_fht = None
        self.re_fht = None

        self.mean_Psi = None
        self.var_Psi = None
        self.re_Psi = None

        self.mean_G_fht = None
        self.mean_G_N= None
        
        self.mean_Psi_rew = None
        self.var_Psi_rew = None
        self.re_Psi_rew = None

        self.mean_J = None

    def set_ansatz_functions(self, mus, sigmas):
        '''This method sets the mean and the standard deviation of the 
           ansatz functions 

        Args:
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert mus.shape == sigmas.shape, "Error"

        self.m = mus.shape[0] 
        self.mus = mus
        self.sigmas = sigmas
    
    def set_uniformly_dist_ansatz_functions(self, m, sigma):
        '''This method sets the number of ansatz functions and their mean
           and standard deviation. The means will be uniformly distributed
           in the set J and the standard deviation is given.

        Args:
            m (int): number of ansatz functions
            sigma (float) : standard deviation
        '''
        J_min = -1.9
        J_max = 0.9
        
        self.m = m
        self.mus = np.linspace(J_min, J_max, m)
        self.sigmas = sigma * np.ones(m)
   
    def set_bias_potential(self, a, mus, sigmas):
        '''
        Args:
            a (ndarray): parameters
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert a.shape == mus.shape == sigmas.shape, "Error"
        self.is_drifted = True
        self.m = a.shape[0] 
        self.a = a
        self.mus = mus
        self.sigmas = sigmas 
    
    def set_bias_potential_from_metadynamics(self):
        # load metadynamics parameters
        bias_pot_coeff = np.load(
            os.path.join(DATA_PATH, 'langevin1d_bias_potential_fake_metadynamics.npz')
            #os.path.join(DATA_PATH, 'langevin1d_bias_potential_metadynamics.npz')
        )
        omegas = bias_pot_coeff['omegas']
        meta_mus = bias_pot_coeff['mus']
        meta_sigmas = bias_pot_coeff['sigmas']
        
        assert omegas.shape == meta_mus.shape == meta_sigmas.shape, "Error"
    
        a = omegas / 2
        
        self.is_drifted = True
        self.m = a.shape[0]
        self.a = a
        self.mus = meta_mus
        self.sigmas = meta_sigmas

    def set_a_from_metadynamics(self):
        '''
        '''
        #TODO assert self.m, self.mus, self.sigmas
        m = self.m
        mus = self.mus
        sigmas= self.sigmas

        # load metadynamics parameters
        bias_pot_coeff = np.load(
            os.path.join(DATA_PATH, 'langevin1d_bias_potential_fake_metadynamics.npz')
            #os.path.join(DATA_PATH, 'langevin1d_bias_potential_metadynamics.npz')
        )
        omegas = bias_pot_coeff['omegas']
        meta_mus = bias_pot_coeff['mus']
        meta_sigmas = bias_pot_coeff['sigmas']
        
        assert omegas.shape == meta_mus.shape == meta_sigmas.shape, "Error"

        # define a coefficients 
        a = np.zeros(m)
        
        # grid on the interval J = [J_min, J_max].
        X = mus

        # ansatz functions evaluated at the grid
        ansatz_functions = self.ansatz_functions(X, mus, sigmas)

        # value function evaluated at the grid
        V_bias = bias_potential(X, omegas, meta_mus, meta_sigmas)
        phi = V_bias / 2

        # solve a V = \Phi
        a = np.linalg.solve(ansatz_functions, phi)

        self.is_drifted = True
        self.a = a

    def set_a_optimal(self):
        sol = langevin_1d_reference_solution(
            beta=self.beta,
            target_set_min=0.9,
            target_set_max=1.1,
        )
        sol.compute_reference_solution()

        X = sol.omega_h
        a = self.ansatz_functions(X).T
        b = sol.F

        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

        self.a_optimal = x

    def ansatz_functions(self, x, mus=None, sigmas=None):
        '''This method computes the ansatz functions evaluated at x

        Args:
            x (float or ndarray) : position/s
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        if mus is None and sigmas is None:
            mus = self.mus
            sigmas = self.sigmas

        assert mus.shape == sigmas.shape, "Error"

        if type(x) == np.ndarray:
            mus = mus.reshape(mus.shape[0], 1)
            sigmas = sigmas.reshape(sigmas.shape[0], 1)

        return stats.norm.pdf(x, mus, sigmas)

    def value_function(self, x, a=None, mus=None, sigmas=None):
        '''This method computes the value function evaluated at x

        Args:
            x (float or ndarray) : position/s
            a (ndarray): parameters 
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        
        Return:
            b ((m,)-ndarray or (m, M)-ndarray)
        '''
        if mus is None and sigmas is None:
            mus = self.mus
            sigmas = self.sigmas
        
        if a is None:
            a = self.a

        assert a.shape == mus.shape == sigmas.shape, "Error"

        # ansatz functions
        v = self.ansatz_functions(x, mus, sigmas)
        
        return np.dot(a, v)

    def control_basis_functions(self, x, mus=None, sigmas=None):
        '''This method computes the control basis functions evaluated at x

        Args:
            x (float or ndarray) : position/s
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian

        Return:
            b ((m,)-ndarray or (m, M)-ndarray)
        '''
        # sampling parameters
        beta = self.beta

        if mus is None and sigmas is None:
            mus = self.mus
            sigmas = self.sigmas

        assert mus.shape == sigmas.shape, "Error"

        if type(x) == np.ndarray:
            mus = mus.reshape(mus.shape[0], 1)
            sigmas = sigmas.reshape(sigmas.shape[0], 1)

        return - np.sqrt(2) * derivative_normal_pdf(x, mus, sigmas)

    def control(self, x, a=None, mus=None, sigmas=None):
        '''This method computes the control evaluated at x

        Args:
            X (float or ndarray) : position/s
            a (ndarray): parameters 
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        if mus is None and sigmas is None:
            mus = self.mus
            sigmas = self.sigmas
        
        if a is None:
            a = self.a

        assert a.shape == mus.shape == sigmas.shape, "Error"

        # control basis functions at x
        b = self.control_basis_functions(x, mus, sigmas)

        return np.dot(a, b)

    def bias_potential(self, x):
        '''This method computes the bias potential at x

        Args:
            x (float or ndarray) : position
        '''
        return 2 * self.value_function(x)

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u (float or ndarray) : control at x
        '''
        return - np.sqrt(2) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x (float or ndarray) : position
        '''
        return double_well_1d_potential(x) + self.bias_potential(x)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x (float or ndarray) : position
            u (float or ndarray) : control at x
        '''
        assert type(x) == type(u), "Error"
        if type(x) == np.ndarray:
            assert x.shape == u.shape, "Error"

        return double_well_1d_gradient(x) + self.bias_gradient(u)
    
    def set_sampling_parameters(self, xzero, M, target_set, dt, N, seed=None):
        '''
        '''
        # set random seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.xzero = xzero
        self.M = M 
        if target_set[0] >= target_set[1]:
            #TODO raise error
            print("The target set interval is not valid")
        self.target_set_min = target_set[0]
        self.target_set_max = target_set[1]

        # Euler-Majurama
        self.dt = dt
        self.N = N

    def preallocate_variables(self, is_sampling_problem=False, 
                              is_soc_problem=False):
        '''
        '''
        N = self.N
        M = self.M
        m = self.m

        self.fht = np.empty(M)
        self.fht[:] = np.NaN
        
        if is_sampling_problem and not self.is_drifted:
            self.Psi = np.empty(M)
            self.Psi[:] = np.NaN

        elif is_sampling_problem and self.is_drifted:
            self.G_fht = np.empty(M, )
            self.G_fht[:] = np.NaN
            self.G_N = np.empty(M)
            self.G_N[:] = np.NaN

            self.Psi_rew = np.empty(M)
            self.Psi_rew[:] = np.NaN

        if is_soc_problem:
            self.J = np.empty(M)
            self.J[:] = np.NaN
            self.gradJ= np.empty((m, M))
            self.gradJ[:] = np.NaN
        
        self.is_sampling_problem = is_sampling_problem 
        self.is_soc_problem = is_soc_problem 

    @timer
    def sample_not_drifted(self):
        M = self.M
        N = self.N
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max

        self.preallocate_variables(is_sampling_problem=True)

        for i in np.arange(M):
            # initialize Xtemp
            Xtemp = xzero
            
            for n in np.arange(1, N+1):
                # Brownian increment
                dB = np.sqrt(dt) * np.random.normal(0, 1)

                # compute gradient
                gradient = double_well_1d_gradient(Xtemp)

                # SDE iteration
                drift = - gradient * dt
                diffusion = np.sqrt(2 / beta) * dB
                Xtemp = Xtemp + drift + diffusion
                
                # check if we have arrived to the target set
                if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                    fht = n * dt

                    # save first hitting time
                    self.fht[i] = fht

                    # save quantity of interest at the fht
                    self.Psi[i] = np.exp(-beta * fht)
                    break
    
    @timer
    def sample_not_drifted_vectorized(self):
        M = self.M
        N = self.N
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max

        self.preallocate_variables(is_sampling_problem=True)

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)
        
        # has arrived in target set
        been_in_target_set = np.repeat([False], M)
        
        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1, M)

            # compute gradient
            gradient = double_well_1d_gradient(Xtemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion
           
            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))
           
            # indices of trajectories new in the target set
            new_in_target_set_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]

            # update trajectories which have been in the target set
            been_in_target_set[new_in_target_set_idx] = True
            
            # save first hitting time
            self.fht[new_in_target_set_idx] = n * dt
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break
        
        # save quantity of interest at the fht
        self.Psi = np.exp(-beta * self.fht)
    
    @timer
    def sample_drifted(self):
        
        M = self.M
        N = self.N
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max
        
        self.preallocate_variables(is_sampling_problem=True)

        k = 10

        for i in np.arange(M):
            # initialize Xtemp
            Xtemp = xzero
            
            # initialize Girsanov Martingale terms, G_t = e^(G1_t + G2_t)
            G1temp = 0
            G2temp = 0
            
            for n in np.arange(1, N+1):
                # Brownian increment
                dB = np.sqrt(dt) * np.random.normal(0, 1)
                
                # compute control at Xtemp
                utemp = self.control(Xtemp)

                # compute gradient
                gradient = self.tilted_gradient(Xtemp, utemp)

                # SDE iteration
                drift = - gradient * dt
                diffusion = np.sqrt(2 / beta) * dB
                Xtemp = Xtemp + drift + diffusion
                
                # compute Girsanov Martingale terms
                # G1_t = int_0^fht -u_t dB_t
                # G2_t = int_0^fht - 1/2 (u_t)^2 dt
                G1temp = G1temp - np.sqrt(1 / beta) * utemp * dB
                G2temp = G2temp - (1 / beta) * 0.5 * (utemp ** 2) * dt 
                
                # save Girsanov Martingale at time k
                if n == k: 
                    self.G_N[i] = np.exp(G1temp + G2temp)

                # check if we have arrived to the target set
                if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                    fht = n * dt

                    # save first hitting time
                    self.fht[i] = fht

                    # save quantity of interest at the fht
                    self.Psi[i] = np.exp(-beta * fht)

                    # save Girsanov Martingale at time k
                    self.G_fht[i] = np.exp(G1temp + G2temp)

                    # save re-weighted first hitting time
                    self.fht_rew[i] = fht * np.exp(G1temp + G2temp)

                    # save re-weighted quantity of interest
                    self.Psi_rew[i] = np.exp(-beta * fht + G1temp + G2temp) 
                    break

    @timer
    def sample_drifted_vectorized(self):
        M = self.M
        N = self.N
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max
        
        self.preallocate_variables(is_sampling_problem=True)

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)
        
        # initialize Girsanov Martingale terms, G_t = e^(G1_t + G2_t)
        G1temp = np.zeros(M)
        G2temp = np.zeros(M)

        # has arrived in target set
        been_in_target_set = np.repeat([False], M)
        
        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1, M)
            
            # control at Xtemp
            utemp = self.control(Xtemp)

            # compute gradient
            gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion

            # Girsanov Martingale terms
            G1temp = G1temp - np.sqrt(1 / beta) * utemp * dB
            G2temp = G2temp - (1 / beta) * 0.5 * (utemp ** 2) * dt 
            
            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))

            # indices of trajectories new in the target set
            new_in_target_set_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]
                
            # update list of indices whose trajectories have been in the target set
            been_in_target_set[new_in_target_set_idx] = True
            
            # save first hitting time
            self.fht[new_in_target_set_idx] = n * dt
            #self.fht_rew[new_in_target_set_idx] = fht * np.exp(
            #    G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            #)
            #self.Psi_rew[new_in_target_set_idx] = np.exp(
            #    - beta * n * dt
            #    + G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            #)
            
            # save Girsanov Martingale
            self.G_fht[new_in_target_set_idx] = np.exp(
                G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            )
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                # save Girsanov Martingale at the time when the last trajectory arrive
                self.G_N = np.exp(G1temp + G2temp)
                break

        # save rew fht
        self.fht_rew = self.fht * self.G_fht

        # save quantity of interest at the fht
        self.Psi = np.exp(-beta * self.fht)
        self.Psi_rew = np.exp(-beta * self.fht) * self.G_fht
    
    @timer
    def sample_soc(self):
        M = self.M
        N = self.N
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max
        m = self.m
        
        self.preallocate_variables(is_soc_problem=True)


        for i in np.arange(M):
            # initialize Xtemp
            Xtemp = xzero
            
            # compute control at Xtemp
            utemp = self.control(Xtemp)

            # bla
            cost = 0
            sum_partial_tilde_gh = np.zeros(m)
            grad_Sh = np.zeros(m)
            
            for n in np.arange(1, N+1):
                # Brownian increment
                dB = np.sqrt(dt) * np.random.normal(0, 1)

                # compute gradient
                gradient = self.tilted_gradient(Xtemp, utemp)

                # SDE iteration
                drift = - gradient * dt
                diffusion = np.sqrt(2 / beta) * dB
                Xtemp = Xtemp + drift + diffusion
                
                # compute control at Xtemp
                utemp = self.control(Xtemp)

                # evaluate the control basis functions at Xtmep
                btemp = self.control_basis_functions(Xtemp)
                    
                # compute cost, ...
                cost = cost + 0.5 * (utemp ** 2) * dt
                sum_partial_tilde_gh = sum_partial_tilde_gh + utemp * btemp * dt  
                grad_Sh = grad_Sh + np.random.normal(0, 1) * btemp
                
                # check if we have arrived to the target set
                if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                    fht = n * dt

                    # save first hitting time
                    self.fht[i] = fht

                    self.J[i] = cost + fht
                    grad_Sh = grad_Sh * (- np.sqrt(dt * beta / 2))
                    self.gradJ[i, :] = sum_partial_tilde_gh - (cost + fht) * grad_Sh
                    
                    break
    
    @timer
    def sample_soc_vectorized(self):
        M = self.M
        N = self.N
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min = self.target_set_min
        target_set_max = self.target_set_max
        m = self.m
        
        self.preallocate_variables(is_soc_problem=True)

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)
            
        # bla
        cost = np.zeros(M)
        sum_grad_gh = np.zeros((m, M))
        grad_Sh = np.zeros((m, M))
            
        # has arrived in target set
        been_in_target_set = np.repeat([False], M)

        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1, M)
            
            # compute control at Xtemp
            utemp = self.control(Xtemp)

            # compute gradient
            gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp += drift + diffusion
                
            # evaluate the control basis functions at Xtmep
            btemp = self.control_basis_functions(Xtemp)
                    
            # compute cost, ...
            cost += 0.5 * (utemp ** 2)
            sum_grad_gh += dt * utemp * btemp 
            grad_Sh += beta * np.sqrt(dt / 2) * np.random.normal(0, 1, M) * btemp
                
            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))
            
            # indices of trajectories new in the target set
            new_in_target_set_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]
            if len(new_in_target_set_idx) == 0:
                continue
            
            # update list of indices whose trajectories have been in the target set
            been_in_target_set[new_in_target_set_idx] = True

            # save first hitting time
            fht = n * dt
            self.fht[new_in_target_set_idx] = fht
            self.J[new_in_target_set_idx] = dt * (fht + cost[new_in_target_set_idx])
            self.gradJ[:, new_in_target_set_idx] = sum_grad_gh[:, new_in_target_set_idx] - \
                                                    dt * (fht + cost[new_in_target_set_idx]) * \
                                                    grad_Sh[:, new_in_target_set_idx]
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break


    def save_variables(self, i, n, x):
        # TODO: deprecated method!
        dt = self.dt
        beta = self.beta
        is_drifted = self.is_drifted

    def compute_mean_variance_and_rel_error(self, x):
        '''This method computes the mean, the variance and the relative
           error of the given ndarray x
        Args:
            x (ndarray) :
        '''
        # check that dim of x is not 0
        if x.shape[0] == 0:
            return np.nan, np.nan, np.nan 

        mean = np.mean(x)
        var = np.var(x)
        if mean != 0:
            re = np.sqrt(var) / mean
        else:
            re = np.nan

        return mean, var, re 

    def compute_statistics(self):
        # sort out trajectories which have not arrived
        self.fht = self.fht[np.where(np.isnan(self.fht) != True)]

        # first and last fht
        self.first_fht = np.min(self.fht)
        self.last_fht = np.max(self.fht)

        # compute mean and variance of fht
        self.mean_fht, \
        self.var_fht, \
        self.re_fht = self.compute_mean_variance_and_rel_error(self.fht)

        if self.is_sampling_problem and not self.is_drifted:
            # compute mean and variance of Psi
            self.Psi = self.Psi[np.where(np.isnan(self.Psi) != True)]
            self.mean_Psi, \
            self.var_Psi, \
            self.re_Psi = self.compute_mean_variance_and_rel_error(self.Psi)
        
        elif self.is_sampling_problem and self.is_drifted:
            # compute mean of M_fht
            self.G_fht = self.G_fht[np.where(np.isnan(self.G_fht) != True)]
            self.mean_G_fht = np.mean(self.G_fht)
            
            # compute mean of M_N
            self.G_N = self.G_N[np.where(np.isnan(self.G_N) != True)]
            self.mean_G_N= np.mean(self.G_N)
            
            # compute mean and variance of Psi re_weighted
            self.Psi_rew = self.Psi_rew[np.where(np.isnan(self.Psi_rew) != True)]
            self.mean_Psi_rew, \
            self.var_Psi_rew, \
            self.re_Psi_rew = self.compute_mean_variance_and_rel_error(self.Psi_rew)
        
        if self.is_soc_problem:
            # compute mean of J
            #self.J = self.J[np.where(np.isnan(self.J) != True)]
            self.mean_J = np.mean(self.J)

            # compute mean of gradJ
            #self.gradJ = self.gradJ[np.where(np.isnan(self.gradJ) != True)]
            self.mean_gradJ = np.mean(self.gradJ, axis=1)


    def save_statistics(self):
        '''
        '''
        # set path
        time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(
            DATA_PATH, 
            'langevin_1d_2well_' + time_stamp + '.txt',
        )

        # write in file
        f = open(file_path, "w")

        f.write('drifted process: {}\n'.format(self.is_drifted))
        f.write('beta: {:2.1f}\n'.format(self.beta))
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('Y_0: {:2.1f}\n'.format(self.xzero))
        f.write('target set: [{:2.1f}, {:2.1f}]\n\n'
                ''.format(self.target_set_min, self.target_set_max))

        f.write('sampled trajectories: {:d}\n'.format(self.M))
        f.write('time steps: {:d}\n\n'.format(self.N))

        f.write('% trajectories which have arrived: {:2.2f}\n\n'
                ''.format(100 * len(self.fht) / self.M))
        
        f.write('E[fhs] = {:.2f}\n\n'.format(self.mean_fht / self.dt))

        f.write('first fht = {:2.4f}\n'.format(self.first_fht))
        f.write('last fht = {:2.4f}\n'.format(self.last_fht))
        f.write('E[fht] = {:2.4f}\n'.format(self.mean_fht))
        f.write('Var[fht] = {:2.4f}\n'.format(self.var_fht))
        f.write('RE[fht] = {:2.4f}\n\n'.format(self.re_fht))
       
        if self.is_sampling_problem and not self.is_drifted:
            f.write('E[exp(-beta * fht)] = {:2.4e}\n'.format(self.mean_Psi))
            f.write('Var[exp(-beta * fht)] = {:2.4e}\n'.format(self.var_Psi))
            f.write('RE[exp(-beta * fht)] = {:2.4e}\n\n'.format(self.re_Psi))

        elif self.is_sampling_problem and self.is_drifted:
            f.write('E[M_fht] = {:2.4e}\n'.format(self.mean_G_fht))
            f.write('E[M_N]: {:2.4e}\n\n'.format(self.mean_G_N))
            
            f.write('E[exp(-beta * fht) * M_fht] = {:2.4e}\n'
                    ''.format(self.mean_Psi_rew))
            f.write('Var[exp(-beta * fht) * M_fht] = {:2.4e}\n'
                    ''.format(self.var_Psi_rew))
            f.write('RE[exp(-beta * fht) * M_fht] = {:2.4e}\n\n'
                    ''.format(self.re_Psi_rew))
        
        if self.is_soc_problem:
            f.write('E[Jh] = {:2.4e}\n'.format(self.mean_J))
            for j in np.arange(self.m):
                f.write('E[(grad_Jh)j] = {:2.4e}\n'.format(self.mean_gradJ[j]))
    
        f.close()

    def plot_potential_and_gradient(self, file_name, dir_path=FIGURES_PATH):
        X = np.linspace(-2, 2, 100)
        V = double_well_1d_potential(X)
        dV = double_well_1d_gradient(X)

        if self.is_drifted:
            Vbias = self.bias_potential(X)
            U = self.control(X)
            dVbias = self.bias_gradient(U)
        else:
            Vbias = np.zeros(X.shape[0])
            dVbias = np.zeros(X.shape[0])

        pl = Plot(
            file_name=file_name,
            dir_path=dir_path,
        )
        pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)
