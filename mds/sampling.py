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

    def __init__(self, beta, is_drifted):
        '''
        '''
        #seed
        self._seed = None

        # sde parameters
        self._beta = beta
        self._is_drifted = is_drifted 

        # sampling
        self._xzero = None
        self._M = None
        self._target_set_min = None
        self._target_set_max = None

        # Euler-Majurama
        self._dt = None
        self._N = None

        # ansatz functions (gaussians) and coefficients
        self._m = None
        self._a = None
        self._a_optimal = None
        self._mus = None
        self._sigmas = None 
        
        # variables
        self._fht = None

        # sampling problem
        self._is_sampling_problem = None
        self._Psi = None
        self._F = None

        # soc problem
        self._is_soc_problem = None
        self._cost = None
        self._J = None
        self._gradJ = None
        self._gradSh = None

        self._do_reweighting = None
        self._G_N = None
        self._G_fht = None
        self._fht_rew = None
        
        self._Psi_rew = None
        self._F_rew = None

        self._J_rew = None
       
        # mean, variance and re
        self._first_fht = None
        self._last_fht = None 
        self._mean_fht = None
        self._var_fht = None
        self._re_fht = None

        self._mean_Psi = None
        self._var_Psi = None
        self._re_Psi = None

        self._mean_G_fht = None
        self._mean_G_N= None

        self._mean_J = None

    def set_ansatz_functions(self, mus, sigmas):
        '''This method sets the mean and the standard deviation of the 
           ansatz functions 

        Args:
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert mus.shape == sigmas.shape, "Error"

        self._m = mus.shape[0] 
        self._mus = mus
        self._sigmas = sigmas
    
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
        
        self._m = m
        self._mus = np.linspace(J_min, J_max, m)
        self._sigmas = sigma * np.ones(m)
   
    def set_bias_potential(self, a, mus, sigmas):
        '''
        Args:
            a (ndarray): parameters
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert a.shape == mus.shape == sigmas.shape, "Error"
        self._m = a.shape[0] 
        self._a = a
        self._mus = mus
        self._sigmas = sigmas 
    
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
        
        self._m = a.shape[0]
        self._a = a
        self._mus = meta_mus
        self._sigmas = meta_sigmas

    def set_a_from_metadynamics(self):
        '''
        '''
        #TODO assert self._m, self._mus, self._sigmas
        m = self._m
        mus = self._mus
        sigmas= self._sigmas

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

        self._a = a

    def set_a_optimal(self):
        sol = langevin_1d_reference_solution(
            beta=self._beta,
            target_set_min=0.9,
            target_set_max=1.1,
        )
        sol.compute_reference_solution()
        X = np.linspace(-2, 2, 399)
        b = sol.u_optimal
        a = self.control_basis_functions(X).T

        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

        self._a_optimal = x

    def ansatz_functions(self, x, mus=None, sigmas=None):
        '''This method computes the ansatz functions evaluated at x

        Args:
            x (float or ndarray) : position/s
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        if mus is None and sigmas is None:
            mus = self._mus
            sigmas = self._sigmas

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
            mus = self._mus
            sigmas = self._sigmas
        
        if a is None:
            a = self._a

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
        beta = self._beta

        if mus is None and sigmas is None:
            mus = self._mus
            sigmas = self._sigmas

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
            mus = self._mus
            sigmas = self._sigmas
        
        if a is None:
            a = self._a

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
        self._xzero = xzero
        self._M = M 
        if target_set[0] >= target_set[1]:
            #TODO raise error
            print("The target set interval is not valid")
        self._target_set_min = target_set[0]
        self._target_set_max = target_set[1]

        # Euler-Majurama
        self._dt = dt
        self._N = N

    def preallocate_variables(self, do_reweighting=False, is_sampling_problem=True,
               is_soc_problem=False):
        '''
        '''
        N = self._N
        M = self._M
        m = self._m

        self._do_reweighting = do_reweighting 
        self._is_sampling_problem = is_sampling_problem 
        self._is_soc_problem = is_soc_problem 

        self._fht = np.empty(M)
        self._fht[:] = np.NaN
        
        if self._is_sampling_problem:
            self._Psi = np.empty(M)
            self._Psi[:] = np.NaN
            #self._F = np.empty(M)
            #self._F[:] = np.NaN

        if self._is_soc_problem:
            self._J = np.empty(M)
            self._J[:] = np.NaN
            self._gradJ= np.empty((m, M))
            self._gradJ[:] = np.NaN
        
        if self._do_reweighting: 
            self._G_fht = np.empty(M, )
            self._G_fht[:] = np.NaN
            self._G_N = np.empty(M)
            self._G_N[:] = np.NaN

            self._fht_rew = np.empty(M)
            self._fht_rew[:] = np.NaN
            
            if self._is_sampling_problem:
                self._Psi_rew = np.empty(M)
                self._Psi_rew[:] = np.NaN
            
            if self._is_soc_problem:
                self._J_rew = np.empty(M)
                self._J_rew[:] = np.NaN

    @timer
    def sample_not_drifted(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max

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
                    self._fht[i] = fht

                    # save quantity of interest at the fht
                    self._Psi[i] = np.exp(-beta * fht)
                    break
    
    @timer
    def sample_not_drifted_vectorized(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max

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
            self._fht[new_in_target_set_idx] = n * dt
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break
        
        # save quantity of interest at the fht
        self._Psi = np.exp(-beta * self._fht)
    
    @timer
    def sample_drifted(self):
        
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max
        
        self.preallocate_variables(
            is_sampling_problem=True,
            do_reweighting=True,
        )

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
                    self._G_N[i] = np.exp(G1temp + G2temp)

                # check if we have arrived to the target set
                if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                    fht = n * dt

                    # save first hitting time
                    self._fht[i] = fht

                    # save quantity of interest at the fht
                    self._Psi[i] = np.exp(-beta * fht)

                    # save Girsanov Martingale at time k
                    self._G_fht[i] = np.exp(G1temp + G2temp)

                    # save re-weighted first hitting time
                    self._fht_rew[i] = fht * np.exp(G1temp + G2temp)

                    # save re-weighted quantity of interest
                    self._Psi_rew[i] = np.exp(-beta * fht + G1temp + G2temp) 
                    break

    @timer
    def sample_drifted_vectorized(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max
        
        self.preallocate_variables(
            is_sampling_problem=True,
            do_reweighting=True,
        )

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
            self._fht[new_in_target_set_idx] = n * dt
            #self._fht_rew[new_in_target_set_idx] = fht * np.exp(
            #    G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            #)
            #self._Psi_rew[new_in_target_set_idx] = np.exp(
            #    - beta * n * dt
            #    + G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            #)
            
            # save Girsanov Martingale
            self._G_fht[new_in_target_set_idx] = np.exp(
                G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            )
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                # save Girsanov Martingale at the time when the last trajectory arrive
                self._G_N = np.exp(G1temp + G2temp)
                break

        # save rew fht
        self._fht_rew = self._fht * self._G_fht

        # save quantity of interest at the fht
        self._Psi = np.exp(-beta * self._fht)
        self._Psi_rew = np.exp(-beta * self._fht) * self._G_fht
    
    @timer
    def sample_soc(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max
        m = self._m
        
        self.preallocate_variables(
            do_reweighting=False,
            is_sampling_problem=False,
            is_soc_problem=True,
        )


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
                    self._fht[i] = fht

                    self._J[i] = cost + fht
                    grad_Sh = grad_Sh * (- np.sqrt(dt * beta / 2))
                    self._gradJ[i, :] = sum_partial_tilde_gh - (cost + fht) * grad_Sh
                    
                    break
    
    @timer
    def sample_soc_vectorized(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max
        m = self._m
        
        self.preallocate_variables(
            do_reweighting=False,
            is_sampling_problem=False,
            is_soc_problem=True,
        )

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
            self._fht[new_in_target_set_idx] = fht
            self._J[new_in_target_set_idx] = dt * (fht + cost[new_in_target_set_idx])
            self._gradJ[:, new_in_target_set_idx] = sum_grad_gh[:, new_in_target_set_idx] - \
                                                    dt * (fht + cost[new_in_target_set_idx]) * \
                                                    grad_Sh[:, new_in_target_set_idx]
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break


    def save_variables(self, i, n, x):
        # TODO: deprecated method!
        dt = self._dt
        beta = self._beta
        is_drifted = self._is_drifted

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
        is_sampling_problem = self._is_sampling_problem
        is_soc_problem = self._is_soc_problem
        do_reweighting = self._do_reweighting

        # sort out trajectories which have not arrived
        self._fht = self._fht[np.where(np.isnan(self._fht) != True)]

        # first and last fht
        self._first_fht = np.min(self._fht)
        self._last_fht = np.max(self._fht)

        # compute mean and variance of fht
        self._mean_fht, \
        self._var_fht, \
        self._re_fht = self.compute_mean_variance_and_rel_error(self._fht)

        if is_sampling_problem:
            # compute mean and variance of Psi
            self._Psi = self._Psi[np.where(np.isnan(self._Psi) != True)]
            self._mean_Psi, \
            self._var_Psi, \
            self._re_Psi = self.compute_mean_variance_and_rel_error(self._Psi)
        
        if is_soc_problem:
            # compute mean of J
            #self._J = self._J[np.where(np.isnan(self._J) != True)]
            self._mean_J = np.mean(self._J)

            # compute mean of gradJ
            #self._gradJ = self._gradJ[np.where(np.isnan(self._gradJ) != True)]
            self._mean_gradJ = np.mean(self._gradJ, axis=1)

        if do_reweighting:
            # compute mean of M_fht
            self._G_fht = self._G_fht[np.where(np.isnan(self._G_fht) != True)]
            self._mean_G_fht = np.mean(self._G_fht)
            
            # compute mean of M_N
            self._G_N = self._G_N[np.where(np.isnan(self._G_N) != True)]
            self._mean_G_N= np.mean(self._G_N)
            
            # compute mean and variance of fht re-weighted
            self._fht_rew = self._fht_rew[np.where(np.isnan(self._fht_rew) != True)]
            self._mean_fht_rew, \
            self._var_fht_rew, \
            self._re_fht_rew = self.compute_mean_variance_and_rel_error(self._fht_rew)

            # compute mean and variance of Psi re_weighted
            self._Psi_rew = self._Psi_rew[np.where(np.isnan(self._Psi_rew) != True)]
            self._mean_Psi_rew, \
            self._var_Psi_rew, \
            self._re_Psi_rew = self.compute_mean_variance_and_rel_error(self._Psi_rew)

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

        f.write('drifted process: {}\n'.format(self._is_drifted))
        f.write('beta: {:2.1f}\n'.format(self._beta))
        f.write('dt: {:2.4f}\n'.format(self._dt))
        f.write('Y_0: {:2.1f}\n'.format(self._xzero))
        f.write('target set: [{:2.1f}, {:2.1f}]\n\n'
                ''.format(self._target_set_min, self._target_set_max))

        f.write('sampled trajectories: {:d}\n'.format(self._M))
        f.write('time steps: {:d}\n\n'.format(self._N))

        f.write('% trajectories which have arrived: {:2.2f}\n\n'
                ''.format(100 * len(self._fht) / self._M))
        
        f.write('E[fhs] = {:.2f}\n\n'.format(self._mean_fht / self._dt))

        f.write('first fht = {:2.4f}\n'.format(self._first_fht))
        f.write('last fht = {:2.4f}\n'.format(self._last_fht))
        f.write('E[fht] = {:2.4f}\n'.format(self._mean_fht))
        f.write('Var[fht] = {:2.4f}\n'.format(self._var_fht))
        f.write('RE[fht] = {:2.4f}\n\n'.format(self._re_fht))
       
        if self._is_sampling_problem:
            f.write('E[exp(-beta * fht)] = {:2.4e}\n'.format(self._mean_Psi))
            f.write('Var[exp(-beta * fht)] = {:2.4e}\n'.format(self._var_Psi))
            f.write('RE[exp(-beta * fht)] = {:2.4e}\n\n'.format(self._re_Psi))

        if self._is_sampling_problem and self._is_drifted and self._do_reweighting:
            f.write('E[M_fht] = {:2.4e}\n'.format(self._mean_G_fht))
            f.write('E[M_N]: {:2.4e}\n\n'.format(self._mean_G_N))
            
            f.write('E[fht * M_fht] = {:2.4f}\n'.format(self._mean_fht_rew))
            f.write('Var[fht * M_fht] = {:2.4f}\n'.format(self._var_fht_rew))
            f.write('RE[fht * M_fht] = {:2.4f}\n\n'.format(self._re_fht_rew))
            
            f.write('E[exp(-beta * fht) * M_fht] = {:2.4e}\n'
                    ''.format(self._mean_Psi_rew))
            f.write('Var[exp(-beta * fht) * M_fht] = {:2.4e}\n'
                    ''.format(self._var_Psi_rew))
            f.write('RE[exp(-beta * fht) * M_fht] = {:2.4e}\n\n'
                    ''.format(self._re_Psi_rew))
        
        if self._is_soc_problem:
            f.write('E[Jh] = {:2.4e}\n'.format(self._mean_J))
            for j in np.arange(self._m):
                f.write('E[(grad_Jh)j] = {:2.4e}\n'.format(self._mean_gradJ[j]))
    
        f.close()

    def plot_potential_and_gradient(self, file_name, dir_path=FIGURES_PATH):
        X = np.linspace(-2, 2, 100)
        V = double_well_1d_potential(X)
        dV = double_well_1d_gradient(X)

        if self._is_drifted:
            Vbias = self.bias_potential(X)
            U = self.control(X)
            dVbias = self.bias_gradient(U)
        else:
            Vbias = np.zeros(X.shape[0])
            dVbias = np.zeros(X.shape[0])

        pl = Plot(
            file_name=file_name,
            file_type='png',
            dir_path=dir_path,
        )
        pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)
