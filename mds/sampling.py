from decorators import timer
from potentials_and_gradients import double_well_1d_potential, \
                                     gradient_double_well_1d_potential, \
                                     one_well_1d_potential, \
                                     gradient_one_well_1d_potential, \
                                     derivative_normal_pdf, \
                                     bias_potential

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

    def __init__(self, beta, xzero, target_set, num_trajectories, seed=None,
                 is_drifted=True, do_reweighting=False, is_sampling_problem=True,
                 is_soc_problem=False):
        '''
        '''
        # set random seed
        if seed:
            np.random.seed(seed)

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
        self._dt = 1e-3
        self._N = 10**7
       
        # ansatz functions (gaussians) and coefficients
        self._m = None
        self._a = None
        self._mus = None
        self._sigmas = None 
        
        # variables
        self._fht = None

        self._is_sampling_problem = is_sampling_problem
        self._Psi = None
        self._F = None

        self._is_soc_problem = is_soc_problem
        self._cost = None
        self._J = None
        self._gradJ = None
        self._gradSh = None

        self._do_reweighting = do_reweighting 
        self._M_fht = None
        self._fht_rew = None
        
        self._Psi_rew = None
        self._F_rew = None

        self._J_rew = None
       
        # mean, variance and re
        self._mean_fht = None
        self._var_fht = None
        self._re_fht = None

        self._mean_Psi = None
        self._var_Psi = None
        self._re_Psi = None

        self._mean_M = None

        self._mean_J = None

    def preallocate_variables(self):
        '''
        '''
        M = self._M

        self._fht = np.empty(M)
        self._fht[:] = np.NaN
        
        if self._is_sampling_problem:
            self._Psi = np.empty(M)
            self._Psi[:] = np.NaN
            #self._F = np.empty(M)
            #self._F[:] = np.NaN

        if self._is_soc_problem:
            self._cost = np.empty(M)
            self._cost[:] = np.NaN
            self._J = np.empty(M)
            self._J[:] = np.NaN
            self._gradJ= np.empty((M, m))
            self._gradJ[:] = np.NaN
            self._gradSh = np.empty((M, m))
            self._gradSh[:] = np.NaN
        
        if self._do_reweighting: 
            self._M_fht = np.empty(M)
            self._M_fht[:] = np.NaN
            self._fht_rew = np.empty(M)
            self._fht_rew[:] = np.NaN
            
            if self._is_sampling_problem:
                self._Psi_rew = np.empty(M)
                self._Psi_rew[:] = np.NaN
            
            if self._is_soc_problem:
                self._J_rew = np.empty(M)
                self._J_rew[:] = np.NaN


    def set_a(self, a):
        self._m = a.shape[0] 
        self._a = a

    def set_v(self, mus, sigmas):
        self._mus = mus
        self._sigmas = sigmas 

    def ansatz_functions(self, x):
        '''This method computes the ansatz functions evaluated at x

        Args:
            x (float) : position
        '''
        mus = self._mus
        sigmas = self._sigmas

        v = stats.norm.pdf(x, mus, sigmas) 

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

        b = - np.sqrt(2 / beta) * derivative_normal_pdf(x, mus, sigmas) 

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

    def bias_potential(self, x):
        '''This method computes the bias potential at x

        Args:
            x (float) : position
        '''
        return self.value_function(x) * 2 / self._beta

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u (float) : control at x 
        '''
        return - np.sqrt(2 / self._beta) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x (float) : position
        '''
        return double_well_1d_potential(x) + self.bias_potential(x)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x (float) : position
            u (float) : control at x 
        '''
        return gradient_double_well_1d_potential(x) + self.bias_gradient(u)

    @timer
    def sample(self):
        M = self._M
        N = self._N
        dt = self._dt
        xzero = self._xzero
        beta = self._beta
        is_drifted = self._is_drifted
        target_set_min = self._target_set_min
        target_set_max = self._target_set_max
        do_reweighting = self._do_reweighting
        is_sampling_problem = self._is_sampling_problem
        is_soc_problem = self._is_soc_problem

        for i in np.arange(M):
            # initialize Xtemp
            Xtemp = xzero
            
            if is_drifted:
                # compute control at Xtemp
                utemp = self.control(Xtemp)

            if do_reweighting:
                # initialize martingale terms, M_t = e^(M1_t + M2_t)
                M1temp = 0
                M2temp = 0
            
            if is_soc_problem:
                    m = self._m
                    cost = 0
                    sum_partial_tilde_gh = np.zeros(m)
                    grad_Sh = np.zeros(m)
                    
                    #norm_dist = np.random.normal(0, 1, N)
            
            # Brownian increments
            #dB = np.sqrt(dt) * np.random.normal(0, 1, N)
            #dB = np.sqrt(dt) * np.random.normal(0, 1, 2 * N)
            #dB1 = dB[:N]
            #dB2 = dB[N:]
            
            for n in np.arange(1, N+1):
                # Brownian increment
                dB = np.sqrt(dt) * np.random.normal(0, 1)

                # compute gradient
                if not is_drifted:
                    gradient = gradient_double_well_1d_potential(Xtemp)
                else:
                    gradient = self.tilted_gradient(Xtemp, utemp)

                # SDE iteration
                drift = - gradient * dt
                #diffusion = np.sqrt(2 / beta) * dB[n-1]
                #diffusion = np.sqrt(2 / beta) * dB1[n-1]
                diffusion = np.sqrt(2 / beta) * dB
                Xtemp = Xtemp + drift + diffusion
                
                if is_drifted:
                    # compute control at Xtemp
                    utemp = self.control(Xtemp)

                if do_reweighting:
                    # compute martingale terms
                    # M1temp = int_0^fht (-u_t dB_t)
                    # M2temp = int_0^fht (- 1/2 (u_t)^2 dt)
                    #M1temp = M1temp - utemp * dB[n-1]
                    #M1temp = M1temp - utemp * dB2[n-1]
                    M1temp = M1temp - utemp * dB
                    M2temp = M2temp - 0.5 * (utemp ** 2) * dt 

                if is_soc_problem:
                    # evaluate the control basis functions at Xtmep
                    btemp = self.control_basis_functions(Xtemp)
                    
                    # compute cost, ...
                    cost = cost + 0.5 * (utemp ** 2) * dt
                    sum_partial_tilde_gh = sum_partial_tilde_gh + utemp * btemp * dt  
                    #grad_Sh = grad_Sh + norm_dist[n] * btemp
                    grad_Sh = grad_Sh + np.random.normal(0, 1) * btemp

                # check if we have arrived to the target set
                if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                    fht = n * dt

                    # save first hitting time
                    self._fht[i] = fht

                    if is_sampling_problem:
                        # save quantity of interest at the fht
                        self._Psi[i] = np.exp(-beta * fht)

                    if is_soc_problem:
                        self._J[i] = cost + fht
                        grad_Sh = grad_Sh * (-np.sqrt(dt * beta))
                        self._gradJ[i] = sum_partial_tilde_gh - (cost + fht) * grad_Sh

                    if do_reweighting:
                        # save Girsanov Martingale at fht
                        self._M_fht[i] = np.exp(M1temp + M2temp)
                        
                        # save re-weighted first hitting time
                        self._fht_rew[i] = fht * np.exp(M1temp + M2temp)

                        # save re-weighted quantity of interest
                        if is_sampling_problem:
                            self._Psi_rew[i] = np.exp(-beta * fht + M1temp + M2temp) 

                    break


    def save_variables(self, i, n, x):
        # TODO: deprecated method!
        dt = self._dt
        beta = self._beta
        is_drifted = self._is_drifted


    def compute_statistics(self):
        is_sampling_problem = self._is_sampling_problem
        is_soc_problem = self._is_soc_problem
        do_reweighting = self._do_reweighting

        # sort out trajectories which have not arrived
        fht = np.array([t for t in self._fht if not np.isnan(t)])
        self._fht = fht

        # compute mean and variance of fht
        mean_fht = np.mean(fht)
        var_fht = np.var(fht)
        if mean_fht != 0:
            re_fht = np.sqrt(var_fht) / mean_fht
        else:
            re_fht = np.nan
        self._mean_fht = mean_fht
        self._var_fht = var_fht
        self._re_fht = re_fht

        if is_sampling_problem:
            # compute mean and variance of I
            Psi = np.array([x for x in self._Psi if not np.isnan(x)])
            mean_Psi = np.mean(Psi)
            var_Psi = np.var(Psi)
            if mean_Psi != 0:
                re_Psi = np.sqrt(var_Psi) / mean_Psi
            else:
                re_Psi = np.nan
            self._mean_Psi = mean_Psi
            self._var_Psi = var_Psi
            self._re_Psi = re_Psi
        
        if is_soc_problem:
            # compute mean of J
            J = np.array([x for x in self._J if not np.isnan(x)])
            self._mean_J = np.mean(J)

            # compute mean of gradJ
            gradJ = np.array([x for x in self._gradJ if not np.isnan(x)])
            self._mean_gradJ = np.mean(grad_J)


        if do_reweighting:
            # compute mean of M_fht
            M_fht = np.array([x for x in self._M_fht if not np.isnan(x)])
            self._mean_M = np.mean(M_fht)
            
            # compute mean and variance of fht re-weighted
            fht_rew = np.array([t for t in self._fht_rew if not np.isnan(t)])
            mean_fht_rew = np.mean(fht_rew)
            var_fht_rew = np.var(fht_rew)
            if mean_fht_rew != 0:
                re_fht_rew = np.sqrt(var_fht_rew) / mean_fht_rew
            else:
                re_fht_rew = np.nan
            self._mean_fht_rew = mean_fht
            self._var_fht_rew = var_fht
            self._re_fht_rew = re_fht

            # compute mean and variance of Psi re_weighted
            Psi_rew = np.array([x for x in self._Psi_rew if not np.isnan(x)])
            mean_Psi_rew = np.mean(Psi_rew)
            var_Psi_rew = np.var(Psi_rew)
            if mean_Psi_rew != 0:
                re_Psi_rew = np.sqrt(var_Psi_rew) / mean_Psi_rew
            else:
                re_Psi_rew = np.nan
            self._mean_Psi_rew = mean_Psi_rew
            self._var_Psi_rew = var_Psi_rew
            self._re_Psi_rew = re_Psi_rew


    def save_statistics(self):
        is_drifted = self._is_drifted
        do_reweighting = self._do_reweighting

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
                ''.format(100 * len(self._fht) / self._M))
        
        f.write('Expectation of fhs: {:.2f}\n\n'.format(self._mean_fht / self._dt))

        f.write('Expectation of fht: {:2.4f}\n'.format(self._mean_fht))
        f.write('Variance of fht: {:2.4f}\n'.format(self._var_fht))
        f.write('Relative error of fht: {:2.4f}\n\n'.format(self._re_fht))
        
        f.write('Expectation of exp(-beta * fht): {:2.4e}\n'.format(self._mean_Psi))
        f.write('Variance of exp(-beta * fht): {:2.4e}\n'.format(self._var_Psi))
        f.write('Relative error of exp(-beta * fht): {:2.4e}\n\n'.format(self._re_Psi))
        if is_drifted and do_reweighting:
            f.write('Expectation of M_fht: {:2.4e}\n\n'.format(self._mean_M))
            
            f.write('Expectation of fht rew: {:2.4f}\n'.format(self._mean_fht_rew))
            f.write('Variance of fht rew: {:2.4f}\n'.format(self._var_fht_rew))
            f.write('Relative error of fht rew: {:2.4f}\n\n'.format(self._re_fht_rew))
            
            f.write('Expectation of exp(-beta * fht) rew: {:2.4e}\n'.format(self._mean_Psi_rew))
            f.write('Variance of exp(-beta * fht) rew: {:2.4e}\n'.format(self._var_Psi_rew))
            f.write('Relative error of exp(-beta * fht) rew: {:2.4e}\n\n'.format(self._re_Psi_rew))
    
        f.close()
