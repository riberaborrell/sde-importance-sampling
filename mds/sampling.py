from decorators import timer
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient, \
                                     one_well_1d_potential, \
                                     one_well_1d_gradient, \
                                     derivative_normal_pdf, \
                                     bias_potential
from plotting import Plot

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
        self._mean_fht = None
        self._var_fht = None
        self._re_fht = None

        self._mean_Psi = None
        self._var_Psi = None
        self._re_Psi = None

        self._mean_G_fht = None
        self._mean_G_N= None

        self._mean_J = None
   
    def set_bias_potential(self, a, mus, sigmas):
        '''
        '''
        self._m = a.shape[0] 
        self._a = a
        self._mus = mus
        self._sigmas = sigmas 
    
    def set_bias_potential_from_metadynamics_original(self):
        beta = self._beta

        # load metadynamics parameters
        bias_pot_coeff = np.load(
            os.path.join(DATA_PATH, 'langevin1d_bias_potential_fake_metadynamics.npz')
            #os.path.join(DATA_PATH, 'langevin1d_bias_potential_metadynamics.npz')
        )
        omegas = bias_pot_coeff['omegas']
        meta_mus = bias_pot_coeff['mus']
        meta_sigmas = bias_pot_coeff['sigmas']
    
        a = omegas * beta / 2
        
        self._m = a.shape[0]
        self._a = a
        self._mus = meta_mus
        self._sigmas = meta_sigmas


    def set_bias_potential_from_metadynamics(self, m, J_min, J_max):
        '''
        '''
        beta = self._beta
        # validate input 
        if J_min >= J_max:
            #TODO raise error
            print("The interval J_h is not valid")
       
        # load metadynamics parameters
        bias_pot_coeff = np.load(
            os.path.join(DATA_PATH, 'langevin1d_bias_potential_fake_metadynamics.npz')
            #os.path.join(DATA_PATH, 'langevin1d_bias_potential_metadynamics.npz')
        )
        omegas = bias_pot_coeff['omegas']
        meta_mus = bias_pot_coeff['mus']
        meta_sigmas = bias_pot_coeff['sigmas']

        # define a coefficients 
        a = np.zeros(m)
        
        # grid on the interval J_h = [J_min, J_max].
        X = np.linspace(J_min, J_max, m)
        # step size
        #h = (J_max - J_min) / m

        # the ansatz functions are gaussians with standard deviation sigma
        # and means uniformly spaced in J_h
        mus = X
        sigmas = 0.3 * np.ones(m)
        
        # ansatz functions evaluated at the grid
        ansatz_functions = self.ansatz_functions(X, mus, sigmas)

        # value function evaluated at the grid
        V_bias = bias_potential(X, omegas, meta_mus, meta_sigmas)
        phi = V_bias * beta / 2

        # solve a V = \Phi
        a = np.linalg.solve(ansatz_functions, phi)

        self._m = m
        self._a = a
        self._mus = mus
        self._sigmas = sigmas


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

        return - np.sqrt(2 / beta) * derivative_normal_pdf(x, mus, sigmas)

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
        return self.value_function(x) * 2 / self._beta

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u (float or ndarray) : control at x
        '''
        return - np.sqrt(2 / self._beta) * u

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
        #TODO assert if x ndarray also u ndarray
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
            self._cost = np.empty(M)
            self._cost[:] = np.NaN
            self._J = np.empty(M)
            self._J[:] = np.NaN
            self._gradJ= np.empty((M, m))
            self._gradJ[:] = np.NaN
            self._gradSh = np.empty((M, m))
            self._gradSh[:] = np.NaN
        
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
            
            # compute control at Xtemp
            utemp = self.control(Xtemp)

            # initialize Girsanov Martingale terms, G_t = e^(G1_t + G2_t)
            G1temp = 0
            G2temp = 0
            
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

                # compute Girsanov Martingale terms
                # G1_t = int_0^fht -u_t dB_t
                # G2_t = int_0^fht - 1/2 (u_t)^2 dt
                G1temp = G1temp - utemp * dB
                G2temp = G2temp - 0.5 * (utemp ** 2) * dt 
                
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
        k = 10

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)
        
        # control at Xtemp
        utemp = self.control(Xtemp)
        
        # initialize Girsanov Martingale terms, G_t = e^(G1_t + G2_t)
        G1temp = np.zeros(M)
        G2temp = np.zeros(M)

        # has arrived in target set
        been_in_target_set = np.repeat([False], M)
        
        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1, M)

            # compute gradient
            gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion

            # control at Xtemp
            utemp = self.control(Xtemp)

            # Girsanov Martingale terms
            G1temp = G1temp - utemp * dB
            G2temp = G2temp - 0.5 * (utemp ** 2) * dt 
            
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
            
            # save Girsanov Martingale
            self._G_fht[new_in_target_set_idx] = np.exp(
                G1temp[new_in_target_set_idx] + G2temp[new_in_target_set_idx]
            )
            # save Girsanov Martingale at time k
            if n == k: 
                self._G_N = np.exp(G1temp + G2temp)
            
            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
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
        fht = np.array([t for t in self._fht if not np.isnan(t)])
        self._fht = fht
        # compute mean and variance of fht
        self._mean_fht, \
        self._var_fht, \
        self._re_fht = self.compute_mean_variance_and_rel_error(fht)

        if is_sampling_problem:
            # compute mean and variance of Psi
            Psi = np.array([x for x in self._Psi if not np.isnan(x)])
            self._mean_Psi, \
            self._var_Psi, \
            self._re_Psi = self.compute_mean_variance_and_rel_error(Psi)
        
        if is_soc_problem:
            # compute mean of J
            J = np.array([x for x in self._J if not np.isnan(x)])
            self._mean_J = np.mean(J)

            # compute mean of gradJ
            gradJ = np.array([x for x in self._gradJ if not np.isnan(x).any()])
            self._mean_gradJ = np.mean(gradJ, axis=0)

        if do_reweighting:
            # compute mean of M_fht
            G_fht = np.array([x for x in self._G_fht if not np.isnan(x)])
            self._mean_G_fht = np.mean(G_fht)
            
            # compute mean of M_N
            G_N = np.array([x for x in self._G_N if not np.isnan(x)])
            self._mean_G_N= np.mean(G_N)
            
            # compute mean and variance of fht re-weighted
            fht_rew = np.array([t for t in self._fht_rew if not np.isnan(t)])
            self._mean_fht_rew, \
            self._var_fht_rew, \
            self._re_fht_rew = self.compute_mean_variance_and_rel_error(fht_rew)

            # compute mean and variance of Psi re_weighted
            Psi_rew = np.array([x for x in self._Psi_rew if not np.isnan(x)])
            self._mean_Psi_rew, \
            self._var_Psi_rew, \
            self._re_Psi_rew = self.compute_mean_variance_and_rel_error(Psi_rew)

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

        f.write('E[fht] = {:2.4f}\n'.format(self._mean_fht))
        f.write('Var[fht] = {:2.4f}\n'.format(self._var_fht))
        f.write('RE[fht] = {:2.4f}\n\n'.format(self._re_fht))
        
        f.write('E[exp(-beta * fht)] = {:2.4e}\n'.format(self._mean_Psi))
        f.write('Var[exp(-beta * fht)] = {:2.4e}\n'.format(self._var_Psi))
        f.write('RE[exp(-beta * fht)] = {:2.4e}\n\n'.format(self._re_Psi))

        if self._is_drifted and self._do_reweighting:
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
    
        f.close()

    def plot_potential_and_gradient(self, file_name):
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
            dir_path=FIGURES_PATH,
        )
        pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)

