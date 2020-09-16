from ansatz_functions import gaussian_ansatz_functions, \
                             derivative_normal_pdf, \
                             bias_potential

from potentials_and_gradients import get_potential_and_gradient
from plotting import Plot
from utils import get_example_data_path, get_time_in_hms
from validation import is_valid_1d_target_set

import numpy as np
from scipy import stats
import time
import os

class langevin_1d:
    '''
    '''

    def __init__(self, potential_name, alpha, beta,
                 target_set, domain=None, h=0.01, is_drifted=False):
        '''
        '''
        # get potential and gradient functions
        potential, gradient = get_potential_and_gradient(potential_name, alpha)

        # TODO: general validators 
        # validate domain and target set
        if domain is None:
            domain = np.array([-3, 3])

        # validate target set
        if not is_valid_1d_target_set(target_set):
            #TODO raise error
            print('invalid target set')
            return

        # dir_path
        if not is_drifted:
            self.dir_path = get_example_data_path(potential_name, alpha, beta,
                                                  target_set, 'not-drifted-sampling')
        else:
            self.dir_path = get_example_data_path(potential_name, alpha, beta, target_set)

        #seed
        self.seed = None

        # sde parameters
        self.potential_name = potential_name
        self.potential = potential
        self.gradient = gradient
        self.alpha = alpha
        self.beta = beta
        self.is_drifted = is_drifted

        # sampling
        self.domain = domain
        self.xzero = None
        self.target_set = target_set
        self.M = None

        # domain discretization
        self.h = h
        self.discretize_domain()

        # Euler-Marujama
        self.dt = None
        self.N_lim = None

        # ansatz functions (gaussians) and coefficients
        self.ansatz = None
        self.theta = None
        self.theta_opt = None

        # variables

        # first hitting time
        self.fht = None
        self.first_fht = None
        self.last_fht = None
        self.mean_fht = None
        self.var_fht = None
        self.re_fht = None

        # sampling problem
        self.is_sampling_problem = None

        self.Psi = None
        self.mean_Psi = None
        self.var_Psi = None
        self.re_Psi = None

        self.G_fht = None
        self.mean_G_fht = None
        self.G_N = None
        self.mean_G_N= None

        self.Psi_rew = None
        self.mean_Psi_rew = None
        self.var_Psi_rew = None
        self.re_Psi_rew = None

        # soc problem
        self.is_soc_problem = None
        self.do_ipa = None

        self.J = None
        self.gradJ = None
        self.cost = None
        self.gradSh = None

        self.mean_J = None
        self.mean_gradJ = None
        self.mean_cost = None
        self.mean_gradSh = None

        # computational time
        self.t_initial = None
        self.t_final = None

    def discretize_domain(self, h=None):
        ''' this method discretizes the domain interval uniformly with step-size h
        Args:
            h (float): step-size 
        '''
        if h is None:
            h = self.h

        D_min, D_max = self.domain
        self.N = int((D_max - D_min) / h) + 1
        self.domain_h = np.around(np.linspace(D_min, D_max, self.N), decimals=3)

    def set_gaussian_ansatz_functions(self, m, sigma):
        '''
        '''
        # set ansatz functions
        ansatz = gaussian_ansatz_functions(
            domain=self.domain,
            m=m,
        )
        ansatz.set_unif_dist_ansatz_functions(sigma)
        #ansatz.set_unif_dist_ansatz_functions_on_S(m)
        self.ansatz = ansatz

    def set_bias_potential(self, theta, mus, sigmas):
        '''
        Args:
            theta (ndarray): parameters
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert theta.shape == mus.shape == sigmas.shape, ''

        ansatz = gaussian_ansatz_functions(domain=self.domain)
        ansatz.set_given_ansatz_functions(mus, sigmas)

        self.is_drifted = True
        self.ansatz = ansatz
        self.theta = theta

    def set_bias_potential_from_metadynamics(self):
        # load metadynamics parameters
        dir_path = get_data_path(self.potential_name, self.alpha, self.beta,
                                 self.target_set, 'metadynamics')
        bias_pot = np.load(os.path.join(dir_path, 'bias_potential.npz'))
        meta_weights = bias_pot['omegas']
        meta_mus = bias_pot['mus']
        meta_sigmas = bias_pot['sigmas']

        assert meta_weights.shape == meta_mus.shape == meta_sigmas.shape, ''

        theta = theta_weights / 2
        self.set_bias_potential(theta, meta_mus, meta_sigmas)


    def set_theta_from_metadynamics(self):
        '''
        '''
        # load metadynamics parameters
        dir_path = get_data_path(self.potential_name, self.alpha, self.beta,
                                 self.target_set, 'metadynamics')
        bias_pot = np.load(os.path.join(dir_path, 'bias_potential.npz'))
        meta_weights = bias_pot['omegas']
        meta_mus = bias_pot['mus']
        meta_sigmas = bias_pot['sigmas']
        assert meta_weights.shape == meta_mus.shape == meta_sigmas.shape, ''

        # get ansatz functions
        assert self.ansatz is not None, ''
        m = self.ansatz.m
        mus = self.ansatz.mus
        sigmas= self.ansatz.sigmas

        # discretized interval domain
        X = self.domain_h

        # ansatz functions evaluated at the grid
        v = self.ansatz.basis_value_f(X)

        # value function evaluated at the grid
        V_bias = bias_potential(X, meta_weights, meta_mus, meta_sigmas)
        value_f = V_bias / 2

        # solve theta V = \Phi
        theta, _, _, _ = np.linalg.lstsq(v.T, value_f, rcond=None)

        self.is_drifted = True
        self.theta = theta

    def set_a_from_gd(self, ipa_type, sigma, lr):
        '''
        '''
        # load gd parameters
        dir_path = get_data_path(self.potential_name, self.alpha, self.beta,
                                 self.target_set, 'metadynamics')
        bias_pot = np.load(os.path.join(dir_path, 'bias_potential.npz'))
        omegas = bias_pot['omegas']
        meta_mus = bias_pot['mus']
        meta_sigmas = bias_pot['sigmas']

        assert omegas.shape == meta_mus.shape == meta_sigmas.shape

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

    def set_theta_optimal(self):
        assert self.ansatz is not None, ''
        dir_path = get_data_path(self.potential_name, self.alpha, self.beta,
                                 self.target_set, 'reference_solution')
        ref_sol = np.load(os.path.join(dir_path, 'reference_solution.npz'))

        # compute the optimal a given a basis of ansatz functions
        X = ref_sol['domain_h']
        a = self.ansatz.basis_value_f(X).T
        b = ref_sol['F']
        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

        self.theta_opt = x

    def value_function(self, x, theta=None):
        '''This method computes the value function evaluated at x

        Args:
            x (float or ndarray) : position/s
            theta (ndarray): parameters

        Return:
        '''
        ansatz = self.ansatz
        if theta is not None:
            mus = ansatz.mus
            sigmas = ansatz.sigmas
            assert theta.shape == mus.shape == sigmas.shape, ''
        else:
            theta = self.theta

        return np.dot(theta, ansatz.basis_value_f(x))

    def control(self, x, theta=None):
        '''This method computes the control evaluated at x

        Args:
            x (float or ndarray) : position/s
            theta (ndarray): parameters
        '''
        ansatz = self.ansatz
        if theta is not None:
            mus = ansatz.mus
            sigmas = ansatz.sigmas
            assert theta.shape == mus.shape == sigmas.shape, ''
        else:
            theta = self.theta

        return np.dot(theta, ansatz.basis_control(x))

    def bias_potential(self, x, theta=None):
        '''This method computes the bias potential at x

        Args:
            x (float or ndarray) : position/s
            theta (ndarray): parameters
        '''
        return 2 * self.value_function(x, theta)

    def bias_gradient(self, u):
        '''This method computes the bias gradient at x

        Args:
            u (float or ndarray) : control at x
        '''
        return - np.sqrt(2) * u

    def tilted_potential(self, x):
        '''This method computes the tilted potential at x

        Args:
            x (float or ndarray) : position/s
            theta (ndarray): parameters
        '''
        return self.potential(x) + self.bias_potential(x, theta)

    def tilted_gradient(self, x, u):
        '''This method computes the tilted gradient at x

        Args:
            x (float or ndarray) : position
            u (float or ndarray) : control at x
        '''
        assert type(x) == type(u)
        if type(x) == np.ndarray:
            assert x.shape == u.shape

        return self.gradient(x) + self.bias_gradient(u)

    def set_sampling_parameters(self, xzero, M, dt, N_lim, seed=None):
        '''
        '''
        # set random seed
        if seed:
            np.random.seed(seed)

        # sampling
        self.xzero = xzero
        self.M = M

        # Euler-Marujama
        self.dt = dt
        self.N_lim = N_lim

        # initialize timer
        self.t_initial = time.time()

    def initialize_sampling_variables(self, is_sampling_problem=False,
                                      is_soc_problem=False):
        '''
        '''
        assert self.M is not None, ''
        M = self.M

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
            assert self.ansatz is not None, ''
            m = self.ansatz.m
            self.J = np.empty(M)
            self.J[:] = np.NaN
            self.cost = np.empty(M)
            self.cost[:] = np.NaN
            if self.do_ipa:
                self.gradJ= np.empty((m, M))
                self.gradJ[:] = np.NaN
                self.gradSh= np.empty((m, M))
                self.gradSh[:] = np.NaN

        self.is_sampling_problem = is_sampling_problem
        self.is_soc_problem = is_soc_problem

    def sample_not_drifted(self):
        M = self.M
        N_lim = self.N_lim
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min, target_set_max = self.target_set

        self.initialize_sampling_variables(is_sampling_problem=True)

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)

        # has arrived in target set
        been_in_target_set = np.repeat([False], M)

        for n in np.arange(1, N_lim +1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1, M)

            # compute gradient
            gradient = self.gradient(Xtemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion

            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))

            # indices of trajectories new in the target set
            new_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]

            # update trajectories which have been in the target set
            been_in_target_set[new_idx] = True

            # save first hitting time
            self.fht[new_idx] = n * dt

            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break

        # save quantity of interest at the fht
        self.Psi = np.exp(-beta * self.fht)


    def sample_drifted(self):
        M = self.M
        N_lim = self.N_lim
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min, target_set_max = self.target_set

        self.initialize_sampling_variables(is_sampling_problem=True)

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)

        # initialize Girsanov Martingale terms, G_t = e^(G1_t + G2_t)
        G1temp = np.zeros(M)
        G2temp = np.zeros(M)

        # has arrived in target set
        been_in_target_set = np.repeat([False], M)

        for n in np.arange(1, N_lim +1):
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
            G1temp = G1temp - np.sqrt(beta) * utemp * dB
            G2temp = G2temp - beta * 0.5 * (utemp ** 2) * dt

            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))

            # indices of trajectories new in the target set
            new_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]

            # update list of indices whose trajectories have been in the target set
            been_in_target_set[new_idx] = True

            # save first hitting time
            self.fht[new_idx] = n * dt

            # save Girsanov Martingale
            self.G_fht[new_idx] = np.exp(G1temp[new_idx] + G2temp[new_idx])

            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                # save Girsanov Martingale at the time when 
                # the last trajectory arrive
                self.G_N = np.exp(G1temp + G2temp)
                break

        # save reweighted quantity of interest
        self.Psi_rew = np.exp(-beta * self.fht) * self.G_fht

    def sample_soc(self, do_ipa=False):
        M = self.M
        N_lim = self.N_lim
        dt = self.dt
        xzero = self.xzero
        beta = self.beta
        target_set_min, target_set_max = self.target_set
        m = self.m
        self.do_ipa = do_ipa

        self.initialize_sampling_variables(is_soc_problem=True)

        # initialize Xtemp
        Xtemp = xzero * np.ones(M)

        # initialize cost, sum of grad of g and grad of S
        cost = np.zeros(M)
        if do_ipa:
            sum_grad_gh = np.zeros((m, M))
            gradSh = np.zeros((m, M))

        # has arrived in target set
        been_in_target_set = np.repeat([False], M)

        for n in np.arange(1, N_lim +1):
            normal_dist_samples = np.random.normal(0, 1, M)

            # Brownian increment
            dB = np.sqrt(dt) * normal_dist_samples

            # compute control at Xtemp
            utemp = self.control(Xtemp)

            # evaluate the control basis functions at Xtmep
            btemp = self.control_basis_functions(Xtemp)

            # compute gradient
            tilted_gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - tilted_gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp += drift + diffusion

            # compute cost, ...
            cost += 0.5 * (utemp ** 2) * dt
            if do_ipa:
                sum_grad_gh += dt * utemp * btemp
                gradSh += normal_dist_samples * btemp

            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))

            # indices of trajectories new in the target set
            new_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]
            if len(new_idx) == 0:
                continue

            # update list of indices whose trajectories have been in the target set
            been_in_target_set[new_idx] = True

            # save first hitting time
            fht = n * dt
            self.fht[new_idx] = fht
            self.cost[new_idx] = cost[new_idx]
            self.J[new_idx] = fht + cost[new_idx]
            if do_ipa:
                #gradSh[:, new_idx] *= beta * np.sqrt(dt / 2)
                gradSh[:, new_idx] *= - np.sqrt(dt * beta)
                self.gradSh[:, new_idx] = gradSh[:, new_idx]
                self.gradJ[:, new_idx] = sum_grad_gh[:, new_idx] \
                                       - (fht + cost[new_idx]) * gradSh[:, new_idx]

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

            if self.do_ipa:
                # compute mean of gradJ
                self.mean_cost = np.mean(self.cost)
                self.mean_gradJ = np.mean(self.gradJ, axis=1)
                self.mean_gradSh = np.mean(self.gradSh, axis=1)

        # stop timer
        self.t_final = time.time()


    def save_statistics(self):
        '''
        '''
        # set path
        if not self.is_drifted:
            sampling_stamp = 'report_not_drifted'
        else:
            sampling_stamp = 'report_drifted'
        trajectories_stamp = 'M{:.0e}'.format(self.M)
        file_name = sampling_stamp + '_' + trajectories_stamp + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        f.write('SDE parameters\n')
        f.write('potential: {}\n'.format(self.potential_name))
        f.write('alpha: {}\n'.format(self.alpha))
        f.write('beta: {:2.1f}\n'.format(self.beta))
        f.write('drifted process: {}\n\n'.format(self.is_drifted))

        f.write('Euler-Maruyama discretization parameters\n')
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('maximal time steps: {:,d}\n\n'.format(self.N_lim))

        f.write('Sampling parameters and statistics\n')
        f.write('xzero: {:2.1f}\n'.format(self.xzero))
        f.write('target set: [{:2.1f}, {:2.1f}]\n'
                ''.format(self.target_set[0], self.target_set[1]))
        f.write('sampled trajectories: {:,d}\n'.format(self.M))
        f.write('trajectories which arrived: {:2.2f} %\n'
                ''.format(100 * len(self.fht) / self.M))
        f.write('time steps last trajectory: {:,d}\n\n'.format(int(self.last_fht / self.dt)))

        if self.is_drifted:
            self.ansatz.write_ansatz_report(f)

        f.write('First hitting time statistics\n')
        f.write('first fht = {:2.3f}\n'.format(self.first_fht))
        f.write('last fht = {:2.3f}\n'.format(self.last_fht))
        f.write('E[fht] = {:2.3f}\n'.format(self.mean_fht))
        f.write('Var[fht] = {:2.3f}\n'.format(self.var_fht))
        f.write('RE[fht] = {:2.3f}\n\n'.format(self.re_fht))

        if self.is_sampling_problem and not self.is_drifted:
            f.write('Moment generation function statistics\n')
            f.write('E[exp(-beta * fht)] = {:2.2e}\n'.format(self.mean_Psi))
            f.write('Var[exp(-beta * fht)] = {:2.2e}\n'.format(self.var_Psi))
            f.write('RE[exp(-beta * fht)] = {:2.2e}\n\n'.format(self.re_Psi))

        elif self.is_sampling_problem and self.is_drifted:
            f.write('Girsanov Martingale\n')
            f.write('E[M_fht] = {:2.2e}\n'.format(self.mean_G_fht))
            f.write('E[M_N]: {:2.2e}\n\n'.format(self.mean_G_N))

            f.write('Reweighted Moment generation function statistics\n')
            f.write('E[exp(-beta * fht) * M_fht] = {:2.2e}\n'
                    ''.format(self.mean_Psi_rew))
            f.write('Var[exp(-beta * fht) * M_fht] = {:2.2e}\n'
                    ''.format(self.var_Psi_rew))
            f.write('RE[exp(-beta * fht) * M_fht] = {:2.2e}\n\n'
                    ''.format(self.re_Psi_rew))

        if self.is_soc_problem:
            f.write('Gradient descent\n')
            f.write('E[Jh] = {:2.2e}\n'.format(self.mean_J))
            if self.do_ipa:
                for j in np.arange(self.m):
                    f.write('E[(grad_Jh)j] = {:2.2e}\n\n'.format(self.mean_gradJ[j]))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        f.close()

    def plot_tilted_potential(self, file_name):
        D_min, D_max = self.domain
        X = np.linspace(D_min, D_max, 1000)
        V = self.potential(X)
        if self.is_drifted:
            Vbias = self.bias_potential(X)
        else:
            Vbias = np.zeros(X.shape[0])

        pl = Plot(dir_path=self.dir_path, file_name=file_name)
        pl.set_ylim(bottom=0, top=self.alpha * 10)
        if self.theta_opt is not None:
            Vopt = V + self.bias_potential(X, self.theta_opt)
            pl.potential_and_tilted_potential(X, V, Vbias, Vopt)
        else:
            pl.potential_and_tilted_potential(X, V, Vbias)


    def plot_tilted_drift(self, file_name):
        D_min, D_max = self.domain
        X = np.linspace(D_min, D_max, 1000)
        dV = self.gradient(X)
        if self.is_drifted:
            U = self.control(X)
            dVbias = self.bias_gradient(U)
        else:
            dVbias = np.zeros(X.shape[0])

        pl = Plot(dir_path=self.dir_path, file_name=file_name)
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        if self.theta_opt is not None:
            U = self.control(X, self.theta_opt)
            dVopt = dV - np.sqrt(2) * U
            pl.drift_and_tilted_drift(X, dV, dVbias, dVopt)
        else:
            pl.drift_and_tilted_drift(X, dV, dVbias)

    #TODO: revise it
    def plot_optimal_potential_and_gradient(self):
        D_min, D_max = self.domain
        X = np.linspace(D_min, D_max, 1000)
        V = self.potential(X)
        dV = self.gradient(X)
        Vbias = self.bias_potential(X, self.a_opt)
        U = self.control(X, self.a_opt)
        dVbias = - np.sqrt(2) * U
        pl = Plot(file_name='potential_and_gradient_optimal')
        pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)

    def plot_ansatz_functions(self):
        D_min, D_max = self.domain
        X = np.linspace(D_min, D_max, 1000)
        v = self.ansatz.basis_value_f(X)
        pl = Plot(dir_path=self.dir_path, file_name='gaussian_ansatz_functions')
        pl.ansatz_functions(X, v)

    def plot_control_basis_functions(self):
        D_min, D_max = self.domain
        X = np.linspace(D_min, D_max, 1000)
        b = self.ansatz.basis_control(X)
        pl = Plot(dir_path=self.dir_path, file_name='control_basis_functions')
        pl.control_basis_functions(X, b)

