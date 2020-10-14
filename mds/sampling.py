from ansatz_functions import gaussian_ansatz_functions
from potentials_and_gradients import get_potential_and_gradient
from plotting import Plot
from utils import get_example_data_path, get_gd_data_path, get_time_in_hms, make_dir_path
from validation import is_1d_valid_domain, is_1d_valid_target_set, is_1d_valid_control

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

        # validate domain and target set
        if domain is None:
            domain = np.array([-3, 3])
        is_1d_valid_domain(domain)
        is_1d_valid_target_set(domain, target_set)

        # dir_path
        self.example_dir_path = get_example_data_path(potential_name, alpha,
                                                      beta, target_set)
        self.dir_path = None
        self.gd_dir_path = None

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
        self.theta_type = None
        self.theta = None

        # variables

        # first hitting time
        self.fht = None
        self.first_fht = None
        self.last_fht = None
        self.mean_fht = None
        self.var_fht = None
        self.re_fht = None

        self.Psi = None
        self.mean_Psi = None
        self.var_Psi = None
        self.re_Psi = None

        # reweighting
        self.G_fht = None
        self.mean_G_fht = None
        self.G_N = None
        self.mean_G_N= None

        self.Psi_rew = None
        self.mean_Psi_rew = None
        self.var_Psi_rew = None
        self.re_Psi_rew = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # reference solution
        self.ref_sol = None

        # metadynamics
        self.meta_bias_pot = None


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

    def set_not_drifted_dir_path(self):
        self.dir_path = os.path.join(self.example_dir_path, 'not-drifted-sampling')
        make_dir_path(self.dir_path)

    def set_drifted_dir_path(self):
        assert self.ansatz is not None, ''
        assert self.ansatz.dir_path is not None, ''

        self.dir_path = os.path.join(self.ansatz.dir_path, 'drifted-sampling')
        make_dir_path(self.dir_path)

    def set_gd_dir_path(self, gd_type, theta_init, lr):
        ansatz_dir_path = self.ansatz.dir_path
        self.gd_dir_path = get_gd_data_path(ansatz_dir_path, gd_type, theta_init, lr)
        return self.gd_dir_path

    def set_gaussian_ansatz_functions(self, m, sigma):
        '''
        '''
        assert self.is_drifted, ''

        # set gaussian ansatz functions
        ansatz = gaussian_ansatz_functions(
            domain=self.domain,
            m=m,
        )
        ansatz.set_unif_dist_ansatz_functions(sigma)
        #ansatz.set_unif_dist_ansatz_functions_on_S(m)

        # set ansatz dir path
        ansatz.set_dir_path(self.example_dir_path)
        self.ansatz = ansatz

    def set_bias_potential(self, theta, mus, sigmas):
        ''' set the gaussian ansatz functions and the coefficients theta
        Args:
            theta (ndarray): parameters
            mus (ndarray): mean of each gaussian
            sigmas (ndarray) : standard deviation of each gaussian
        '''
        assert self.is_drifted, ''
        assert theta.shape == mus.shape == sigmas.shape, ''

        # set gaussian ansatz functions
        ansatz = gaussian_ansatz_functions(domain=self.domain)
        ansatz.set_given_ansatz_functions(mus, sigmas)

        self.ansatz = ansatz
        self.theta = theta

    def load_meta_bias_potential(self):
        if not self.meta_bias_pot:
            file_path = os.path.join(
                self.example_dir_path,
                'metadynamics',
                'bias_potential.npz',
            )
            self.meta_bias_pot = np.load(file_path)

    def set_bias_potential_from_metadynamics(self):
        ''' set the gaussian ansatz functions and the coefficients from metadynamics
        '''
        self.load_meta_bias_potential()
        meta_theta = self.meta_bias_pot['omegas'] / 2
        meta_mus = self.meta_bias_pot['mus']
        meta_sigmas = self.meta_bias_pot['sigmas']

        assert meta_theta.shape == meta_mus.shape == meta_sigmas.shape, ''

        self.set_bias_potential(meta_theta, meta_mus, meta_sigmas)

    def load_reference_solution(self):
        if not self.ref_sol:
            file_path = os.path.join(
                self.example_dir_path,
                'reference_solution',
                'reference_solution.npz',
            )
            self.ref_sol = np.load(file_path)

    def set_theta_optimal(self):
        assert self.ansatz is not None, ''

        self.load_reference_solution()
        ref_sol = self.ref_sol

        x = ref_sol['domain_h']
        F = ref_sol['F']

        # compute the optimal theta given a basis of ansatz functions
        v = self.ansatz.basis_value_f(x)
        self.theta, _, _, _ = np.linalg.lstsq(v, F, rcond=None)
        self.theta_type = 'optimal'

    def set_theta_null(self):
        assert self.ansatz is not None, ''
        m = self.ansatz.m
        self.theta = np.zeros(m)
        self.theta_type = 'null'

    def set_theta_from_metadynamics(self):
        '''
        '''
        x = self.domain_h

        self.load_meta_bias_potential()
        meta_theta = self.meta_bias_pot['omegas'] / 2
        meta_mus = self.meta_bias_pot['mus']
        meta_sigmas = self.meta_bias_pot['sigmas']
        assert meta_theta.shape == meta_mus.shape == meta_sigmas.shape, ''

        # create ansatz functions from meta
        meta_ansatz = gaussian_ansatz_functions(domain=self.domain)
        meta_ansatz.set_given_ansatz_functions(meta_mus, meta_sigmas)

        # meta value function evaluated at the grid
        value_f_meta = self.value_function(x, meta_theta, meta_ansatz)

        # ansatz functions evaluated at the grid
        v = self.ansatz.basis_value_f(x)

        # solve theta V = \Phi
        self.theta, _, _, _ = np.linalg.lstsq(v, value_f_meta, rcond=None)
        self.theta_type = 'meta'

    def set_theta_from_gd(self, gd_type, gd_theta_init, gd_lr):
        '''
        '''
        # load gd parameters
        gd_dir_path = self.set_gd_dir_path(gd_type, gd_theta_init, gd_lr)
        gd = np.load(os.path.join(gd_dir_path, 'gd.npz'))
        x = gd['domain_h']
        idx_last_epoch = gd['epochs'][-1]
        u = gd['u'][idx_last_epoch]

        # ansatz functions evaluated at the grid
        b = self.ansatz.basis_control(x)

        # solve a V = \Phi
        self.theta, _, _, _ = np.linalg.lstsq(b, u, rcond=None)
        self.theta_type = 'gd'

        # set drifted sampling dir path
        self.dir_path = os.path.join(gd_dir_path, 'drifted-sampling')
        make_dir_path(self.dir_path)

    def value_function(self, x, theta=None, ansatz=None):
        '''This method computes the value function evaluated at x

        Args:
            x (float or ndarray) : position/s
            theta (ndarray): parameters
            ansatz (object): ansatz functions

        Return:
        '''
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz
        assert theta.shape == ansatz.mus.shape == ansatz.sigmas.shape, ''

        return np.dot(ansatz.basis_value_f(x), theta)

    def control(self, x, theta=None, ansatz=None):
        '''This method computes the control evaluated at x

        Args:
            x (float or ndarray) : position/s
            theta (ndarray): parameters
            ansatz (object): ansatz functions
        '''
        if theta is None:
            theta = self.theta
        if ansatz is None:
            ansatz = self.ansatz
        assert theta.shape == ansatz.mus.shape == ansatz.sigmas.shape, ''

        return np.dot(ansatz.basis_control(x), theta)

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

    def initialize_sampling_variables(self):
        '''
        '''
        assert self.M is not None, ''
        M = self.M

        self.fht = np.empty(M)
        self.fht[:] = np.NaN

        if not self.is_drifted:
            self.Psi = np.empty(M)
            self.Psi[:] = np.NaN

        else:
            self.G_fht = np.empty(M)
            self.G_fht[:] = np.NaN
            self.G_N = np.empty(M)
            self.G_N[:] = np.NaN

            self.Psi_rew = np.empty(M)
            self.Psi_rew[:] = np.NaN

    def sample_not_drifted(self):
        beta = self.beta
        dt = self.dt
        N_lim = self.N_lim
        xzero = self.xzero
        M = self.M
        target_set_min, target_set_max = self.target_set

        self.initialize_sampling_variables()

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
            Xtemp += drift + diffusion

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
        beta = self.beta
        dt = self.dt
        N_lim = self.N_lim
        xzero = self.xzero
        M = self.M
        target_set_min, target_set_max = self.target_set

        self.initialize_sampling_variables()

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
            Xtemp += drift + diffusion

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

    def sample_loss(self):
        alpha = self.alpha
        beta = self.beta
        dt = self.dt
        N_lim = self.N_lim
        xzero = self.xzero
        M = self.M
        target_set_min, target_set_max = self.target_set
        m = self.ansatz.m

        # initialize statistics 
        J = np.zeros(M)
        grad_J = np.zeros((M, m))

        # initialize temp variables
        Xtemp = xzero * np.ones(M)
        cost_temp = np.zeros(M)
        grad_phi_temp = np.zeros((M, m))
        grad_S_temp = np.zeros((M, m))

        # has arrived in target set
        been_in_target_set = np.repeat([False], M)

        for n in np.arange(1, N_lim+1):
            normal_dist_samples = np.random.normal(0, 1, M)

            # Brownian increment
            dB = np.sqrt(dt) * normal_dist_samples

            # control
            btemp = self.ansatz.basis_control(Xtemp)
            utemp = self.control(Xtemp)
            if not is_1d_valid_control(utemp, -self.alpha * 10, self.alpha * 10):
                return False, None, None

            # ipa statistics 
            cost_temp += 0.5 * (utemp ** 2) * dt
            grad_phi_temp += (utemp * btemp.T * dt).T
            grad_S_temp -= (np.sqrt(beta) * btemp.T * dB).T

            # compute gradient
            tilted_gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - tilted_gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp += drift + diffusion

            # trajectories in the target set
            is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))

            # indices of trajectories new in the target set
            new_idx = np.where(
                (is_in_target_set == True) & (been_in_target_set == False)
            )[0]

            # update list of indices whose trajectories have been in the target set
            been_in_target_set[new_idx] = True

            # save ipa statistics
            J[new_idx] = n * dt + cost_temp[new_idx]
            grad_J[new_idx, :] = grad_phi_temp[new_idx, :] \
                               - ((n * dt + cost_temp[new_idx]) \
                               * grad_S_temp[new_idx, :].T).T

            # check if all trajectories have arrived to the target set
            if been_in_target_set.all() == True:
                break

        # compute averages
        mean_J = np.mean(J)
        mean_grad_J = np.mean(grad_J, axis=0)

        return True, mean_J, mean_grad_J

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

        if not self.is_drifted:
            # compute mean and variance of Psi
            self.Psi = self.Psi[np.where(np.isnan(self.Psi) != True)]
            self.mean_Psi, \
            self.var_Psi, \
            self.re_Psi = self.compute_mean_variance_and_rel_error(self.Psi)

        else:
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

        # stop timer
        self.t_final = time.time()

    def write_sde_parameters(self, f):
        '''
        '''
        f.write('SDE parameters\n')
        f.write('potential: {}\n'.format(self.potential_name))
        f.write('alpha: {}\n'.format(self.alpha))
        f.write('beta: {:2.1f}\n'.format(self.beta))
        f.write('drifted process: {}\n\n'.format(self.is_drifted))

    def write_euler_maruyama_parameters(self, f):
        f.write('Euler-Maruyama discretization parameters\n')
        f.write('dt: {:2.4f}\n'.format(self.dt))
        f.write('maximal time steps: {:,d}\n\n'.format(self.N_lim))

    def write_sampling_parameters(self, f):
        f.write('Sampling parameters\n')
        f.write('xzero: {:2.1f}\n'.format(self.xzero))
        f.write('target set: [{:2.1f}, {:2.1f}]\n'
                ''.format(self.target_set[0], self.target_set[1]))
        f.write('sampled trajectories: {:,d}\n\n'.format(self.M))

    def write_report(self):
        '''
        '''
        # set file path
        if self.is_drifted:
            theta_stamp = 'theta-{}_'.format(self.theta_type)
        else:
            theta_stamp = ''

        trajectories_stamp = 'M{:.0e}'.format(self.M)
        file_name = 'report_' + theta_stamp + trajectories_stamp + '.txt'
        file_path = os.path.join(self.dir_path, file_name)

        # write in file
        f = open(file_path, "w")

        self.write_sde_parameters(f)
        self.write_euler_maruyama_parameters(f)
        self.write_sampling_parameters(f)

        if self.is_drifted:
            self.ansatz.write_ansatz_parameters(f)

        f.write('Statistics\n\n')

        f.write('trajectories which arrived: {:2.2f} %\n'
                ''.format(100 * len(self.fht) / self.M))
        f.write('time steps last trajectory: {:,d}\n\n'.format(int(self.last_fht / self.dt)))

        f.write('First hitting time\n')
        f.write('first fht = {:2.3f}\n'.format(self.first_fht))
        f.write('last fht = {:2.3f}\n'.format(self.last_fht))
        f.write('E[fht] = {:2.3f}\n'.format(self.mean_fht))
        f.write('Var[fht] = {:2.3f}\n'.format(self.var_fht))
        f.write('RE[fht] = {:2.3f}\n\n'.format(self.re_fht))

        if not self.is_drifted:
            f.write('Moment generation function\n')
            f.write('E[exp(-beta * fht)] = {:2.3e}\n'.format(self.mean_Psi))
            f.write('Var[exp(-beta * fht)] = {:2.3e}\n'.format(self.var_Psi))
            f.write('RE[exp(-beta * fht)] = {:2.3e}\n\n'.format(self.re_Psi))

        else:
            f.write('Girsanov Martingale\n')
            f.write('E[M_fht] = {:2.3e}\n'.format(self.mean_G_fht))
            f.write('E[M_N]: {:2.3e}\n\n'.format(self.mean_G_N))

            f.write('Reweighted Moment generation function\n')
            f.write('E[exp(-beta * fht) * M_fht] = {:2.3e}\n'
                    ''.format(self.mean_Psi_rew))
            f.write('Var[exp(-beta * fht) * M_fht] = {:2.3e}\n'
                    ''.format(self.var_Psi_rew))
            f.write('RE[exp(-beta * fht) * M_fht] = {:2.3e}\n\n'
                    ''.format(self.re_Psi_rew))

        h, m, s = get_time_in_hms(self.t_final - self.t_initial)
        f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        f.close()

    def plot_appr_mgf(self, file_name, dir_path=None):
        beta = self.beta
        x = self.domain_h

        Vbias = self.bias_potential(x)
        appr_F = Vbias / 2
        appr_Psi = np.exp(- beta * appr_F)

        self.load_reference_solution()
        Psi = self.ref_sol['Psi']

        if dir_path is None:
            dir_path = self.dir_path

        pl = Plot(dir_path, file_name)
        pl.set_ylim(bottom=0, top=self.alpha * 2)
        pl.mgf(x, Psi, appr_Psi)

    def plot_appr_free_energy(self, file_name, dir_path=None):
        x = self.domain_h

        Vbias = self.bias_potential(x)
        appr_F = Vbias / 2

        self.load_reference_solution()
        F = self.ref_sol['F']

        if dir_path is None:
            dir_path = self.dir_path

        pl = Plot(dir_path, file_name)
        pl.set_ylim(bottom=0, top=self.alpha * 3)
        pl.free_energy(x, F, appr_F)

    def plot_control(self, file_name, dir_path=None):
        x = self.domain_h

        u = self.control(x)

        self.load_reference_solution()
        u_opt = self.ref_sol['u_opt']

        if dir_path is None:
            dir_path = self.dir_path

        pl = Plot(dir_path, file_name)
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        pl.control(x, u_opt, u)

    def plot_potential_and_tilted_potential(self, file_name, dir_path=None):
        x = self.domain_h

        V = self.potential(x)

        if self.is_drifted:
            Vb = self.bias_potential(x)
        else:
            Vb = np.zeros(x.shape[0])

        self.load_reference_solution()
        F = self.ref_sol['F']
        Vb_opt = 2 * F

        if dir_path is None:
            dir_path = self.dir_path

        pl = Plot(dir_path, file_name)
        pl.set_ylim(bottom=0, top=self.alpha * 10)
        pl.potential_and_tilted_potential(x, V, Vb, Vb_opt)

    def plot_tilted_potential(self, file_name, dir_path=None):
        x = self.domain_h

        V = self.potential(x)

        if self.is_drifted:
            Vb = self.bias_potential(x)
        else:
            Vb = np.zeros(x.shape[0])

        self.load_reference_solution()
        F = self.ref_sol['F']
        Vb_opt = 2 * F

        if dir_path is None:
            dir_path = self.dir_path

        pl = Plot(dir_path, file_name)
        pl.set_ylim(bottom=0, top=self.alpha * 10)
        pl.tilted_potential(x, V, Vb, Vb_opt)

    def plot_tilted_drift(self, file_name, dir_path=None):
        x = self.domain_h

        dV = self.gradient(x)

        if self.is_drifted:
            U = self.control(x)
            dVb = self.bias_gradient(U)
        else:
            dVb = np.zeros(x.shape[0])

        self.load_reference_solution()
        u_opt = self.ref_sol['u_opt']
        dVb_opt = - np.sqrt(2) * u_opt

        if dir_path is None:
            dir_path = self.dir_path

        pl = Plot(dir_path, file_name)
        pl.set_ylim(bottom=-self.alpha * 5, top=self.alpha * 5)
        pl.drift_and_tilted_drift(x, dV, dVb, dVb_opt)
