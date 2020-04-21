from plotting import Plot
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient, \
                                     bias_potential, \
                                     bias_gradient

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
METADYNAMICS_DATA_PATH = os.path.join(MDS_PATH, 'data/metadynamics')
METADYNAMICS_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/metadynamics')


def metadynamics_algorithm(beta, xzero, well_set, k, dt, N, seed=None, do_plots=False):
    '''
    '''
    # set random seed
    if seed:
        np.random.seed(seed)

    # check well set
    well_set_min = well_set[0]
    well_set_max = well_set[1]
    if well_set_min >= well_set_max:
        #TODO raise error
        print("The well set interval is not valid")
        exit() 

    # grid x coordinate
    X = np.linspace(-2, 2, 100)

    # potential and gradient at the grid
    V = double_well_1d_potential(X)
    dV = double_well_1d_gradient(X)

    # time interval, time steps and number of time steps
    T = N * dt
    t = np.linspace(0, T, N+1)

    # steps before adding a bias function
    if np.mod(N, k) != 0:
        print("N has to be a multiple of the number of steps k")
        exit()
    # bias functions
    i = 0

    # preallocate mean and standard deviation for the bias funcitons
    mus = np.zeros(int(N/k))
    sigmas = np.zeros(int(N/k))

    # set the weights of the bias functions
    # constant
    #omegas = 0.1 * np.ones(int(N/k))

    # exp factor
    omegas = 0.95 * np.ones(int(N/k))
    omegas = 1 * np.array([w**(i+1) for i, w in enumerate(omegas)])

    # inv proportional 
    #omegas = 0.1 * np.ones(int(N/k))
    #omegas = np.array([w / (i+1) for i, w in enumerate(omegas)])
    
    # set the standard desviation of the bias functions
    # constant
    sigmas = 0.3 * np.ones(int(N/k))
    
    # exp factor
    #sigmas = 0.3 * np.ones(int(N/k))
    #sigmas = 0.2 * np.array([sigma**(i+1) for i, sigma in enumerate(sigmas)])


    # 1D MD SDE: dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x
    Xtemp = xzero
    if do_plots:
        Xem = np.zeros(N+1) 
        Xem[0] = Xtemp

    Xhelp = np.zeros(k+1)

    # Brownian increments
    dB = np.sqrt(dt) * np.random.normal(0, 1, N)

    for n in np.arange(1, N+1): 
        # stop simulation if particle leave the well set T
        if (Xtemp < well_set_min or Xtemp > well_set_max):
            print('The trajectory HAS left the well set!')
            if do_plots:
                Xem[n:N+1] = np.nan
            break

        # every k-steps add new bias function
        if (np.mod(n, k) == 0):
            mus[i] = np.mean(Xhelp)
            #sigmas[i] = 0.2 * np.max(np.abs(Xhelp - mus[i]))
            #sigmas[i] = 20 * np.var(Xhelp)
            Xhelp = np.zeros(k+1)
            i += 1
            
            if do_plots:
                # plot tilted potential and gradient
                Vbias = bias_potential(X, omegas[:i], mus[:i], sigmas[:i])
                dVbias = bias_gradient(X, omegas[:i], mus[:i], sigmas[:i])
                pl = Plot(
                    file_name='tilted_potential_and_gradient_i_' + str(i),
                    file_type='png', 
                    dir_path=METADYNAMICS_FIGURES_PATH,
                )
                pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)
            
        
        # compute gradient
        if i == 0:
            gradient = double_well_1d_gradient(Xtemp)
        else:
            dVbias = gradient_bias_potential(Xtemp, omegas[:i], mus[:i], sigmas[:i])
            gradient = double_well_1d_gradient(Xtemp) + dVbias

        # compute drift and diffusion coefficients
        drift = - gradient * dt
        diffusion = np.sqrt(2 / beta) * dB[n-1]

        # compute Xtemp
        Xtemp = Xtemp + drift + diffusion

        if do_plots:
            Xem[n] = Xtemp
        
        # update Xhelp
        Xhelp[np.mod(n, k)] = Xtemp


    if n == N:
        print('The trajectory has NOT left the well set!')
    if do_plots:
        # plot trajectory
        pl = Plot(
            file_name='trajectory_metadynamics',
            file_type='png',
            dir_path=METADYNAMICS_FIGURES_PATH,
        )
        pl.trajectory(t, Xem)
    
    # report 
    print('Steps: {:8.2f}'.format(n))
    print('Time: {:8.2f}'.format(n*dt))
    print('Bias functions: {:d}'.format(i))

    return omegas[:i], mus[:i], sigmas[:i]

def get_a_from_fake_metadynamics(beta):
    # load metadynamics parameters
    bias_pot_coeff = np.load(os.path.join(METADYNAMICS_DATA_PATH, 'langevin1d_fake_bias_potential.npz'))
    omegas = bias_pot_coeff['omegas']
    meta_mus = bias_pot_coeff['mus']
    meta_sigmas = bias_pot_coeff['sigmas']
    
    a = omegas * beta / 2

    return a, meta_mus, meta_sigmas

def get_a_from_metadynamics(beta, m, J_min, J_max):
    '''
    '''
    # validate input 
    if J_min >= J_max:
        #TODO raise error
        print("The interval J_h is not valid")
   
    # load metadynamics parameters
    #bias_pot_coeff = np.load(os.path.join(METADYNAMICS_DATA_PATH, 'langevin1d_meta_bias_potential.npz'))
    bias_pot_coeff = np.load(os.path.join(METADYNAMICS_DATA_PATH, 'langevin1d_fake_bias_potential.npz'))
    omegas = bias_pot_coeff['omegas']
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
    sigmas = 0.3 * np.ones(m)
    
    # ansatz functions evaluated at the grid
    ansatz_functions = np.zeros((m, m))
    for i in np.arange(m):
        #for j in np.arange(m):
        #    ansatz_functions[i, j] = stats.norm.pdf(X[j], mus[i], sigmas[i])
        ansatz_functions[i, :] = stats.norm.pdf(X, mus[i], sigmas[i])

    # value function evaluated at the grid
    V_bias = bias_potential(X, omegas, meta_mus, meta_sigmas)
    phi = V_bias * beta / 2

    # solve a V = \Phi
    a = np.linalg.solve(ansatz_functions, phi)

    return a, mus, sigmas
