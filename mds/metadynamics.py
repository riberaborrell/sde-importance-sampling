from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient, \
                                     bias_potential, \
                                     bias_gradient

import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
METADYNAMICS_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/metadynamics')

def metadynamics_algorithm(beta, xzero, well_set, k, dt, N, potential_name='sym_2well',
                           seed=None, do_plots=False):
    '''
    '''
    # set random seed
    if seed:
        np.random.seed(seed)

    if do_plots:
        pl = Plot(dir_path=METADYNAMICS_FIGURES_PATH)

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
    potential, gradient = get_potential_and_gradient(potential_name)
    V = potential(X)
    dV = gradient(X)

    # time interval, time steps and number of time steps
    T = N * dt
    t = np.linspace(0, T, N+1)

    # steps before adding a bias function
    if np.mod(N, k) != 0:
        print("N has to be a multiple of the number of steps k")
        exit()
    # bias functions
    i = 0

    # set the weights of the bias functions
    # constant
    #omegas = 0.1 * np.ones(int(N/k))

    # exp factor
    omegas = 0.95 * np.ones(int(N/k))
    omegas = 0.1 * np.array([w**(i+1) for i, w in enumerate(omegas)])

   # inv proportional 
    #omegas = 0.1 * np.ones(int(N/k))
    #omegas = np.array([w / (i+1) for i, w in enumerate(omegas)])

    # preallocate means of the gaussians of the bias functions 
    mus = np.zeros(int(N/k))

    # set the standard desviation of the bias functions
    # constant
    sigmas = 0.2 * np.ones(int(N/k))

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
                pl.file_name = 'tilted_potential_and_gradient_i_' + str(i)
                pl.tilted_potential_and_gradient(X, V, dV, Vbias, dVbias)


        # compute drift and diffusion coefficients
        if i == 0:
            drift = - gradient(Xtemp) * dt
        else:
            dVbias = bias_gradient(Xtemp, omegas[:i], mus[:i], sigmas[:i])
            drift = - (gradient(Xtemp) + dVbias) * dt

        diffusion = np.sqrt(2 / beta) * dB[n-1]

        # compute Xtemp
        Xtemp = Xtemp + drift + diffusion

        if do_plots:
            Xem[n] = Xtemp

        # update Xhelp
        Xhelp[np.mod(n, k)] = Xtemp


    if n == N:
        print('The trajectory has NOT left the well set!')
    #if do_plots:
        # plot trajectory
        #pl.file_name = 'trajectory_metadynamics'
        #pl.trajectory(t, Xem)

    # report 
    print('Steps: {:8.2f}'.format(n))
    print('Time: {:8.2f}'.format(n*dt))
    print('Bias functions: {:d}'.format(i))

    return omegas[:i], mus[:i], sigmas[:i]
