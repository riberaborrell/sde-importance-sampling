from decorators import timer
from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient, \
                                     bias_potential, \
                                     bias_gradient
import sampling
from utils import empty_dir, get_data_path

import argparse
import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Metadynamics')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=['sym_1well', 'sym_2well', 'asym_2well'],
        default='sym_2well',
        help='Set the potential for the 1D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=1,
        help='Set the parameter alpha for the chosen potential. Default: 1',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 1D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1.,
        help='Set the value of the process at time t=0. Default: -1',
    )
    parser.add_argument(
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
    )
    parser.add_argument(
        '--M',
        dest='M',
        type=int,
        default=1,
        help='Set number of trajectories to sample. Default: 1',
    )
    parser.add_argument(
        '--well-set',
        nargs=2,
        dest='well_set',
        type=float,
        default=[-1.7, 0],
        help='Set the well set interval. Default: [-1.7, -0.0]',
    )
    parser.add_argument(
        '--k',
        dest='k',
        type=int,
        default=100,
        help='Steps before adding a bias function. Default: 100',
    )
    parser.add_argument(
        '--dt',
        dest='dt',
        type=float,
        default=0.001,
        help='Set dt. Default: 0.001',
    )
    parser.add_argument(
        '--N',
        dest='N',
        type=int,
        default=10**5,
        help='Set number of time steps. Default: 100.000',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser

@timer
def main():
    args = get_parser().parse_args()

    # set random seed
    if args.seed:
        np.random.seed(args.seed)

    M = args.M

    # initialize bias potentials coefficients
    meta_omegas = np.zeros(0)
    meta_mus = np.zeros(0)
    meta_sigmas = np.zeros(0)

    # metadynamics algorythm for M trajectories 
    for i in range(M):
        omegas, mus, sigmas = metadynamics_algorithm(
            potential_name=args.potential_name,
            alpha=args.alpha,
            beta=args.beta,
            xzero=args.xzero,
            well_set=args.well_set,
            k=args.k,
            dt=args.dt,
            N=args.N,
            do_plots=args.do_plots,
        )
        # add coefficients
        meta_omegas = np.concatenate((meta_omegas, omegas))
        meta_mus= np.concatenate((meta_mus, mus))
        meta_sigmas = np.concatenate((meta_sigmas, sigmas))

    # normalize
    meta_omegas /= M

    # initialize langevin_1d object
    sample = sampling.langevin_1d(
        potential_name=args.potential_name,
        alpha=args.alpha,
        beta=args.beta,
        target_set=args.target_set,
    )

    # set bias potential
    a = meta_omegas / 2
    sample.set_bias_potential(a, meta_mus, meta_sigmas)

    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential(file_name='metadynamics_tilted_potential')
        sample.plot_tilted_drift(file_name='metadynamics_tilted_drift')

    # save bias
    dir_path = get_data_path(args.potential_name, args.alpha,
                             args.beta, args.target_set, 'metadynamics')
    np.savez(
        os.path.join(dir_path, 'bias_potential.npz'),
        omegas=meta_omegas,
        mus=meta_mus,
        sigmas=meta_sigmas,
    )

def metadynamics_algorithm(potential_name, alpha, beta, xzero, well_set, k, dt, N,
                           target_set=[0.9, 1.1], seed=None, do_plots=False):
    '''
    '''
    # set random seed
    if seed:
        np.random.seed(seed)

    if do_plots:
        dir_path = get_data_path(potential_name, alpha, beta, target_set, 'metadynamics')
        empty_dir(dir_path)
        pl = Plot(dir_path=dir_path)

    # target set
    target_set_min, target_set_max = target_set

    # check well set
    well_set_min, well_set_max = well_set
    if well_set_min >= well_set_max:
        #TODO raise error
        print("The well set interval is not valid")
        exit()

    # grid x coordinate
    X = np.linspace(-3, 3, 1000)

    # potential and gradient at the grid
    potential, gradient = get_potential_and_gradient(potential_name, alpha)
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
    omegas = 0.2 * alpha * np.array([w**(i+1) for i, w in enumerate(omegas)])

    # inv proportional 
    #omegas = 0.1 * np.ones(int(N/k))
    #omegas = np.array([w / (i+1) for i, w in enumerate(omegas)])

    # preallocate means of the gaussians of the bias functions 
    mus = np.zeros(int(N/k))

    # set the standard desviation of the bias functions
    # constant
    sigmas = 0.2 * np.ones(int(N/k))


    # 1D MD SDE: dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x
    Xtemp = xzero
    if do_plots:
        Xem = np.zeros(N+1)
        Xem[0] = Xtemp

    Xhelp = np.zeros(k+1)

    # Brownian increments
    dB = np.sqrt(dt) * np.random.normal(0, 1, N)

    for n in np.arange(1, N+1):
        # stop simulation if particle arrives in the target set
        if (Xtemp >= target_set_min and Xtemp <= target_set_max):
            print('The trajectory HAS left the well set!')
            if do_plots:
                Xem[n:N+1] = np.nan
            break

        # every k-steps add new bias function
        if (np.mod(n, k) == 0):
            mus[i] = np.mean(Xhelp)
            #sigmas[i] = 10 * np.var(Xhelp)
            #print('{:2.3f}, {:2.3f}'.format(np.mean(Xhelp), np.var(Xhelp)))
            #sigmas[i] = 0.2 * np.max(np.abs(Xhelp - mus[i]))
            #sigmas[i] = 20 * np.var(Xhelp)
            Xhelp = np.zeros(k+1)
            i += 1

            if do_plots:
                # plot tilted potential
                Vbias = bias_potential(X, omegas[:i], mus[:i], sigmas[:i])
                pl.file_name = 'tilted_potential_i_' + str(i)
                pl.set_ylim(bottom=0, top=alpha * 10)
                #pl.potential_and_tilted_potential(X, V, Vbias)

                # plot tilted gradient
                dVbias = bias_gradient(X, omegas[:i], mus[:i], sigmas[:i])
                pl.file_name = 'tilted_drift_i_' + str(i)
                pl.set_ylim(bottom=-alpha * 5, top=alpha * 5)
                #pl.drift_and_tilted_drift(X, dV, dVbias)

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

if __name__ == "__main__":
    main()
