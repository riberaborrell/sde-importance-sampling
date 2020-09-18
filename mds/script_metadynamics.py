from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES
from ansatz_functions import bias_potential, \
                             bias_gradient
import sampling
from utils import empty_dir, get_example_data_path, get_time_in_hms

import argparse
import numpy as np
import time

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
        default='1d_sym_2well',
        choices=POTENTIAL_NAMES,
        help='Set the potential for the 1D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1],
        help='Set the parameter alpha for the chosen potentials. Default: [1]',
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
        '--N-lim',
        dest='N_lim',
        type=int,
        default=10**5,
        help='Set maximal number of time steps. Default: 100,000',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    alpha = np.array(args.alpha)
    target_set = np.array(args.target_set)

    # start timer
    t_initial = time.time()

    # set random seed
    if args.seed:
        np.random.seed(args.seed)

    xzero = args.xzero
    M = args.M
    k = args.k

    # initialize bias potentials coefficients
    meta_omegas = np.zeros(0)
    meta_mus = np.zeros(0)
    meta_sigmas = np.zeros(0)
    meta_steps = 0

    # metadynamics algorythm for M trajectories 
    for i in range(M):
        omegas, mus, sigmas, steps = metadynamics_algorithm(
            potential_name=args.potential_name,
            alpha=alpha,
            beta=args.beta,
            xzero=xzero,
            target_set=target_set,
            well_set=args.well_set,
            k=k,
            dt=args.dt,
            N_lim=args.N_lim,
            do_plots=args.do_plots,
        )
        # add coefficients
        meta_omegas = np.concatenate((meta_omegas, omegas))
        meta_mus= np.concatenate((meta_mus, mus))
        meta_sigmas = np.concatenate((meta_sigmas, sigmas))
        meta_steps += steps

    # normalize
    meta_omegas /= M

    # initialize langevin_1d object
    sample = sampling.langevin_1d(
        potential_name=args.potential_name,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
    )

    # set bias potential
    a = meta_omegas / 2
    sample.set_bias_potential(a, meta_mus, meta_sigmas)

    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential(file_name='metadynamics_tilted_potential')
        sample.plot_tilted_drift(file_name='metadynamics_tilted_drift')

    # save bias
    meta_path = get_example_data_path(args.potential_name, alpha,
                              args.beta, target_set, 'metadynamics')
    np.savez(
        os.path.join(meta_path, 'bias_potential.npz'),
        omegas=meta_omegas,
        mus=meta_mus,
        sigmas=meta_sigmas,
    )

    # end timer
    t_final = time.time()

    # write report
    write_meta_report(meta_path, args.seed, args.xzero, M, k,
                      omegas.shape[0], meta_steps, t_final - t_initial)


def metadynamics_algorithm(potential_name, alpha, beta, xzero, well_set, k, dt, N_lim,
                           target_set=[0.9, 1.1], seed=None, do_plots=False):
    '''
    '''
    # set random seed
    if seed:
        np.random.seed(seed)

    if do_plots:
        dir_path = get_example_data_path(potential_name, alpha, beta, target_set, 'metadynamics')
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

    # maximal time interval, time steps and number of time steps
    T = N_lim * dt
    t = np.linspace(0, T, N_lim+1)

    # steps before adding a bias function
    if np.mod(N_lim, k) != 0:
        print("N_lim has to be a multiple of the number of steps k")
        exit()
    # bias functions
    i = 0

    # set the weights of the bias functions
    # constant
    #omegas = 0.1 * np.ones(int(N_lim/k))

    # exp factor
    omegas = 0.95 * np.ones(int(N_lim/k))
    omegas = 0.2 * alpha * np.array([w**(i+1) for i, w in enumerate(omegas)])

    # inv proportional 
    #omegas = 0.1 * np.ones(int(N_lim/k))
    #omegas = np.array([w / (i+1) for i, w in enumerate(omegas)])

    # preallocate means of the gaussians of the bias functions 
    mus = np.zeros(int(N_lim/k))

    # set the standard desviation of the bias functions
    # constant
    sigmas = 0.2 * np.ones(int(N_lim/k))


    # 1D MD SDE: dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x
    Xtemp = xzero
    if do_plots:
        Xem = np.zeros(N_lim+1)
        Xem[0] = Xtemp

    Xhelp = np.zeros(k+1)

    # Brownian increments
    dB = np.sqrt(dt) * np.random.normal(0, 1, N_lim)

    for n in np.arange(1, N_lim+1):
        # stop simulation if particle arrives in the target set
        if (Xtemp >= target_set_min and Xtemp <= target_set_max):
            if do_plots:
                Xem[n:N_lim+1] = np.nan
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


    if n == N_lim:
        print('The trajectory has NOT left the well set!')
    #if do_plots:
        # plot trajectory
        #pl.file_name = 'trajectory_metadynamics'
        #pl.trajectory(t, Xem)

    return omegas[:i], mus[:i], sigmas[:i], n

def write_meta_report(dir_path, seed, xzero, M, k, m, N, c_time):
    file_path = os.path.join(dir_path, 'report.txt')

    # write in file
    f = open(file_path, "w")

    f.write('Metadynamics parameters and statistics\n')
    f.write('seed: {:2.1f}\n'.format(seed))
    f.write('xzero: {:2.1f}\n'.format(xzero))
    f.write('sampled trajectories: {:,d}\n'.format(M))
    f.write('k: {:d}\n'.format(k))
    f.write('m: {:d}\n'.format(m))
    f.write('N: {:d}\n'.format(N))
    h, m, s = get_time_in_hms(c_time)
    f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))


if __name__ == "__main__":
    main()
