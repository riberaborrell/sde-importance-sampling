from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_importance_sampling import Sampling
from mds.plots_1d import Plot1d
from mds.utils import make_dir_path, empty_dir, get_example_data_path, get_time_in_hms

import numpy as np
import time

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for the 1D overdamped Langevin SDE'
    parser.add_argument(
        '--k',
        dest='k',
        type=int,
        default=100,
        help='Steps before adding a bias function. Default: 100',
    )
    return parser

def main():
    args = get_parser().parse_args()
    breakpoint()

    alpha = np.array(args.alpha)
    target_set = np.array(args.target_set)

    # start timer
    t_initial = time.time()

    # set random seed
    if args.seed:
        np.random.seed(args.seed)

    # initialize langevin_1d object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
        is_drifted=True,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    k = args.k

    # initialize bias potentials coefficients
    meta_omegas = np.zeros(0)
    meta_mus = np.zeros(0)
    meta_sigmas = np.zeros(0)
    meta_steps = 0

    # metadynamics algorythm for M trajectories 
    for i in range(sample.M):
        omegas, mus, sigmas, steps = metadynamics_algorithm(sample, k, do_plots=args.do_plots)
        # add coefficients
        meta_omegas = np.concatenate((meta_omegas, omegas))
        meta_mus= np.concatenate((meta_mus, mus))
        meta_sigmas = np.concatenate((meta_sigmas, sigmas))
        meta_steps += steps

    # normalize
    meta_omegas /= sample.M

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
    write_meta_report(sample, meta_path, args.seed, k, meta_omegas.shape[0],
                      meta_steps, t_final - t_initial)

    # plot potential and gradient
    if args.do_plots:

        # set bias potential
        theta = meta_omegas / 2
        sample.set_bias_potential(theta, meta_mus, meta_sigmas)

        sample.plot_tilted_potential(
            file_name='tilted_potential',
            dir_path=meta_path,
        )
        sample.plot_tilted_drift(
            file_name='tilted_drift',
            dir_path=meta_path,
        )


def metadynamics_algorithm(sample, k, seed=None, do_plots=False):
    '''
    '''
    alpha = sample.alpha
    beta = sample.beta
    M = sample.M
    xzero = sample.xzero
    dt = sample.dt
    N_lim = sample.N_lim

    # set random seed
    if seed:
        np.random.seed(seed)

    if do_plots:
        dir_path = os.path.join(sample.example_dir_path, 'metadynamics')
        make_dir_path(dir_path)
        empty_dir(dir_path)
        plt1d = Plot1d(dir_path=dir_path)

    # target set
    target_set_min, target_set_max = sample.target_set

    # grid x coordinate
    x = np.linspace(-3, 3, 1000)

    # potential and gradient at the grid
    V = sample.potential(x)
    dV = sample.gradient(x)

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

            theta = omegas[:i] / 2
            sample.set_bias_potential(theta, mus[:i], sigmas[:i])

            if do_plots:
                pass
                # plot tilted potential
                #sample.plot_tilted_potential(
                #    file_name='tilted_potential_i_' + str(i),
                #    dir_path=meta_path,
                #)

                # plot tilted gradient
                #sample.plot_tilted_drift(
                #    file_name='tilted_drift_i_' + str(i),
                #    dir_path=meta_path,
                #)

        # compute drift and diffusion coefficients
        if i == 0:
            drift = - sample.gradient(Xtemp) * dt
        else:
            utemp = sample.control(Xtemp)
            gradient = sample.tilted_gradient(Xtemp, utemp)
            drift = - gradient * dt

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

def write_meta_report(sample, dir_path, seed, k, m, N, c_time):
    file_path = os.path.join(dir_path, 'report.txt')

    # write in file
    f = open(file_path, "w")

    sample.write_sde_parameters(f)
    sample.write_euler_maruyama_parameters(f)
    sample.write_sampling_parameters(f)

    f.write('Metadynamics parameters and statistics\n')
    f.write('seed: {:2.1f}\n'.format(seed))
    f.write('k: {:d}\n'.format(k))
    f.write('m: {:d}\n'.format(m))
    f.write('N: {:d}\n\n'.format(N))

    h, m, s = get_time_in_hms(c_time)
    f.write('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))
    f.close()


if __name__ == "__main__":
    main()
