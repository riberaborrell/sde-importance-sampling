from decorators import timer
from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient, \
                                     bias_potential, \
                                     bias_gradient
import sampling
from utils import make_dir_path, empty_dir, get_data_path

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

    omegas, mus, sigmas = metadynamics_algorithm(
        potential_name=args.potential_name,
        beta=args.beta,
        xzero=args.xzero,
        well_set=args.well_set,
        k=args.k,
        dt=args.dt,
        N=args.N,
        do_plots=args.do_plots,
        seed=args.seed,
    )

    # initialize langevin_1d object
    sample = sampling.langevin_1d(
        potential_name=args.potential_name,
        beta=args.beta,
        target_set=args.target_set,
    )

    # set bias potential
    a = omegas / 2
    sample.set_bias_potential(a, mus, sigmas)

    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential(file_name='metadynamics_tilted_potential')
        sample.plot_tilted_drift(file_name='metadynamics_tilted_drift')

    # save bias
    dir_path = get_data_path(args.potential_name, args.beta, args.target_set)
    np.savez(
        os.path.join(dir_path, 'metadynamics_bias_potential.npz'),
        omegas=omegas,
        mus=mus,
        sigmas=sigmas,
    )

def metadynamics_algorithm(potential_name, beta, xzero, well_set, k, dt, N,
                           target_set=[0.9, 1.1], seed=None, do_plots=False):
    '''
    '''
    # set random seed
    if seed:
        np.random.seed(seed)

    if do_plots:
        dir_path = os.path.join(
            get_data_path(potential_name, beta, target_set),
            'metadynamics',
        )
        make_dir_path(dir_path)
        empty_dir(dir_path)
        pl = Plot(dir_path=dir_path)

    # check well set
    well_set_min = well_set[0]
    well_set_max = well_set[1]
    if well_set_min >= well_set_max:
        #TODO raise error
        print("The well set interval is not valid")
        exit()

    # grid x coordinate
    X = np.linspace(-3, 3, 1000)

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
                # plot tilted potential
                Vbias = bias_potential(X, omegas[:i], mus[:i], sigmas[:i])
                pl.file_name = 'tilted_potential_i_' + str(i)
                pl.potential_and_tilted_potential(X, V, Vbias)

                # plot tilted gradient
                dVbias = bias_gradient(X, omegas[:i], mus[:i], sigmas[:i])
                pl.file_name = 'tilted_drift_i_' + str(i)
                pl.drift_and_tilted_drift(X, dV, dVbias)

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
