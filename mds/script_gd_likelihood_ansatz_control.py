from decorators import timer
from potentials_and_gradients import get_potential_and_gradient
from script_utils import get_reference_solution, \
                         get_ansatz_functions, \
                         set_unif_dist_ansatz_functions, \
                         get_optimal_coefficients, \
                         get_meta_coefficients, \
                         set_initial_coefficients, \
                         control_on_grid, \
                         free_energy_on_grid, \
                         plot_ansatz_functions, \
                         plot_control, \
                         plot_free_energy, \
                         plot_tilted_potential, \
                         plot_gd_tilted_potentials
from utils import make_dir_path, empty_dir, get_data_path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import lsq_linear

import argparse

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Likelihood')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=None,
        help='Set seed. Default: None',
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
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1,
        help='Set the initial position. Default: -1',
    )
    parser.add_argument(
        '--M',
        dest='M',
        type=int,
        default=500,
        help='Set number of trajectories to sample. Default: 500',
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=10,
        help='Set number of epochs. Default: 10',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.1,
        help='Set learning rate. Default: 0.1',
    )
    parser.add_argument(
        '--m',
        dest='m',
        type=int,
        default=20,
        help='Set the number of uniformly distributed ansatz functions \
              that you want to use. Default: 20',
    )
    parser.add_argument(
        '--a-init',
        dest='a_init',
        choices=['null', 'meta', 'optimal'],
        default='optimal',
        help='Type of initial control. Default: optimal',
    )
    return parser

@timer
def main():
    args = get_parser().parse_args()

    # set seed
    if args.seed:
        np.random.seed(args.seed)

    # set data_path and gd path
    data_path = get_data_path(args.potential_name, args.beta, args.target_set)
    gd_path = os.path.join(data_path, 'gd-likelihood-ansatz-control')
    make_dir_path(gd_path)
    empty_dir(gd_path)

    # set potential
    potential, gradient = get_potential_and_gradient(args.potential_name)

    # sampling parameters
    beta = args.beta
    xzero = args.xzero
    target_set = args.target_set
    M = args.M

    # get reference solution
    omega_h, F_opt, u_opt = get_reference_solution(data_path)

    # ansatz functions basis
    m = args.m
    mus_min, mus_max = (-2, 2)
    mus, sigmas = set_unif_dist_ansatz_functions(mus_min, mus_max, target_set, m)

    # plot ansatz functions
    plot_ansatz_functions(gd_path, omega_h, mus, sigmas)

    # get optimal coefficients
    a_opt = get_optimal_coefficients(omega_h, target_set, u_opt, mus, sigmas)

    # get meta coefficients
    a_meta = get_meta_coefficients(args.potential_name, args.beta, args.target_set,
                                   omega_h, mus, sigmas)

    # set gd parameters
    lr = args.lr
    epochs = args.epochs

    # coefficients, performance function, control and free energy
    a_s = np.zeros((epochs + 1, m))
    loss = np.zeros(epochs + 1)
    u = np.zeros((epochs + 1, len(omega_h)))
    F = np.zeros((epochs + 1, len(omega_h)))

    # set initial coefficients
    a_s[0, :] = set_initial_coefficients(m, np.mean(sigmas), a_opt, a_meta, args.a_init)

    # gradient descent
    for epoch in range(epochs):
        print(epoch)
        u[epoch, :] = control_on_grid(omega_h, a_s[epoch, :], mus, sigmas)
        F[epoch, :] = free_energy_on_grid(omega_h, target_set, a_s[epoch, :], mus, sigmas)
        plot_control(gd_path, epoch, omega_h, u_opt, u[epoch])
        plot_free_energy(gd_path, epoch, omega_h, potential, F_opt, F[epoch])
        plot_tilted_potential(gd_path, epoch, omega_h, potential, F_opt, F[epoch])

        mean_fht, mean_cost, mean_grad \
            = sample_loss(gradient, beta, xzero, target_set,
                          M, m, a_s[epoch], mus, sigmas)

        # save performance function
        loss[epoch] = mean_fht + mean_cost

        # gradient of the loss function (gradient of an expectation!)
        grad_loss = mean_grad

        # Update parameters
        a_s[epoch + 1, :] = a_s[epoch, :] - lr * grad_loss

    epoch += 1
    print(epoch)
    u[epoch, :] = control_on_grid(omega_h, a_s[epoch, :], mus, sigmas)
    F[epoch, :] = free_energy_on_grid(omega_h, target_set, a_s[epoch, :], mus, sigmas)
    plot_control(gd_path, epoch, omega_h, u_opt, u[epoch])
    plot_free_energy(gd_path, epoch, omega_h, potential, F_opt, F[epoch])
    plot_tilted_potential(gd_path, epoch, omega_h, potential, F_opt, F[epoch])
    plot_gd_tilted_potentials(gd_path, epochs, omega_h, potential, F_opt, F)


def sample_loss(gradient, beta, xzero, target_set, M, m, a, mus, sigmas):

    # sampling parameters
    target_set_min, target_set_max = target_set

    # Euler-Majurama
    dt = 0.001
    N = 100000

    # initialize statistics 
    fht = np.zeros(M)
    cost = np.zeros(M)
    grad_phi = np.zeros((m, M))
    sto_int = np.zeros((m, M))
    grad = np.zeros((m, M))

    # initialize temp variables
    Xtemp = xzero * np.ones(M)
    cost_temp = np.zeros(M)
    grad_phi_temp = np.zeros((m, M))
    sto_int_temp = np.zeros((m, M))
    grad_temp = np.zeros((m, M))

    # has arrived in target set
    been_in_target_set = np.repeat([False], M)

    for n in np.arange(1, N+1):
        #breakpoint()
        normal_dist_samples = np.random.normal(0, 1, M)

        # Brownian increment
        dB = np.sqrt(dt) * normal_dist_samples

        # control
        btemp = get_ansatz_functions(Xtemp, mus, sigmas)
        utemp = np.dot(a, btemp)

        # likelihood statistics
        cost_temp += 0.5 * (utemp ** 2) * dt
        grad_phi_temp += utemp * btemp * dt
        sto_int_temp += btemp * dB

        # compute gradient
        tilted_gradient = gradient(Xtemp) - np.sqrt(2) * utemp

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
        if len(new_idx) == 0:
            continue

        # update list of indices whose trajectories have been in the target set
        been_in_target_set[new_idx] = True

        # save statistics
        fht[new_idx] = n * dt
        cost[new_idx] = cost_temp[new_idx]
        grad_phi[:, new_idx] = grad_phi_temp[:, new_idx]
        sto_int[:, new_idx] = sto_int_temp[:, new_idx]

        # check if all trajectories have arrived to the target set
        if been_in_target_set.all() == True:
            grad = grad_phi + (fht + cost) * sto_int
            break

    # compute averages
    mean_fht = np.mean(fht)
    mean_cost = np.mean(cost)
    mean_grad = np.mean(grad, axis=1)

    return mean_fht, mean_cost, mean_grad


if __name__ == "__main__":
    main()
