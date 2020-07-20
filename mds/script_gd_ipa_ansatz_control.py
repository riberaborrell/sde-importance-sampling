from decorators import timer
from potentials_and_gradients import get_potential_and_gradient
from script_utils import get_reference_solution, \
                         get_F_opt_at_x, \
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
from utils import empty_dir, get_data_path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import lsq_linear

import argparse

import os

def get_parser():
    parser = argparse.ArgumentParser(description='IPA')
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
        '--theta-init',
        dest='theta_init',
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

    # set gd path
    gd_path = get_data_path(args.potential_name, args.alpha, args.beta,
                            args.target_set, 'gd-ipa-ansatz-control')
    empty_dir(gd_path)

    # get ref sol path
    ref_sol_path = get_data_path(args.potential_name, args.alpha, args.beta,
                                 args.target_set, 'reference_solution')

    # set potential
    potential, gradient = get_potential_and_gradient(args.potential_name, args.alpha)

    # sampling parameters
    beta = args.beta
    xzero = args.xzero
    target_set = args.target_set
    M = args.M

    # get reference solution
    omega_h, F_opt, u_opt = get_reference_solution(ref_sol_path)

    # get value function at xzero
    value_f = get_F_opt_at_x(omega_h, F_opt, xzero)

    # ansatz functions basis
    m = args.m
    mus_min, mus_max = (-2, 2)
    mus, sigmas = set_unif_dist_ansatz_functions(mus_min, mus_max, target_set, m)

    # plot ansatz functions
    plot_ansatz_functions(gd_path, omega_h, mus, sigmas)

    # get optimal coefficients
    theta_opt = get_optimal_coefficients(omega_h, target_set, u_opt, mus, sigmas)

    # get meta coefficients
    theta_meta = get_meta_coefficients(args.potential_name, args.alpha, args.beta,
                                       args.target_set, omega_h, mus, sigmas)

    # set gd parameters
    lr = args.lr
    epochs = args.epochs

    # coefficients, performance function, control and free energy
    thetas = np.zeros((epochs + 1, m))
    loss = np.zeros(epochs)
    u = np.zeros((epochs, len(omega_h)))
    F = np.zeros((epochs, len(omega_h)))

    # set initial coefficients
    thetas[0, :] = set_initial_coefficients(m, np.mean(sigmas), theta_opt,
                                            theta_meta, args.theta_init)

    # gradient descent
    for epoch in range(epochs):
        print(epoch)

        # plot and save control, free_energy and tilted potential
        u[epoch, :] = control_on_grid(omega_h, thetas[epoch, :], mus, sigmas)
        F[epoch, :] = free_energy_on_grid(omega_h, target_set, thetas[epoch, :], mus, sigmas)
        plot_control(gd_path, epoch, omega_h, u_opt, u[epoch])
        plot_free_energy(gd_path, epoch, omega_h, potential, F_opt, F[epoch])
        plot_tilted_potential(gd_path, epoch, omega_h, potential, F_opt, F[epoch])

        # get loss and its gradient 
        loss[epoch], grad_loss = sample_loss(gradient, beta, xzero, target_set,
                                             M, m, thetas[epoch], mus, sigmas)
        # check if we are close enought to the optimal
        print(value_f, loss[epoch])
        if np.isclose(value_f, loss[epoch], atol=0.05):
            break

        # Update parameters
        thetas[epoch + 1, :] = thetas[epoch, :] - lr * grad_loss

    if epoch < epochs - 1:
        loss[epoch+1:] = np.nan
        u[epoch+1:] = np.nan
        F[epoch+1:] = np.nan

    plot_gd_tilted_potentials(gd_path, omega_h, potential, F_opt, F[:epoch+1])


def sample_loss(gradient, beta, xzero, target_set, M, m, theta, mus, sigmas):

    # sampling parameters
    target_set_min, target_set_max = target_set

    # Euler-Marujama
    dt = 0.001
    N = 100000

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

    for n in np.arange(1, N+1):
        normal_dist_samples = np.random.normal(0, 1, M)

        # Brownian increment
        dB = np.sqrt(dt) * normal_dist_samples

        # control
        btemp = get_ansatz_functions(Xtemp, mus, sigmas)
        utemp = np.dot(btemp, theta)

        # ipa statistics 
        cost_temp += 0.5 * (utemp ** 2) * dt
        grad_phi_temp += (utemp * btemp.T * dt).T
        grad_S_temp -= (np.sqrt(beta) * btemp.T * dB).T

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

    return mean_J, mean_grad_J


def sample_loss_not_vectorized(gradient, beta, xzero, target_set, M, m, theta, mus, sigmas):

    # sampling parameters
    target_set_min, target_set_max = target_set

    # Euler-Marujama
    dt = 0.001
    N = 100000

    # initialize statistics 
    J = np.zeros(M)
    grad_J = np.zeros((M, m))

    for i in np.arange(M):
        Xtemp = xzero
        cost_temp = 0
        grad_phi_temp = np.zeros(m)
        grad_S_temp = np.zeros(m)

        for n in np.arange(1, N+1):
            normal_dist_sample = np.random.normal(0, 1)

            # Brownian increment
            dB = np.sqrt(dt) * normal_dist_sample

            # control
            btemp = get_ansatz_functions(Xtemp, mus, sigmas)
            utemp = np.dot(theta, btemp)

            # ipa statistics 
            cost_temp += 0.5 * (utemp ** 2) * dt
            grad_phi_temp += utemp * btemp * dt
            grad_S_temp -= np.sqrt(beta) * btemp * dB

            # compute gradient
            tilted_gradient = gradient(Xtemp) - np.sqrt(2) * utemp

            # SDE iteration
            drift = - tilted_gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp += drift + diffusion

            # trajectories in the target set
            if (Xtemp > target_set_min and Xtemp < target_set_max):
                break

        # save ipa statistics 
        J[i] = n * dt + cost_temp
        grad_J[i, :] = grad_phi_temp \
                    - (n * dt + cost_temp) * grad_S_temp

    # compute averages
    mean_J = np.mean(J)
    mean_grad_J = np.mean(grad_J, axis=0)

    return mean_J, mean_grad_J

if __name__ == "__main__":
    main()
