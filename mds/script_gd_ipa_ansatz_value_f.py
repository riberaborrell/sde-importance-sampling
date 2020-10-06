from ansatz_functions import derivative_normal_pdf
from potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES
from script_utils import get_reference_solution, \
                         get_F_opt_at_x, \
                         get_derivative_ansatz_functions, \
                         set_unif_dist_ansatz_functions, \
                         set_unif_dist_ansatz_functions_on_S, \
                         get_optimal_coefficients2, \
                         get_meta_coefficients2, \
                         set_initial_coefficients, \
                         control_on_grid2, \
                         free_energy_on_grid2, \
                         plot_control, \
                         plot_free_energy, \
                         plot_tilted_potential, \
                         plot_gd_tilted_potentials, \
                         plot_gd_losses, \
                         save_gd_statistics, \
                         write_gd_report

from utils import empty_dir, get_example_data_path, get_ansatz_data_path, get_gd_data_path
from validation import is_valid_control

import argparse
import numpy as np
import time

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
        choices=POTENTIAL_NAMES,
        default='1d_sym_2well',
        help='Set the potential for the 1D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1],
        help='Set the parameter alpha for the chosen potential. Default: [1]',
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
        '--epochs-lim',
        dest='epochs_lim',
        type=int,
        default=50,
        help='Set maximal number of epochs. Default: 10',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.1,
        help='Set learning rate. Default: 0.1',
    )
    parser.add_argument(
        '--atol',
        dest='atol',
        type=float,
        default=0.01,
        help='Set absolute tolerance between value funtion and loss at xinit. Default: 0.01',
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
        '--sigma',
        dest='sigma',
        type=float,
        default=0.1,
        help='Set the standard deviation of the gaussian ansatz functions \
              that you want to use. Default: 0.1',
    )
    parser.add_argument(
        '--theta-init',
        dest='theta_init',
        choices=['null', 'meta', 'optimal'],
        default='optimal',
        help='Type of initial control. Default: optimal',
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

    # start timer
    t_initial = time.time()

    # set seed
    if args.seed:
        np.random.seed(args.seed)

    # set potential
    alpha = np.array(args.alpha)
    potential, gradient = get_potential_and_gradient(args.potential_name, alpha)

    # sampling parameters
    beta = args.beta
    xzero = args.xzero
    target_set = np.array(args.target_set)
    M = args.M

    # set gd path
    example_dir_path = get_example_data_path(args.potential_name, alpha, beta, target_set)
    ansatz_dir_path = get_ansatz_data_path(example_dir_path, 'gaussian-ansatz', args.m, args.sigma)
    gd_dir_path = get_gd_data_path(ansatz_dir_path, 'ipa-value-f', args.theta_init, args.lr)

    # get ref sol path
    ref_sol_dir_path = get_example_data_path(args.potential_name, alpha, beta,
                                             target_set, 'reference_solution')

    # get reference solution
    omega_h, F_opt, u_opt = get_reference_solution(ref_sol_dir_path)

    # get value function at xzero
    value_f = get_F_opt_at_x(omega_h, F_opt, xzero)

    # ansatz functions basis
    m = args.m
    sigma = args.sigma
    mus_min, mus_max = (-3, 3)
    mus, sigmas = set_unif_dist_ansatz_functions(mus_min, mus_max, m, sigma)
    #mus, sigmas = set_unif_dist_ansatz_functions_on_S(mus_min, mus_max, target_set, m)

    # get optimal coefficients
    theta_opt = get_optimal_coefficients2(omega_h, target_set, F_opt, mus, sigmas)

    # get meta coefficients
    theta_meta = get_meta_coefficients2(args.potential_name, alpha, beta,
                                        target_set, omega_h, mus, sigmas)
    # set gd parameters
    epochs_lim = args.epochs_lim
    lr = args.lr
    atol = args.atol

    # coefficients, performance function, control and free energy
    thetas = np.zeros((epochs_lim + 1, m))
    loss = np.zeros(epochs_lim + 1)
    u = np.zeros((epochs_lim + 1, len(omega_h)))
    F = np.zeros((epochs_lim + 1, len(omega_h)))

    # set initial coefficients
    thetas[0, :] = set_initial_coefficients(m, np.mean(sigmas), theta_opt,
                                            theta_meta, args.theta_init)

    # gradient descent
    for epoch in range(epochs_lim):
        print(epoch)

        # compute control and free energy on the grid
        u[epoch, :] = control_on_grid2(omega_h, thetas[epoch, :], mus, sigmas)
        F[epoch, :] = free_energy_on_grid2(omega_h, target_set, thetas[epoch, :], mus, sigmas)

        # plot control, free_energy and tilted potential
        if args.do_plots:
            plot_control(gd_dir_path, epoch, omega_h, u_opt, u[epoch])
            plot_free_energy(gd_dir_path, epoch, omega_h, potential, F_opt, F[epoch])
            plot_tilted_potential(gd_dir_path, epoch, omega_h, potential, F_opt, F[epoch])

        # get loss and its gradient 
        succ, loss[epoch], grad_loss = sample_loss(gradient, alpha, beta, xzero, target_set,
                                                   M, m, thetas[epoch], mus, sigmas)
        # check if sample succeeded
        if not succ:
            break

        print('{:2.3f}, {:2.3f}'.format(value_f, loss[epoch]))

        # check if we are close enought to the optimal
        if np.isclose(value_f, loss[epoch], atol=atol):
            break

        # Update parameters
        thetas[epoch + 1, :] = thetas[epoch, :] - lr * grad_loss

    # if num of max epochs not reached
    if epoch < epochs_lim - 1:
        loss[epoch+1:] = np.nan
        u[epoch+1:] = np.nan
        F[epoch+1:] = np.nan

    # plot titled potential and loss per epoch
    if args.do_plots and succ:
        plot_gd_tilted_potentials(gd_dir_path, omega_h, potential, F_opt, F[:epoch+1])
        plot_gd_losses(gd_dir_path, value_f, loss[:epoch+1])

    # save gd statistics
    if succ:
        save_gd_statistics(gd_dir_path, omega_h, u[:epoch+1], F[:epoch+1], loss[:epoch+1], value_f)

    # end timer
    t_final = time.time()

    # write gd report
    write_gd_report(gd_dir_path, xzero, value_f, M, m, epochs_lim, epoch+1, lr, atol,
                    loss[epoch], t_final - t_initial)


def sample_loss(gradient, alpha, beta, xzero, target_set, M, m, theta, mus, sigmas):

    # sampling parameters
    target_set_min, target_set_max = target_set

    # Euler-Majurama
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
        btemp = - np.sqrt(2) * get_derivative_ansatz_functions(Xtemp, mus, sigmas)
        utemp = np.dot(btemp, theta)
        if not is_valid_control(utemp, -alpha * 10, alpha * 10):
            return False, None, None

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

    return True, mean_J, mean_grad_J


if __name__ == "__main__":
    main()
