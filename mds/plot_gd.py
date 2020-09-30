from potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES
from script_utils import get_reference_solution, \
                         plot_control, \
                         plot_free_energy, \
                         plot_tilted_potential, \
                         plot_gd_tilted_potentials, \
                         plot_gd_losses, \
                         plot_gd_losses_bar

from utils import get_example_data_path, get_ansatz_data_path, get_gd_data_path

import argparse
import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Plot IPA')
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
        '--m',
        dest='m',
        type=float,
        default=30,
        help='Set number of ansatz functions. Default: 30',
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
        '--lr',
        dest='lr',
        type=float,
        default=0.1,
        help='Set learning rate. Default: 0.1',
    )
    parser.add_argument(
        '--theta-init',
        dest='theta_init',
        choices=['null', 'meta', 'optimal'],
        default='optimal',
        help='Type of initial control. Default: optimal',
    )
    return parser

def main():
    args = get_parser().parse_args()
    alpha = np.array(args.alpha)
    target_set = np.array(args.target_set)

    # set potential
    potential, gradient = get_potential_and_gradient(args.potential_name, alpha)

    # get ref sol path
    ref_sol_path = get_example_data_path(args.potential_name, alpha, args.beta,
                                         target_set, 'reference_solution')

    # set gd path
    example_dir_path = get_example_data_path(args.potential_name, alpha, args.beta, target_set)
    ansatz_dir_path = get_ansatz_data_path(example_dir_path, 'gaussian-ansatz', args.m, args.sigma)
    gd_dir_path = get_gd_data_path(ansatz_dir_path, 'ipa-value-f', args.lr)

    # get reference solution
    omega_h, F_opt, u_opt = get_reference_solution(ref_sol_path)

    # get gd
    gd = np.load(os.path.join(gd_dir_path, 'gd.npz'))
    domain_h = gd['domain_h']
    epochs = gd['epochs'].shape[0]
    u = gd['u']
    F = gd['F']
    loss = gd['loss']
    value_f = gd['value_f']

    # plot each epoch
    for epoch in range(epochs):
        plot_control(gd_dir_path, epoch, omega_h, u_opt, u[epoch])
        plot_free_energy(gd_dir_path, epoch, omega_h, potential, F_opt, F[epoch])
        plot_tilted_potential(gd_dir_path, epoch, omega_h, potential, F_opt, F[epoch])
        pass

    # plot all epochs
    plot_gd_tilted_potentials(gd_dir_path, omega_h, potential, F_opt, F)
    plot_gd_losses(gd_dir_path, value_f, loss)
    plot_gd_losses_bar(gd_dir_path, value_f, loss)

if __name__ == "__main__":
    main()
