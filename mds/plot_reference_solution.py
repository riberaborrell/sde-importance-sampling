from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient
from utils import get_data_path

import argparse
import numpy as np
import matplotlib.pyplot as plt

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Plots the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=['sym_1well', 'sym_2well', 'asym_2well'],
        default='sym_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=1,
        help='Set the parameter alpha for the chosen potential. Default: 1',
    )
    parser.add_argument(
        '--betas',
        dest='betas',
        type=float,
        nargs='+',
        default=[1.0],
        help='Set list of betas for the 1D MD SDE. Default: [1.0]',
    )
    parser.add_argument(
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
    )
    return parser

def main():
    args = get_parser().parse_args()
    betas = np.array(args.betas)

    # get discretized grid
    ref_sol_path = get_data_path(args.potential_name, args.alpha, betas[0],
                                 args.target_set, 'reference_solution')
    ref_sol = np.load(os.path.join(ref_sol_path, 'reference_solution.npz'))
    X = ref_sol['omega_h']

    # get potential on grid
    potential, gradient = get_potential_and_gradient(args.potential_name, args.alpha)
    V = potential(X)
    dV = gradient(X)

    # initialize F and u_opt
    F = np.zeros((betas.shape[0], X.shape[0]))
    u_opt = np.zeros((betas.shape[0], X.shape[0]))

    # load F and u_opt
    for i, beta in enumerate(betas):
        ref_sol_path = get_data_path(args.potential_name, args.alpha, beta,
                                      args.target_set, 'reference_solution')
        ref_sol = np.load(os.path.join(ref_sol_path, 'reference_solution.npz'))
        F[i, :] = ref_sol['F']
        u_opt[i, :] = ref_sol['u_opt']

    # get path
    dir_path = get_data_path(args.potential_name)

    # plot free energy
    file_name = 'free_energy_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=0, top=args.alpha * 2.5)
    plot.free_energy_wrt_betas(X, betas, F)

    # plot tilted potential
    file_name = 'optimal_tilted_potential_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=0, top=args.alpha * 10)
    plot.tilted_potential_wrt_betas(X, V, betas, 2 * F)

    # plot tilted drift 
    file_name = 'optimal_tilted_drift_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=-args.alpha * 5, top=args.alpha * 5)
    plot.tilted_drift_wrt_betas(X, dV, betas, -np.sqrt(2) * u_opt)

    # plot control
    file_name = 'optimal_control_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=-args.alpha * 5, top=args.alpha * 5)
    plot.control_wrt_betas(X, betas, u_opt)


if __name__ == "__main__":
    main()
