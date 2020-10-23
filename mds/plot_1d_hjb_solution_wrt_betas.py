from mds.hjb_1d_solver import langevin_hjb_1d_solver
from mds.plots_1d import Plot1d
from mds.potentials_and_gradients import get_potential_and_gradient, POTENTIAL_NAMES
from mds.utils import get_data_path

import argparse
import numpy as np
import matplotlib.pyplot as plt

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Plots the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='1d_sym_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
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
        '--betas',
        dest='betas',
        type=float,
        nargs='+',
        default=[1.0],
        help='Set list of betas for the 1D MD SDE. Default: [1.0]',
    )
    parser.add_argument(
        '--domain',
        nargs=2,
        dest='domain',
        type=float,
        default=[-3, 3],
        help='Set the domain interval. Default: [-3, 3]',
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
    alpha = np.array(args.alpha)
    betas = np.array(args.betas)

    # get discretized grid
    # get solver for the first beta
    sol = langevin_hjb_1d_solver(
        potential_name=args.potential_name,
        alpha=alpha,
        beta=betas[0],
        target_set=np.array(args.target_set),
    )

    # get grid, potential and gradient
    ref_sol = np.load(os.path.join(sol.dir_path, 'reference_solution.npz'))
    X = ref_sol['domain_h']
    V = sol.potential(X)
    dV = sol.gradient(X)

    # initialize F and u_opt
    F = np.zeros((betas.shape[0], X.shape[0]))
    u_opt = np.zeros((betas.shape[0], X.shape[0]))

    # get F and u_opt for each beta
    for i, beta in enumerate(betas):
        sol = langevin_hjb_1d_solver(
            potential_name=args.potential_name,
            alpha=alpha,
            beta=beta,
            target_set=np.array(args.target_set),
        )
        ref_sol = np.load(os.path.join(sol.dir_path, 'reference_solution.npz'))
        F[i, :] = ref_sol['F']
        u_opt[i, :] = ref_sol['u_opt']

    # get path
    dir_path = get_data_path(args.potential_name)

    # plot free energy
    file_name = 'free_energy_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=0, top=alpha[0] * 3)
    plot.free_energy_wrt_betas(X, betas, F)

    # plot tilted potential
    file_name = 'optimal_tilted_potential_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=0, top=alpha[0] * 10)
    plot.tilted_potential_wrt_betas(X, V, betas, 2 * F)

    # plot tilted drift 
    file_name = 'optimal_tilted_drift_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=-alpha[0] * 5, top=alpha[0] * 5)
    plot.tilted_drift_wrt_betas(X, dV, betas, -np.sqrt(2) * u_opt)

    # plot control
    file_name = 'optimal_control_wrt_betas'
    plot = Plot(dir_path, file_name)
    plot.set_ylim(bottom=-alpha[0] * 5, top=alpha[0] * 5)
    plot.control_wrt_betas(X, betas, u_opt)


if __name__ == "__main__":
    main()
