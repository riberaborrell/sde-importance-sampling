from hjb_2d_solver import langevin_2d_hjb_solver
from potentials_and_gradients import POTENTIAL_NAMES

import argparse
import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Plots the potential landscape')
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='2d_2well',
        help='Set the type of potential to plot. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1, 1, 1, 1],
        help='Set the parameters for the 2D MD SDE potential. Default: [1, 1, 1, 1]',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 2D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--target-set',
        nargs=4,
        dest='target_set',
        type=float,
        default=[0.9, 1.1, 0.9, 1.1],
        help='Set the target set interval. Default: [[0.9, 1.1],[0.9, 1.1]]',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # get solver
    sol = langevin_2d_hjb_solver(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set).reshape((2, 2)),
    )

    # load already computed solution
    ref_sol = np.load(
        os.path.join(sol.dir_path, 'reference_solution.npz'),
        allow_pickle=True,
    )
    sol.meshgrid = ref_sol['meshgrid']
    sol.Psi = ref_sol['Psi']
    sol.F = ref_sol['F']
    sol.u_opt_x = ref_sol['u_opt_x']
    sol.u_opt_y = ref_sol['u_opt_y']

    # plot solution
    sol.plot_psi()
    sol.plot_free_energy()
    #sol.plot_optimal_tilted_potential()
    #sol.plot_optimal_tilted_drift()
    sol.plot_optimal_control()

if __name__ == "__main__":
    main()
