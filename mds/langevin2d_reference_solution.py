from hjb_2d_solver import langevin_2d_hjb_solver
from potentials_and_gradients import POTENTIAL_NAMES

import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(
        description='Computes the numerical solution of the HJB equation associated to the'
                    ' overdamped Langevin SDE'
    )
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='2d_2well',
        help='Set the potential for the 2D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        nargs='2',
        dest='alpha',
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
        '--domain',
        nargs=4,
        dest='domain',
        type=float,
        default=[-3, 3, -3, 3],
        help='Set the domain set. Default: [[-3, 3],[-3, 3]]',
    )
    parser.add_argument(
        '--target-set',
        nargs=4,
        dest='target_set',
        type=float,
        default=[0.9, 1.1, 0.9, 1.1],
        help='Set the target set interval. Default: [[0.9, 1.1],[0.9, 1.1]]',
    )
    parser.add_argument(
        '--h',
        dest='h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
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

    # compute reference solution
    sol = langevin_2d_hjb_solver(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain).reshape((2, 2)),
        target_set=np.array(args.target_set).reshape((2, 2)),
        h=args.h,
    )
    sol.discretize_domain()
    sol.solve_bvp()
    sol.compute_optimal_control()
    sol.save_reference_solution()

    if args.do_plots:
        sol.plot_psi()
        sol.plot_free_energy()
        #sol.plot_optimal_tilted_potential()
        #sol.plot_optimal_tilted_drift()
        sol.plot_optimal_control()


if __name__ == "__main__":
    main()
