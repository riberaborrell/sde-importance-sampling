from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient
from reference_solution import langevin_1d_reference_solution

import argparse
import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

def get_parser():
    parser = argparse.ArgumentParser(description='Computes the reference solution for the 1D '
                                                 'overdamped Langevin')
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
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    potential, gradient = get_potential_and_gradient(args.potential_name)

    # compute reference solution
    sol = langevin_1d_reference_solution(
        gradient=gradient,
        beta=args.beta,
        target_set_min=args.target_set[0],
        target_set_max=args.target_set[1],
    )
    sol.compute_reference_solution()

    #TODO approximate the first hitting time

    # save optimal performance function from the reference solution
    np.savez(
        os.path.join(DATA_PATH, 'langevin1d_reference_solution.npz'),
        omega_h=sol.omega_h,
        Psi=sol.Psi,
        F=sol.F,
        u_opt=sol.u_opt,
    )

    if args.do_plots:
        pl = Plot()

        # plot optimal performance function from the reference solution 
        X = sol.omega_h
        F = sol.F
        pl.file_name = 'performance_function_reference_solution'
        pl.performance_function(X, F)

        # plot tilted optimal potential and gradient from reference solution
        V = potential(X)
        DV = gradient(X)
        Vbias = 2 * F
        U = sol.u_opt
        DVbias = - np.sqrt(2) * U

        pl.file_name = 'potential_and_gradient_reference_solution'
        pl.tilted_potential_and_gradient(X, V, DV, Vbias, DVbias)

if __name__ == "__main__":
    main()
