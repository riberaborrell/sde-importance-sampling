from plotting import Plot
from potentials_and_gradients import get_potential_and_gradient
from reference_solution import langevin_1d_reference_solution
from utils import get_data_path

import argparse
import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Computes the reference solution for the 1D overdamped Langevin')
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
        '--alpha',
        dest='alpha',
        type=float,
        default=1,
        help='Set the parameter alpha for the chosen potential. Default: 1',
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
        '--h',
        dest='h',
        type=float,
        default=0.001,
        help='Set the discretization step size. Default: 0.001',
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

    potential, gradient = get_potential_and_gradient(args.potential_name, args.alpha)

    # compute reference solution
    sol = langevin_1d_reference_solution(
        gradient=gradient,
        beta=args.beta,
        target_set=args.target_set,
        h=args.h,
    )
    sol.compute_reference_solution()

    #TODO approximate the first hitting time

    # save optimal performance function from the reference solution
    dir_path = get_data_path(args.potential_name, args.beta, args.target_set)
    np.savez(
        os.path.join(dir_path, 'reference_solution.npz'),
        omega_h=sol.omega_h,
        Psi=sol.Psi,
        F=sol.F,
        u_opt=sol.u_opt,
    )

    if args.do_plots:
        pl = Plot()
        dir_path = get_data_path(args.potential_name, args.beta, args.target_set)
        pl.dir_path = dir_path

        # plot optimal performance function (i.e free energy) from the reference solution 
        X = sol.omega_h
        F = sol.F
        pl.file_name = 'reference_solution_free_energy'
        pl.free_energy(X, F)

        # plot optimal tilted potential
        V = potential(X)
        Vb = 2 * F
        pl.file_name = 'reference_solution_optimal_tilted_potential'
        pl.potential_and_tilted_potential(X, V, Vb)

        # plot optimal tilted gradient from reference solution
        dV = gradient(X)
        u = sol.u_opt
        dVb = - np.sqrt(2) * u
        pl.file_name = 'reference_solution_optimal_tilted_drift'
        pl.drift_and_tilted_drift(X, dV, dVb)

        # plot optimal control
        pl.file_name = 'reference_solution_optimal_control'
        pl.control(X, u)

        #pl.file_name = 'reference_solution_optimal_tilted_potential'
        #pl.tilted_potential_and_gradient(X, V, dV, Vb, dVb)

if __name__ == "__main__":
    main()
