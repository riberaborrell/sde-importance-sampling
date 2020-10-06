from potentials_and_gradients import POTENTIAL_NAMES
from hjb_1d_solver import langevin_hjb_1d_solver

import argparse
import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(
        description='Plot the numerical solution of the HJB equation associated to the'
                    'overdamped Langevin SDE'
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
        help='Set the parameter alpha for the chosen potentials. Default: [1]',
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
    return parser

def main():
    args = get_parser().parse_args()

    # get hjb solver
    sol = langevin_hjb_1d_solver(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set),
    )

    # load already computed solution
    ref_sol = np.load(os.path.join(sol.dir_path, 'reference_solution.npz'))
    sol.domain_h = ref_sol['domain_h']
    sol.Psi = ref_sol['Psi']
    sol.F = ref_sol['F']
    sol.u_opt = ref_sol['u_opt']
    sol.exp_fht = ref_sol['exp_fht']

    # plot
    sol.plot_mgf()
    sol.plot_free_energy()
    sol.plot_optimal_tilted_potential()
    sol.plot_optimal_tilted_drift()
    sol.plot_optimal_control()
    sol.plot_exp_fht()

if __name__ == "__main__":
    main()
