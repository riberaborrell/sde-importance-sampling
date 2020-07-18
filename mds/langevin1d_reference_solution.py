from reference_solution import langevin_1d_reference_solution

import argparse

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

    # compute reference solution
    sol = langevin_1d_reference_solution(
        potential_name=args.potential_name,
        alpha=args.alpha,
        beta=args.beta,
        target_set=args.target_set,
        h=args.h,
    )
    sol.compute_reference_solution()
    sol.save_reference_solution()

    if args.do_plots:
        sol.plot_free_energy()
        sol.plot_optimal_tilted_potential()
        sol.plot_optimal_tilted_drift()
        sol.plot_optimal_control()

    #TODO approximate the first hitting time

if __name__ == "__main__":
    main()
