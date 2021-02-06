from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_hjb_solver import Solver

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    def f(x):
        return 1

    def g(x):
        return 0

    # compute reference solution
    sol = Solver(
        n=args.n,
        f=f,
        g=g,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h,
    )
    sol.start_timer()
    sol.discretize_domain()
    sol.solve_bvp()
    sol.compute_free_energy()
    sol.compute_optimal_control()
    sol.stop_timer()
    sol.save_hjb_solution()
    sol.write_report(x=np.full(args.n, args.xzero_i))

    if args.do_plots:
        sol.plot_psi()
        sol.plot_free_energy()
        sol.plot_optimal_tilted_potential()
        sol.plot_optimal_control()
        sol.plot_optimal_tilted_drift()


if __name__ == "__main__":
    main()
