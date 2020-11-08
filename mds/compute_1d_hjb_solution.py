from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_hjb_solver import Solver

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
        f=f,
        g=g,
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set),
        domain=np.array(args.domain),
        h=args.h,
    )
    sol.discretize_domain()
    sol.solve_bvp()
    sol.compute_free_energy()
    sol.compute_optimal_control()
    #sol.compute_exp_fht()
    sol.save_reference_solution()
    sol.write_report(x=args.xzero)

    if args.do_plots:
        sol.plot_mgf()
        sol.plot_free_energy()
        sol.plot_optimal_tilted_potential()
        sol.plot_optimal_tilted_drift()
        sol.plot_optimal_control()
        #sol.plot_exp_fht()

if __name__ == "__main__":
    main()
