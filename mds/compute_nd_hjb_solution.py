from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_hjb_solver import Solver

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE'
    parser.add_argument(
        '--load',
        dest='load',
        action='store_true',
        help='Load already computed hjb solution. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize hjb solver
    sol = Solver(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h_hjb,
    )

    # compute soltuion
    if not args.load:

        # discretize domain
        sol.start_timer()
        sol.discretize_domain()

        # compute hjb solution 
        sol.solve_bvp()
        sol.compute_free_energy()
        sol.compute_optimal_control()
        sol.stop_timer()

        # save solution
        sol.save_hjb_solution()

    # load already computed solution
    else:
        if not sol.load_hjb_solution():
            return

    # report solution
    if args.do_report:
        sol.write_report(x=np.full(args.n, args.xzero_i))

    # print solution
    if args.do_plots:

        # evaluate in grid
        sol.get_controlled_potential_and_drift()

        if sol.n == 1:
            sol.plot_1d_psi(sol.Psi, label='num sol HJB PDE')
            sol.plot_1d_free_energy(sol.F, label='num sol HJB PDE')
            sol.plot_1d_controlled_potential(sol.controlled_potential, label='num sol HJB PDE')
            sol.plot_1d_control(sol.u_opt[:, 0], label='num sol HJB PDE')
            sol.plot_1d_controlled_drift(sol.controlled_drift[:, 0], label='num sol HJB PDE')

        elif sol.n == 2:
            sol.plot_2d_psi(sol.Psi)
            sol.plot_2d_free_energy(sol.F)
            sol.plot_2d_controlled_potential(sol.controlled_potential)
            sol.plot_2d_control(sol.u_opt)
            sol.plot_2d_controlled_drift(sol.controlled_drift)
        else:
            pass

if __name__ == "__main__":
    main()
