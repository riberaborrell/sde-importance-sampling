from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_hjb_solver import SolverHJB

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize hjb solver
    sol_hjb = SolverHJB(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h_hjb,
    )

    # compute soltuion
    if not args.load:

        # discretize domain
        sol_hjb.start_timer()
        sol_hjb.discretize_domain()

        # compute hjb solution 
        sol_hjb.solve_bvp()
        sol_hjb.compute_free_energy()
        sol_hjb.compute_optimal_control()
        sol_hjb.stop_timer()

        # save solution
        sol_hjb.save()

    # load already computed solution
    else:
        if not sol_hjb.load():
            return

    # report solution
    if args.do_report:
        sol_hjb.write_report(x=np.full(args.n, args.xzero_i))

    # do plots
    if args.do_plots:

        # evaluate in grid
        sol_hjb.get_controlled_potential_and_drift()

        if sol.n == 1:
            sol_hjb.plot_1d_psi(sol.Psi, label='num sol HJB PDE')
            sol_hjb.plot_1d_free_energy(sol.F, label='num sol HJB PDE')
            sol_hjb.plot_1d_controlled_potential(sol.controlled_potential, label='num sol HJB PDE')
            sol_hjb.plot_1d_control(sol.u_opt[:, 0], label='num sol HJB PDE')
            sol_hjb.plot_1d_controlled_drift(sol.controlled_drift[:, 0], label='num sol HJB PDE')

        elif sol.n == 2:
            sol_hjb.plot_2d_psi(sol.Psi)
            sol_hjb.plot_2d_free_energy(sol.F)
            sol_hjb.plot_2d_controlled_potential(sol.controlled_potential)
            sol_hjb.plot_2d_control(sol.u_opt)
            sol_hjb.plot_2d_controlled_drift(sol.controlled_drift)
        else:
            pass

if __name__ == "__main__":
    main()
