from mds.base_parser_nd import get_base_parser
from mds.langevin_det_hjb_solver import SolverHJBDet

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE with deterministic time horizont'
    return parser

def main():
    args = get_parser().parse_args()

    # set alpha array
    if args.potential_name == 'nd_2well':
        alpha = np.full(args.n, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.n)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # set nu array
    if args.potential_name == 'nd_2well':
        nu = np.full(args.n, args.nu_i)
    elif args.potential_name == 'nd_2well_asym':
        nu = np.empty(args.n)
        nu[0] = args.nu_i
        nu[1:] = args.nu_j

    # initialize hjb solver
    sol_hjb = SolverHJBDet(
        problem_name='langevin_det-t',
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        T=args.T,
        nu=nu,
        h=args.h_hjb,
        dt=args.dt_hjb,
    )

    # compute solution
    if not args.load:

        # discretize domain
        sol_hjb.start_timer()
        sol_hjb.discretize_domain()
        sol_hjb.preallocate_psi_i()
        sol_hjb.preallocate_u_opt_i()

        # compute hjb solution 
        #sol_hjb.solve_bvp()
        sol_hjb.solve_bvp_eigenproblem()
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
        sol_hjb.write_report(
            t=0.,
            x=np.full(args.n, args.xzero_i),
        )

    # do plots
    if args.do_plots:

        sol_hjb.plot_1d_psi_i()
        #sol_hjb.plot_1d_value_function()
        #sol_hjb.plot_1d_controlled_potential()
        #sol_hjb.plot_1d_control()
        #sol_hjb.plot_1d_controlled_drift()

if __name__ == "__main__":
    main()
