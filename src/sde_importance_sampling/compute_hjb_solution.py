from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.hjb_solver import SolverHJB

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE'
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

    # set target set array
    if args.potential_name == 'nd_2well':
        target_set = np.full((args.n, 2), [1, 3])
    elif args.potential_name == 'nd_2well_asym':
        target_set = np.empty((args.n, 2))
        target_set[0] = [1, 3]
        target_set[1:] = [-3, 3]

    # initialize hjb solver
    sol_hjb = SolverHJB(
        problem_name='langevin_stop-t',
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
        h=args.h_hjb,
    )

    # compute soltuion
    if not args.load:

        # discretize domain
        sol_hjb.start_timer()
        sol_hjb.discretize_domain()

        # compute hjb solution 
        sol_hjb.solve_bvp()
        sol_hjb.compute_value_function()
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

        # 1d
        if sol_hjb.n == 1:
            sol_hjb.plot_1d_psi()
            sol_hjb.plot_1d_value_function()
            sol_hjb.plot_1d_controlled_potential()
            sol_hjb.plot_1d_control()
            sol_hjb.plot_1d_controlled_drift()

        # 2d
        elif sol_hjb.n == 2:
            sol_hjb.plot_2d_psi()
            sol_hjb.plot_2d_value_function()
            sol_hjb.plot_2d_controlled_potential()
            sol_hjb.plot_2d_control()
            sol_hjb.plot_2d_controlled_drift()
        else:
            pass

if __name__ == "__main__":
    main()