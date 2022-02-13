from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.hjb_solver_1d import SolverHJB1d

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the 1d HJB equation associated to' \
                         'the overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # set dimension
    n = 1

    # initialize hjb solver
    sol_hjb = SolverHJB1d(
        problem_name='langevin_stop-t',
        potential_name='nd_2well',
        n=n,
        alpha=np.full(n, args.alpha_i),
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
        sol_hjb.compute_value_function()
        sol_hjb.compute_optimal_control()

        #TODO! debut
        #sol_hjb.compute_exp_fht()

        sol_hjb.stop_timer()
        sol_hjb.save()

    # load already computed solution
    else:
        if not sol_hjb.load():
            return

    # report solution
    if args.do_report:
        sol_hjb.write_report(x=np.full(n, args.xzero_i))

        import matplotlib.pyplot as plt
        x = sol_hjb.domain_h[:, 0]
        y = sol_hjb.u_opt[:, 0]
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(-2, 2)
        plt.plot(x, y)
        plt.grid()
        plt.show()

    if args.do_plots:

        # evaluate in grid
        sol_hjb.get_controlled_potential_and_drift()

        sol_hjb.plot_1d_psi()
        sol_hjb.plot_1d_value_function()
        sol_hjb.plot_1d_controlled_potential()
        sol_hjb.plot_1d_control()
        sol_hjb.plot_1d_controlled_drift()

        #TODO! debug
        #sol_hjb.plot_exp_fht()

if __name__ == "__main__":
    main()
