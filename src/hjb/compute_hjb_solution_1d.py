import numpy as np

from hjb.hjb_solver_1d import SolverHJB1d
from sde.langevin_sde import LangevinSDE
from utils.base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the 1d HJB equation associated to' \
                         'the overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # set dimension
    d = 1

    # initialize hjb solver
    sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name='nd_2well',
        d=d,
        alpha=np.full(d, args.alpha_i),
        beta=args.beta,
    )

    # initialize hjb solver
    sol_hjb = SolverHJB1d(sde, h=args.h_hjb)

    # compute soltuion
    if not args.load:

        # discretize domain
        sol_hjb.start_timer()
        sol_hjb.sde.discretize_domain()

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
        sol_hjb.write_report(x=np.full(d, args.xzero_i))

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
