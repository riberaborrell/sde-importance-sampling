from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_hjb_solver import Solver

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Computes the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE'
    parser.add_argument(
        '--load-hjb-sol',
        dest='load_hjb_sol',
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
    if not args.load_hjb_sol:

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
        sol.load_hjb_solution()

    # report solution
    if args.do_report:
        sol.write_report(x=np.full(args.n, args.xzero_i))

    # print solution
    if args.do_plots:

        # flatten domain_h
        x = sol.domain_h.reshape(sol.Nh, sol.n)

        # psi, free energy
        psi = sol.Psi
        free = sol.F

        # potential, bias potential and tilted potential
        V = sol.potential(x).reshape(sol.Nx)
        Vbias = 2 * sol.F
        Vtilted = V + Vbias

        # gradient, control and tilted drift
        dV = sol.gradient(x).reshape(sol.domain_h.shape)
        u_opt = sol.u_opt
        dVtilted = - dV + np.sqrt(2) * sol.u_opt

        if sol.n == 1:
            pass
        elif sol.n == 2:
            sol.plot_2d_psi(psi)
            sol.plot_2d_free_energy(free)
            sol.plot_2d_tilted_potential(Vtilted)
            sol.plot_2d_control(u_opt)
            sol.plot_2d_tilted_drift(dVtilted)

if __name__ == "__main__":
    main()
