from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_hjb_solver import Solver
from mds.plots_1d import Plot1d
from mds.utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Plot the potential landscape'
    parser.add_argument(
        '--betas',
        dest='betas',
        type=float,
        nargs='+',
        default=[1.0],
        help='Set list of betas for the 1D MD SDE. Default: [1.0]',
    )
    return parser

def main():
    args = get_parser().parse_args()
    alpha = np.array(args.alpha)
    betas = np.array(args.betas)

    def f(x):
        return 1

    def g(x):
        return 0

    # get solver for the first beta
    sol = Solver(
        f=f,
        g=g,
        potential_name=args.potential_name,
        alpha=alpha,
        beta=betas[0],
        target_set=np.array(args.target_set),
    )

    # get grid, potential and gradient
    ref_sol = np.load(os.path.join(sol.dir_path, 'reference_solution.npz'))
    X = ref_sol['domain_h']
    V = sol.potential(X)
    dV = sol.gradient(X)

    # initialize F and u_opt
    F = np.zeros((betas.shape[0], X.shape[0]))
    u_opt = np.zeros((betas.shape[0], X.shape[0]))

    # get F and u_opt for each beta
    for i, beta in enumerate(betas):
        sol = Solver(
            f=f,
            g=g,
            potential_name=args.potential_name,
            alpha=alpha,
            beta=beta,
            target_set=np.array(args.target_set),
        )
        ref_sol = np.load(os.path.join(sol.dir_path, 'reference_solution.npz'))
        F[i, :] = ref_sol['F']
        u_opt[i, :] = ref_sol['u_opt']

    # get path
    dir_path = get_example_data_path(args.potential_name)

    # plot free energy
    file_name = 'free_energy_wrt_betas'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_ylim(bottom=0, top=alpha[0] * 3)
    plt1d.free_energy_wrt_betas(X, betas, F)

    # plot tilted potential
    file_name = 'optimal_tilted_potential_wrt_betas'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_ylim(bottom=0, top=alpha[0] * 10)
    plt1d.tilted_potential_wrt_betas(X, V, betas, 2 * F)

    # plot tilted drift 
    file_name = 'optimal_tilted_drift_wrt_betas'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_ylim(bottom=-alpha[0] * 5, top=alpha[0] * 5)
    plt1d.tilted_drift_wrt_betas(X, dV, betas, -np.sqrt(2) * u_opt)

    # plot control
    file_name = 'optimal_control_wrt_betas'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_ylim(bottom=-alpha[0] * 5, top=alpha[0] * 5)
    plt1d.control_wrt_betas(X, betas, u_opt)


if __name__ == "__main__":
    main()
