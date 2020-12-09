from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_hjb_solver import Solver
from mds.potentials_and_gradients_1d import get_potential_and_gradient
from mds.plots_1d import Plot1d
from mds.utils import get_example_data_path

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Plot the potential landscape'
    parser.add_argument(
        '--alphas',
        dest='alphas',
        nargs='+',
        type=float,
        default=[1],
        help='Set the parameter alpha for the chosen setting. Default: [1]',
    )
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

    # get parameters
    potential_name = args.potential_name
    alphas = np.array(args.alphas)
    betas = np.array(args.betas)
    num_plots = betas.shape[0]

    assert alphas.shape[0] % num_plots == 0, ''
    alpha_dim = alphas.shape[0] // num_plots
    alphas = alphas.reshape((num_plots, alpha_dim))

    def f(x):
        return 1

    def g(x):
        return 0

    labels = []

    for i in range(num_plots):
        # make labels 
        potential, \
        gradient, \
        pot_formula, \
        grad_formula, \
        pot_parameters = get_potential_and_gradient(potential_name, alphas[i])
        labels.append(pot_parameters +  r'$, \ \beta = {:2.1f}$'.format(betas[i]))

        # get solver
        sol = Solver(
            f=f,
            g=g,
            potential_name=args.potential_name,
            alpha=alphas[i],
            beta=betas[i],
            target_set=np.array(args.target_set),
            h=args.h,
        )
        # get solution
        h_ext = '_h{:.0e}'.format(args.h)
        file_name = 'reference_solution' + h_ext + '.npz'

        ref_sol = np.load(os.path.join(sol.dir_path, file_name))
        if i == 0:
            x = ref_sol['domain_h']
            V = sol.potential(x)
            dV = sol.gradient(x)
            F = ref_sol['F']
            u_opt = ref_sol['u_opt']
        else:
            x = ref_sol['domain_h']
            V = np.vstack((V, sol.potential(x)))
            dV = np.vstack((dV, sol.gradient(x)))
            F = np.vstack((F, ref_sol['F']))
            u_opt = np.vstack((u_opt, ref_sol['u_opt']))

    # get path
    dir_path = get_example_data_path(args.potential_name)

    # plot free energy
    file_name = 'free_energy_wrt_parameters'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_xlim(-3, 3)
    plt1d.set_ylim(0, 9)
    plt1d.multiple_lines_plot(x, F, labels)

    # plot tilted potential
    file_name = 'optimal_tilted_potential_wrt_parameters'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_xlim(-3, 3)
    plt1d.set_ylim(0, 30)
    plt1d.multiple_lines_plot(x, V + 2*F, labels)

    # plot control
    file_name = 'optimal_control_wrt_parameters'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_xlim(-3, 3)
    plt1d.set_ylim(-1, 20)
    plt1d.multiple_lines_plot(x, u_opt, labels)

    # plot tilted drift 
    file_name = 'optimal_tilted_drift_wrt_parameters'
    plt1d = Plot1d(dir_path, file_name)
    plt1d.set_xlim(-3, 3)
    plt1d.set_ylim(-10, 30)
    plt1d.multiple_lines_plot(x, -dV + np.sqrt(2) * u_opt, labels)


if __name__ == "__main__":
    main()
