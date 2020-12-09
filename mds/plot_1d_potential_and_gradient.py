from mds.base_parser_1d import get_base_parser
from mds.potentials_and_gradients_1d import get_potential_and_gradient
from mds.plots_1d import Plot1d
from mds.utils import get_example_data_path

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Plot potential landscape and gradient'
    parser.add_argument(
        '--pot-y-lim',
        dest='pot_y_lim',
        nargs=2,
        type=float,
        default=[0, 10],
        help='Set y limits for the potential plot. Default: [0, 10]',
    )
    parser.add_argument(
        '--grad-y-lim',
        dest='grad_y_lim',
        nargs=2,
        type=float,
        default=[-5, 5],
        help='Set y limits for the potential ploti. Default: [-5, 5]',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # get parameters
    alpha = np.array(args.alpha)
    pot_ymin, pot_ymax = args.pot_y_lim
    grad_ymin, grad_ymax = args.grad_y_lim

    # plot potentials
    x = np.linspace(-3, 3, 1000)

    # compute potential
    potential, \
    gradient, \
    pot_formula, \
    grad_formula, \
    parameters = get_potential_and_gradient(args.potential_name, alpha)
    V = potential(x)
    dV = gradient(x)

    # get plot path
    dir_path = get_example_data_path(args.potential_name, alpha)

    # plot potential
    plt1d = Plot1d(dir_path, 'potential')
    plt1d.set_ylim(pot_ymin, pot_ymax)
    plt1d.one_line_plot(x, V, label=parameters)

    # plot gradient
    plt1d = Plot1d(dir_path, 'gradient')
    plt1d.set_ylim(grad_ymin, grad_ymax)
    plt1d.one_line_plot(x, dV, label=parameters)

if __name__ == "__main__":
    main()
