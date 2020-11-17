from mds.base_parser_1d import get_base_parser
from mds.potentials_and_gradients_1d import get_potential_and_gradient
from mds.plots_1d import Plot1d
from mds.utils import get_example_data_path

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Plot potential landscape and gradient'
    parser.add_argument(
        '--alphas',
        dest='alphas',
        nargs='+',
        type=float,
        default=[1],
        help='Set the parameter alpha for the chosen setting. Default: [1]',
    )
    parser.add_argument(
        '--num-plots',
        dest='num_plots',
        type=int,
        default=1,
        help='Set number of plots',
    )
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
    potential_name = args.potential_name
    alphas = np.array(args.alphas)
    num_plots = args.num_plots
    pot_ymin, pot_ymax = args.pot_y_lim
    grad_ymin, grad_ymax = args.grad_y_lim

    assert alphas.shape[0] % num_plots == 0, ''

    # plot potentials
    x = np.linspace(-3, 3, 1000)

    Vs = np.empty((num_plots, x.shape[0]))
    dVs = np.empty((num_plots, x.shape[0]))
    alpha_dim = int(alphas.shape[0] / num_plots)
    alphas = alphas.reshape((num_plots, alpha_dim))
    labels = []
    for i in range(num_plots):
        potential, \
        gradient, \
        pot_formula, \
        grad_formula, \
        parameters = get_potential_and_gradient(potential_name, alphas[i])
        labels.append(parameters)
        Vs[i] = potential(x)
        dVs[i] = gradient(x)

    # get plot path
    dir_path = get_example_data_path(potential_name)

    # plot potentials
    plt1d = Plot1d(dir_path, 'potentials')
    plt1d.set_ylim(pot_ymin, pot_ymax)
    #plt1d.set_title(pot_formula)
    plt1d.potentials_or_gradients(x, Vs, labels)

    # plot gradients
    plt1d = Plot1d(dir_path, 'gradients')
    plt1d.set_ylim(grad_ymin, grad_ymax)
    #plt1d.set_title(pot_formula)
    plt1d.potentials_or_gradients(x, dVs, labels)

if __name__ == "__main__":
    main()
