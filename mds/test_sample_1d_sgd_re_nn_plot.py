from mds.base_parser_nd import get_base_parser

from mds.neural_networks import FeedForwardNN, DenseNN

import numpy as np

import torch

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    parser.add_argument(
        '--d-layers',
        nargs='+',
        dest='d_layers',
        type=int,
        help='Set dimensions of the NN inner layers',
    )
    parser.add_argument(
        '--dense',
        dest='dense',
        action='store_true',
        help='Chooses a dense feed forward NN. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()
    assert args.n == 1, ''

    # get dimensions of each layer
    if args.d_layers is not None:
        d_layers = [args.n] + args.d_layers + [args.n]
    else:
        d_layers = [args.n, args.n]

    # initialize nn model 
    if not args.dense:
        model = FeedForwardNN(d_layers)
    else:
        model = DenseNN(d_layers)

    # load nn coeficients
    thetas = load_nn_coefficients()

    # define discretize domain
    h = 0.01
    x = np.arange(-3, 3 + h, h)
    N = x.shape[0]

    # load flatten parameters in the network
    model.load_parameters(thetas[-1])

    # evaluate control in x
    input = torch.tensor(x.reshape(N, args.n), dtype=torch.float)
    control = model(input).detach().numpy().reshape(N,)

    # plot
    plot_1d_control(x, control)

def load_nn_coefficients():
    from mds.utils import make_dir_path
    import os
    dir_path = 'data/testing_1d_sgd_re_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'som.npz')
    data = np.load(
        file_path,
        allow_pickle=True,
    )
    return data['thetas']

def plot_1d_control(x, control):
    from mds.plots import Plot

    dir_path = 'data/testing_1d_sgd_re_nn'
    plt = Plot(dir_path, 'control')
    plt.xlabel = 'x'
    plt.set_ylim(- 5, 5)
    plt.one_line_plot(x, control)

if __name__ == "__main__":
    main()
