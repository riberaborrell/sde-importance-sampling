from mds.base_parser_nd import get_base_parser

from mds.neural_networks import TwoLayerNet

import numpy as np

import torch
import torch.optim as optim

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    parser.add_argument(
        '--updates-lim',
        dest='updates_lim',
        type=int,
        default=100,
        help='Set maximal number of updates. Default: 100',
    )
    parser.add_argument(
        '--hidden-layer-dim',
        dest='hidden_layer_dim',
        type=int,
        default=10,
        help='Set dimension of the hidden layer. Default: 10',
    )
    return parser

def main():
    args = get_parser().parse_args()
    assert args.n == 1, ''

    # initialize control parametrization by a nn 
    d_in, d_1, d_out = args.n, args.hidden_layer_dim, args.n
    model = TwoLayerNet(d_in, d_1, d_out)

    # load nn coeficients
    thetas = load_nn_coefficients()

    # define discretize domain
    h = 0.01
    x = np.arange(-3, 3 + h, h)
    N = x.shape[0]

    # load flatten parameters in the network
    model.load_parameters(thetas[-1])

    # evaluate control in x
    input = torch.tensor(x.reshape(N, d_in), dtype=torch.float)
    control = model(input).detach().numpy().reshape(N,)

    # plot
    plot_1d_control(x, control)

def load_nn_coefficients():
    from mds.utils import make_dir_path
    import os
    dir_path = 'mds/data/testing_1d_gd_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'gd.npz')
    gd = np.load(
        file_path,
        allow_pickle=True,
    )
    return gd['thetas']

def plot_1d_control(x, control):
    from mds.plots import Plot

    dir_path = 'mds/data/testing_1d_gd_nn'
    plt = Plot(dir_path, 'control')
    plt.xlabel = 'x'
    plt.set_ylim(- 5, 5)
    plt.one_line_plot(x, control)

if __name__ == "__main__":
    main()
