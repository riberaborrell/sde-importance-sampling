from mds.base_parser_nd import get_base_parser
from mds.test_sample_sgd_nn import load_som, plot_1d_control
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
    parser.add_argument(
        '--activation',
        dest='activation_type',
        choices=['relu', 'tanh'],
        default='relu',
        help='Type of activation function. Default: relu',
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
        model = FeedForwardNN(d_layers, args.activation_type)
    else:
        model = DenseNN(d_layers, args.activation_type)

    # set dir path
    dir_path = 'data/testing_1d_sgd_nn/ipa'
    #dir_path = 'data/testing_1d_sgd_re_nn'
    #dir_path = 'data/testing_1d_bf_logvar_nn'

    # load nn coeficients
    data = load_som(dir_path)
    breakpoint()
    thetas = data['thetas']

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
    plot_1d_control(x, control, dir_path)


if __name__ == "__main__":
    main()
