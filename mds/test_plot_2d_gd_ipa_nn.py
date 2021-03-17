from mds.base_parser_nd import get_base_parser

from mds.neural_networks import TwoLayerNet

import numpy as np

import torch

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
    assert args.n == 2, ''

    # initialize control parametrization by a nn 
    d_in, d_1, d_out = args.n, args.hidden_layer_dim, args.n
    model = TwoLayerNet(d_in, d_1, d_out)

    # load nn coeficients
    thetas = load_nn_coefficients()

    # define discretize domain
    h = 0.01
    h = 0.01
    mgrid_input = [
        slice(-3, 3 + h, h),
        slice(-3, 3 + h, h),
    ]
    domain_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)
    Nx = domain_h.shape[0]
    Ny = domain_h.shape[1]
    Nh = Nx * Ny
    domain_h_flattened = domain_h.reshape(Nh, args.n)

    # load flattened parameters in the network
    model.load_parameters(thetas[-1])

    # evaluate control in flattened discretized domain
    input = torch.tensor(domain_h_flattened, dtype=torch.float)
    control_flattened = model(input).detach().numpy()
    control = control_flattened.reshape(Nx, Ny, args.n)

    # plot
    plot_2d_control(domain_h, control)

def load_nn_coefficients():
    from mds.utils import make_dir_path
    import os
    dir_path = 'mds/data/testing_nd_gd_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'gd.npz')
    gd = np.load(
        file_path,
        allow_pickle=True,
    )
    return gd['thetas']

def plot_2d_control(domain_h, control):
    from mds.plots import Plot

    X = domain_h[:, :, 0]
    Y = domain_h[:, :, 1]
    U = control[:, :, 0]
    V = control[:, :, 1]

    dir_path = 'mds/data/testing_nd_gd_nn'
    plt = Plot(dir_path, 'control')
    plt.vector_field(X, Y, U, V, scale=8)


if __name__ == "__main__":
    main()
