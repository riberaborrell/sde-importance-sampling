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

    # initialize control parametrization by a nn 
    d_in, d_1, d_out = args.n, args.hidden_layer_dim, args.n
    model = TwoLayerNet(d_in, d_1, d_out)

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # preallocate parameters
    thetas = np.empty((args.updates_lim, model.d_flatten))

    # save initial parameters
    thetas[0] = model.get_flatten_parameters()
    print(thetas[0])

    for update in np.arange(args.updates_lim):
        # reset gradients
        optimizer.zero_grad()

        # compute loss
        loss, tilted_loss = sample_loss(model, args.N_gd)
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        tilted_loss.backward()

        # update parameters
        optimizer.step()

        # save parameters
        thetas[update] = model.get_flatten_parameters()
        print(thetas[update])

    save_nn_coefficients(thetas)

def save_nn_coefficients(thetas):
    from mds.utils import make_dir_path
    import os
    dir_path = 'mds/data/testing_gd_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'gd.npz')
    np.savez(
        file_path,
        thetas=thetas,
    )

def double_well_1d_gradient(x):
    alpha = 1
    return 4 * alpha * x * (x**2 - 1)

def sample_loss(model, N):
    beta = 1

    dt = 0.001
    k_max = 100000

    loss = np.zeros(N)
    tilted_loss = torch.empty(0)

    for i in np.arange(N):

        # initialize trajectory
        xt_traj = -1
        a_traj = torch.zeros(1)
        b_traj = torch.zeros(1)
        c_traj = torch.zeros(1)

        for k in np.arange(1, k_max + 1):

            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)

            # control
            xt_traj_tensor = torch.tensor([xt_traj], dtype=torch.float)
            ut_traj_tensor = model.forward(xt_traj_tensor)
            ut_traj_tensor_det = model.forward(xt_traj_tensor).detach()
            ut_traj = model.forward(xt_traj_tensor).detach().numpy()[0]

            # sde update
            drift = (- double_well_1d_gradient(xt_traj) + np.sqrt(2) * ut_traj) * dt
            diffusion = np.sqrt(2 / beta) * dB
            xt_traj += drift + diffusion

            # update statistics
            loss[i] += 0.5 * (ut_traj ** 2) * dt

            a_traj = a_traj + ut_traj_tensor_det * ut_traj_tensor * dt
            b_traj = b_traj + 0.5 * (ut_traj_tensor_det ** 2) * dt
            c_traj = c_traj - np.sqrt(beta) * dB * ut_traj_tensor

            # stop if xt_traj in target set
            if xt_traj >= 1:

                # update statistics
                loss[i] += k * dt

                b_traj = b_traj + k * dt
                tilted_loss_traj = a_traj - b_traj * c_traj

                # save tilted loss for the given trajectory
                tilted_loss = torch.cat((tilted_loss, tilted_loss_traj))
                break

    return np.mean(loss), torch.mean(tilted_loss)


if __name__ == "__main__":
    main()
