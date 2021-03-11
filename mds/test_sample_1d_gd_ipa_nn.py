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

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        #loss, tilted_loss = sample_loss_vect(model, args.N_gd)
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


def sample_loss_vect(model, N):
    beta = 1

    dt = 0.001
    k_max = 100000

    loss = np.zeros(N)
    tilted_loss = torch.empty(0)

    # initialize trajectory
    xt = np.full(N, -1.).reshape(N, 1)
    a = torch.zeros((N, 1))
    b = torch.zeros((N, 1))
    c = torch.zeros((N, 1))
    been_in_target_set = np.repeat([False], N).reshape(N, 1)


    for k in np.arange(1, k_max + 1):

        # Brownian increment
        dB = np.sqrt(dt) * np.random.normal(0, 1, N).reshape(N, 1)
        dB_tensor = torch.tensor(dB, requires_grad=False)

        # control
        xt_tensor = torch.tensor(xt, dtype=torch.float)
        ut_tensor = model.forward(xt_tensor)
        ut_tensor_det = model.forward(xt_tensor).detach()
        ut = model.forward(xt_tensor).detach().numpy()

        # sde update
        drift = (- double_well_1d_gradient(xt) + np.sqrt(2) * ut) * dt
        diffusion = np.sqrt(2 / beta) * dB
        xt += drift + diffusion

        # update statistics
        loss += (0.5 * (ut ** 2) * dt).reshape(N,)

        a = a + ut_tensor_det * ut_tensor * dt
        b = b + 0.5 * (ut_tensor_det ** 2) * dt
        c = c - np.sqrt(beta) * dB_tensor * ut_tensor

        # update statistics for trajectories which arrived
        idx_new = get_idx_new_in_ts(xt, been_in_target_set)

        loss[idx_new] += k * dt
        for idx in idx_new:
            b[idx] = b[idx] + k * dt
            tilted_loss = torch.cat((tilted_loss, a[idx] - b[idx] * c[idx]))

        # stop if xt_traj in target set
        if been_in_target_set.all() == True:
           break

    return np.mean(loss), torch.mean(tilted_loss)

def get_idx_new_in_ts(x, been_in_target_set):
    is_in_target_set = x > 1

    idx_new = np.where(
            (is_in_target_set == True) &
            (been_in_target_set == False)
    )[0]

    been_in_target_set[idx_new] = True

    return idx_new

if __name__ == "__main__":
    main()
