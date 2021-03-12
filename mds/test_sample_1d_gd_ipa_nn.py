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

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize control parametrization by a nn 
    d_in, d_1, d_out = args.n, args.hidden_layer_dim, args.n
    model = TwoLayerNet(d_in, d_1, d_out).to(device)

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
        #loss, tilted_loss = sample_loss(model, args.N_gd, device)
        loss, tilted_loss = sample_loss_vect(model, args.N_gd, device)
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        tilted_loss.backward(retain_graph=True)

        # update parameters
        optimizer.step()

        # save parameters
        thetas[update] = model.get_flatten_parameters()
        print(thetas[update])

    save_nn_coefficients(thetas)

def save_nn_coefficients(thetas):
    from mds.utils import make_dir_path
    import os
    dir_path = 'mds/data/testing_1d_gd_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'gd.npz')
    np.savez(
        file_path,
        thetas=thetas,
    )

def double_well_1d_gradient(x):
    alpha = 1
    return 4 * alpha * x * (x**2 - 1)

def sample_loss(model, N, device):
    beta = 1

    dt = 0.001
    k_max = 100000

    loss = np.zeros(N)
    tilted_loss = torch.empty(0).to(device)

    for i in np.arange(N):

        # initialize trajectory
        xt_traj = -1.
        a_traj_tensor = torch.zeros(1).to(device)
        b_traj_tensor = torch.zeros(1).to(device)
        c_traj_tensor = torch.zeros(1).to(device)

        for k in np.arange(1, k_max + 1):

            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)

            # control
            xt_traj_tensor = torch.tensor([xt_traj], dtype=torch.float).to(device)
            ut_traj_tensor = model.forward(xt_traj_tensor).to(device)
            ut_traj_tensor_det = ut_traj_tensor.detach()
            ut_traj = ut_traj_tensor_det.numpy()[0]

            # sde update
            drift = (- double_well_1d_gradient(xt_traj) + np.sqrt(2) * ut_traj) * dt
            diffusion = np.sqrt(2 / beta) * dB
            xt_traj += drift + diffusion

            # update statistics
            loss[i] += 0.5 * (ut_traj ** 2) * dt

            a_traj_tensor = a_traj_tensor + ut_traj_tensor_det * ut_traj_tensor * dt
            b_traj_tensor = b_traj_tensor + 0.5 * (ut_traj_tensor_det ** 2) * dt
            c_traj_tensor = c_traj_tensor - np.sqrt(beta) * dB * ut_traj_tensor

            # stop if xt_traj in target set
            if xt_traj >= 1:

                # update statistics
                loss[i] += k * dt

                b_traj_tensor = b_traj_tensor + k * dt
                tilted_loss_traj = a_traj_tensor - b_traj_tensor * c_traj_tensor

                # save tilted loss for the given trajectory
                tilted_loss = torch.cat((tilted_loss, tilted_loss_traj))
                break

    return np.mean(loss), torch.mean(tilted_loss)


def sample_loss_vect(model, N, device):
    beta = 1

    dt = 0.001
    k_max = 100000

    loss = np.zeros(N)
    tilted_loss = torch.zeros(N).to(device)

    # initialize trajectory
    xt = np.full(N, -1.).reshape(N, 1)
    a_tensor = torch.zeros(N).to(device)
    b_tensor = torch.zeros(N).to(device)
    c_tensor = torch.zeros(N).to(device)

    been_in_target_set = np.repeat([False], N).reshape(N, 1)

    for k in np.arange(1, k_max + 1):

        # Brownian increment
        dB = np.sqrt(dt) * np.random.normal(0, 1, N).reshape(N, 1)
        dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32).to(device)

        # control
        xt_tensor = torch.tensor(xt, dtype=torch.float).to(device)
        ut_tensor = model.forward(xt_tensor).to(device)
        ut_tensor_det = ut_tensor.detach()
        ut = ut_tensor_det.numpy()

        # sde update
        drift = (- double_well_1d_gradient(xt) + np.sqrt(2) * ut) * dt
        diffusion = np.sqrt(2 / beta) * dB
        xt += drift + diffusion

        # update statistics
        loss += (0.5 * (ut ** 2) * dt).reshape(N,)

        a_tensor = a_tensor + (ut_tensor_det * ut_tensor * dt).reshape(N,)
        b_tensor = b_tensor + ((1 + 0.5 * (ut_tensor_det ** 2)) * dt).reshape(N,)
        c_tensor = c_tensor - (np.sqrt(beta) * dB_tensor * ut_tensor).reshape(N,)

        # get indices of trajectories which are new to the target set
        idx = get_idx_new_in_ts(xt, been_in_target_set)

        # update statistics
        if idx.shape[0] != 0:

            # update loss
            loss[idx] += k * dt

            # update tilted loss
            idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)
            tilted_loss[idx_tensor] = a_tensor.index_select(0, idx_tensor) \
                                    - b_tensor.index_select(0, idx_tensor) \
                                    * c_tensor.index_select(0, idx_tensor)

        # stop if xt_traj in target set
        if been_in_target_set.all() == True:
           break

    return np.mean(loss), torch.mean(tilted_loss)

def get_idx_new_in_ts(x, been_in_target_set):
    is_in_target_set = x > 1

    idx = np.where(
            (is_in_target_set == True) &
            (been_in_target_set == False)
    )[0]

    been_in_target_set[idx] = True

    return idx

if __name__ == "__main__":
    main()
