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
        default=30,
        help='Set dimension of the hidden layer. Default: 30',
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
    #print(thetas[0])

    for update in np.arange(args.updates_lim):
        # reset gradients
        optimizer.zero_grad()

        # compute loss
        loss, tilted_loss = sample_loss(device, model, args.dt, args.N_gd)
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        tilted_loss.backward()

        # update parameters
        optimizer.step()

        # save parameters
        thetas[update] = model.get_flatten_parameters()
        #print(thetas[update])

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

def get_idx_new_in_ts(x, been_in_target_set):
    is_in_target_set = x > 1

    idx = np.where(
            (is_in_target_set == True) &
            (been_in_target_set == False)
    )[0]

    been_in_target_set[idx] = True

    return idx

def sample_loss(device, model, dt, N):
    beta = 1

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
        a_tensor = a_tensor + (ut_tensor_det * ut_tensor * dt).reshape(N,)
        b_tensor = b_tensor + ((1 + 0.5 * (ut_tensor_det ** 2)) * dt).reshape(N,)
        c_tensor = c_tensor - (np.sqrt(beta) * dB_tensor * ut_tensor).reshape(N,)

        # get indices of trajectories which are new to the target set
        idx = get_idx_new_in_ts(xt, been_in_target_set)

        if idx.shape[0] != 0:

            # get tensor indices if there are new trajectories 
            idx_tensor = torch.tensor(idx, dtype=torch.long).to(device)

            # save loss and tilted loss for the arrived trajectorries
            loss[idx] = b_tensor.numpy()[idx]
            tilted_loss[idx_tensor] = a_tensor.index_select(0, idx_tensor) \
                                    - b_tensor.index_select(0, idx_tensor) \
                                    * c_tensor.index_select(0, idx_tensor)

        # stop if xt_traj in target set
        if been_in_target_set.all() == True:
           break

    return np.mean(loss), torch.mean(tilted_loss)


if __name__ == "__main__":
    main()
