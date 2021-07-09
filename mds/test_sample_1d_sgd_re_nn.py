from mds.base_parser_nd import get_base_parser

from mds.neural_networks import FeedForwardNN, DenseNN

import numpy as np

import torch
import torch.optim as optim

def get_parser():
    parser = get_base_parser()
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

    # fix seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

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

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # preallocate parameters
    thetas = np.empty((args.iterations_lim, model.d_flat))

    # save initial parameters
    thetas[0] = model.get_parameters()

    for update in np.arange(args.iterations_lim):
        # reset gradients
        optimizer.zero_grad()

        # compute loss
        loss_tensor = sample_loss(model, args.dt, args.N)
        loss = loss_tensor.detach().numpy()
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        loss_tensor.backward()

        # update parameters
        optimizer.step()

        # save parameters
        thetas[update] = model.get_parameters()
        #print(thetas[update])

    save_nn_coefficients(thetas)

def save_nn_coefficients(thetas):
    from mds.utils import make_dir_path
    import os
    dir_path = 'data/testing_1d_sgd_re_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'som.npz')
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

def sample_loss(model, dt, N):
    beta = 1
    k_max = 100000

    # initialize phi
    phi_fht = torch.zeros(N)
    phi_t = torch.zeros(N)

    # initialize trajectory
    xt = np.full(N, -1.).reshape(N, 1)
    #xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32)
    xt_tensor = torch.tensor(xt, dtype=torch.float32)

    # control
    ut_tensor = model.forward(xt_tensor)

    been_in_target_set = np.repeat([False], N).reshape(N, 1)

    for k in np.arange(1, k_max + 1):

        # Brownian increment
        dB = np.sqrt(dt) * np.random.normal(0, 1, N).reshape(N, 1)
        dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32)

        # sde update
        drift = (- double_well_1d_gradient(xt_tensor) + np.sqrt(2) * ut_tensor) * dt
        diffusion = np.sqrt(2 / beta) * dB_tensor
        xt_tensor = xt_tensor + drift + diffusion
        xt = xt_tensor.detach().numpy()

        # update statistics
        phi_t = phi_t + ((1 + 0.5 * (ut_tensor ** 2)) * dt).reshape(N,)

        # control
        ut_tensor = model.forward(xt_tensor)
        #breakpoint()

        # get indices of trajectories which are new to the target set
        idx = get_idx_new_in_ts(xt, been_in_target_set)

        if idx.shape[0] != 0:

            # get tensor indices if there are new trajectories 
            idx_tensor = torch.tensor(idx, dtype=torch.long)

            # save phi for trajectories which arrived
            phi_fht[idx_tensor] = phi_t.index_select(0, idx_tensor)

        # stop if xt_traj in target set
        if been_in_target_set.all() == True:
           break

    return torch.mean(phi_fht)


if __name__ == "__main__":
    main()
