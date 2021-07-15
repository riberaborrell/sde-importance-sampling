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
        model = FeedForwardNN(d_layers, args.activation_type)
    else:
        model = DenseNN(d_layers, args.activation_type)

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
        loss, tilted_loss = sample_loss(model, args.dt, args.N)
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        tilted_loss.backward()

        # update parameters
        optimizer.step()

        # save parameters
        thetas[update] = model.get_parameters()

    save_nn_coefficients(thetas)

def save_nn_coefficients(thetas):
    from mds.utils import make_dir_path
    import os
    dir_path = 'data/testing_1d_sgd_ipa_nn'
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

    loss = np.zeros(N)
    tilted_loss = torch.zeros(N)

    # initialize trajectory
    xt = np.full(N, -1.).reshape(N, 1)
    a_tensor = torch.zeros(N)
    b_tensor = torch.zeros(N)
    c_tensor = torch.zeros(N)

    been_in_target_set = np.repeat([False], N).reshape(N, 1)

    for k in np.arange(1, k_max + 1):

        # Brownian increment
        dB = np.sqrt(dt) * np.random.normal(0, 1, N).reshape(N, 1)
        dB_tensor = torch.tensor(dB, requires_grad=False, dtype=torch.float32)

        # control
        xt_tensor = torch.tensor(xt, dtype=torch.float)
        ut_tensor = model.forward(xt_tensor)
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
            idx_tensor = torch.tensor(idx, dtype=torch.long)

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
