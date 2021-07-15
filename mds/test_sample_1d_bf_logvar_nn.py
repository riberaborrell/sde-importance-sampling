from mds.base_parser_nd import get_base_parser

from mds.neural_networks import FeedForwardNN, DenseNN

import numpy as np

import torch
import torch.optim as optim

import time

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


def sample_loss(model, dt, N):
    beta = 1
    k_max = 100000

    # flags
    adaptive_forward_process = True
    detach_forward = True

    t_0 = time.time()

    # time increments as tensors
    dt = torch.tensor([dt])
    sq_dt = torch.sqrt(dt)

    # initialize trajectories of processes X and Y 
    xt = - torch.ones(N).reshape(N, 1)
    yt = torch.zeros(N).reshape(N)

    # initialize Z sum
    Z_sum = torch.zeros(N)

    # number of trajectories not in the target set
    N_not_in_ts = N

    for k in np.arange(1, k_max + 1):

        # stop trajetories if all trajectories are in the target set
        idx = xt[:, 0] < 1
        N_not_in_ts = torch.sum(idx)
        if N_not_in_ts == 0:
            break

        # normal distributed vector
        xi = torch.randn(N_not_in_ts).reshape(N_not_in_ts, 1)

        # Brownian increment
        dB = np.sqrt(dt) * xi

        # Z process
        zt = - model.forward(xt[idx, :])

        # control
        ut = torch.zeros(N_not_in_ts).reshape(N_not_in_ts, 1)
        if adaptive_forward_process is True:
            ut = - zt

        if detach_forward is True:
            ut = ut.detach()

        # step dynamics process X forward
        drift = (- double_well_1d_gradient(xt[idx, :]) + np.sqrt(2) * ut) * dt
        diffusion = np.sqrt(2 / beta) * dB
        xt[idx, :] = xt[idx, :] + drift + diffusion

        # step dynamics process Y
        h = (-1 + 0.5 * zt ** 2).reshape(N_not_in_ts,)
        yt[idx] = yt[idx] \
                + (h + torch.sum(zt * ut, dim=1)) * dt \
                + torch.sum(zt * xi, dim=1) * sq_dt

        # update Z_sum
        #Z_sum[selection] += dw.f(X[selection, :], n * delta_t) * delta_t # (0.5 * pt.sum(Z**2, dim=1) + dw.f(X[selection, :], n * delta_t)) * delta_t

    #loss = (Z_sum).mean()
    loss = torch.var(yt)

    return loss

def double_well_1d_gradient(x):
    alpha = 1
    return 4 * alpha * x * (x**2 - 1)

def save_nn_coefficients(thetas):
    from mds.utils import make_dir_path
    import os
    dir_path = 'data/testing_1d_bf_logvar_nn'
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'som.npz')
    np.savez(file_path, thetas=thetas)

if __name__ == "__main__":
    main()
