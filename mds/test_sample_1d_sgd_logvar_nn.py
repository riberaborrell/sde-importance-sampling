from mds.base_parser_nd import get_base_parser
from mds.neural_networks import FeedForwardNN, DenseNN
from mds.test_sample_sgd_nn import double_well_1d_gradient, get_idx_new_in_ts, save_som

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

    # get dimensions of each layer
    if args.d_layers is not None:
        d_layers = [args.n] + args.d_layers + [args.n]
    else:
        d_layers = [args.n, args.n]

    # initialize nn model 
    if not args.dense:
        model = FeedForwardNN(d_layers, args.activation_type, args.seed)
    else:
        model = DenseNN(d_layers, args.activation_type, args.seed)

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # preallocate parameters
    thetas = np.empty((args.iterations_lim, model.d_flat))
    losses = np.empty(args.iterations_lim)
    means = np.empty(args.iterations_lim)
    res = np.empty(args.iterations_lim)
    time_steps = np.empty(args.iterations_lim, dtype=np.int64)
    cts = np.empty(args.iterations_lim)

    # save initial parameters
    thetas[0] = model.get_parameters()

    for update in np.arange(args.iterations_lim):
        # reset gradients
        optimizer.zero_grad()

        # compute loss
        log_var_loss, loss, mean, re, k, ct = sample_loss(model, args.dt, args.N)
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        log_var_loss.backward()

        # update parameters
        optimizer.step()

        # save parameters
        thetas[update] = model.get_parameters()
        losses[update] = loss
        means[update] = mean
        res[update] = re
        time_steps[update] = k
        cts[update] = ct

    dir_path = 'data/testing_1d_sgd_nn/logvar'
    files_dict = {
        'thetas': thetas,
        'losses': losses,
        'means': means,
        'res': res,
        'time_steps': time_steps,
        'cts': cts,
    }
    save_som(dir_path, files_dict)


def sample_loss(model, dt, N):
    beta = 1
    k_max = 10**6

    # start timer
    ct_initial = time.time()

    # time increments as tensors
    dt = torch.tensor([dt])

    # initialize trajectories of processes X and Y 
    xt = - torch.ones(N).reshape(N, 1)
    yt = torch.zeros(N).reshape(N)

    # initialize fht
    fht = np.zeros(N)

    # initialize running integrals
    det_int_fht = np.zeros(N)
    stoch_int_fht = np.zeros(N)

    # initialize Z sum
    zt_sum = np.zeros(N)

    # number of trajectories not in the target set
    N_not_in_ts = N

    for k in np.arange(1, k_max + 1):

        # stop trajetories if all trajectories are in the target set
        idx = xt[:, 0] < 1
        breakpoint()
        N_not_in_ts = torch.sum(idx)
        if N_not_in_ts == 0:
            break

        # normal distributed vector
        xi = torch.randn(N_not_in_ts).reshape(N_not_in_ts, 1)

        # Brownian increment
        dB = torch.sqrt(dt) * xi

        # Z process
        zt = - model.forward(xt[idx, :])

        # control
        ut = torch.zeros(N_not_in_ts).reshape(N_not_in_ts, 1)

        # adaptive forward process
        ut = - zt

        # detach
        ut = ut.detach()

        # step dynamics process X forward
        drift = (- double_well_1d_gradient(xt[idx, :]) + np.sqrt(2) * ut) * dt
        diffusion = np.sqrt(2 / beta) * dB
        xt[idx, :] = xt[idx, :] + drift + diffusion

        # step dynamics process Y
        h = (-1 + 0.5 * zt ** 2).reshape(N_not_in_ts,)
        yt[idx] = yt[idx] \
                + (h + torch.sum(zt * ut, dim=1)) * dt \
                + torch.sum(zt * dB, dim=1)

        # update zt_sum
        zt_sum[idx] += (1 + 0.5 * np.sum(zt.detach().numpy() ** 2, axis=1)) * dt.detach().numpy()

        # numpy array indices
        idx = idx.detach().numpy()

        # update fht
        fht[idx] += dt.detach().numpy()

        # update running integrals
        det_int_fht[idx] += (np.linalg.norm(ut.detach().numpy(), axis=1) ** 2) * dt.detach().numpy()
        stoch_int_fht[idx] += np.matmul(
            ut.detach().numpy()[:, np.newaxis, :],
            dB.detach().numpy()[:, :, np.newaxis],
        ).squeeze()


    # compute loss
    log_var_loss = torch.var(yt)
    loss = np.mean(zt_sum)

    # compute mean and re of I_u
    I_u = np.exp(- fht - np.sqrt(beta) * stoch_int_fht - (beta / 2) * det_int_fht)
    mean_I_u = np.mean(I_u)
    var_I_u = np.var(I_u)
    re_I_u = np.sqrt(var_I_u) / mean_I_u

    # end timer
    ct_final = time.time()

    return log_var_loss, loss, mean_I_u, re_I_u, k, ct_final - ct_initial


if __name__ == "__main__":
    main()
