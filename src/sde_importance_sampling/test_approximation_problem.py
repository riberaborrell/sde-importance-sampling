from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz
from sde_importance_sampling.neural_networks import FeedForwardNN, DenseNN
from sde_importance_sampling.function_approximation import FunctionApproximation
from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.utils_path import make_dir_path, get_time_in_hms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch

import time
import os

def get_parser():
    parser = get_base_parser()
    return parser

def classic_training_algorithm(sde, ansatz, func, n_iterations_lim, N_train, lr):
    '''
    '''
    # model
    model = func.model

    # preallocate losses
    losses_train = np.empty(n_iterations_lim)
    losses_test = np.empty(n_iterations_lim)

    # define mean square error loss
    loss = nn.MSELoss()

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # get data points
    sde.discretize_domain(h=0.1)
    data = sde.get_flat_domain_h()
    N_data = data.shape[0]

    # split into training and test data
    N_train = int(0.75 * N_data)
    np.random.shuffle(data)
    x_train = data[:N_train]
    x_test = data[N_train:]
    N_test = x_test.shape[0]

    # convert to tensors
    x_train_tensor = torch.tensor(x_train, requires_grad=False, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, requires_grad=False, dtype=torch.float32)

    # evaluation
    target_train = ansatz.control(x_train)
    target_test = ansatz.control(x_test)

    # tensorize target function evaluation
    target_train_tensor = torch.tensor(
        target_train,
        requires_grad=False,
        dtype=torch.float32,
    )
    target_test_tensor = torch.tensor(
        target_test,
        requires_grad=False,
        dtype=torch.float32,
    )

    for i in np.arange(n_iterations_lim):

        # evaluate nn control at the training data
        inputs_train = model.forward(x_train_tensor)
        inputs_test = model.forward(x_test_tensor)

        # compute loss
        output_train = loss(inputs_train, target_train_tensor)
        output_test = loss(inputs_test, target_test_tensor)
        if i % 100 == 0:
            msg = 'it.: {:d}, loss (train): {:2.3e}, loss (test): {:2.3e}' \
                  ''.format(i, output_train, output_test)
            print(msg)

        # save loss
        losses_train[i] = output_train.detach().numpy()
        losses_test[i] = output_test.detach().numpy()

        # compute gradients
        output_train.backward()

        # update parameters
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

    print('it.: {:d}, loss (train): {:2.3e}, loss (test): {:2.3e} \n' \
          ''.format(i, output_train, output_test))

    # number of iterations used
    n_iterations = i
    return losses_train, losses_test


def alternative_training_algorithm(sde, ansatz, func, n_iterations_lim, N_train, lr):
    '''
    '''
    # model
    model = func.model

    # preallocate losses
    losses_train = np.empty(n_iterations_lim)

    # define mean square error loss
    loss = nn.MSELoss()

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # multivariate normal sampling
    mean = ansatz.means[0]
    cov = ansatz.cov

    for i in np.arange(n_iterations_lim):

        # sample training data
        #x = sde.sample_domain_uniformly(N=N_train, subset=np.full((sde.n, 2), [-2, 0]))
        #x = sde.sample_domain_ith_coordinate_uniformly(i=0, N=N_train)
        x = sde.sample_multivariate_normal(mean=mean, cov=cov, N=N_train)
        x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

        # evaluate target function
        target = ansatz.control(x)
        target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

        # evaluate model
        inputs = model.forward(x_tensor)

        # compute mse loss
        output = loss(inputs, target_tensor)

        if i % 100 == 0:
            print('it.: {:d}, loss: {:2.3e}'.format(i, output))

        # save loss
        losses_train[i] = output.detach().numpy()

        # compute gradients
        output.backward()

        # update parameters
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

    print('it.: {:d}, loss: {:2.3e}\n'.format(i, output))
    return losses_train

def get_dir_path(training_algorithm):
    dir_path = 'data/testing_approximation_problem/{}/'.format(training_algorithm)
    return dir_path

def get_file_name(n, n_layers, d_layer, n_iterations_lim, N_train, lr):
    file_name = ''
    file_name += 'n-{:d}_'.format(n)
    file_name += 'n-layers-{:d}_'.format(n_layers)
    file_name += 'd-layer-{:d}_'.format(d_layer)
    file_name += 'n-iter-{:.0e}_'.format(n_iterations_lim)
    file_name += 'N-train-{:.0e}_'.format(N_train)
    file_name += 'lr-{:.0e}'.format(lr)
    file_name += '.npz'
    return file_name

def save(dir_path, file_name, files_dict):
    make_dir_path(dir_path)

    file_path = os.path.join(dir_path, file_name)
    np.savez(file_path, **files_dict)

def load(dir_path, file_name):
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, file_name)
    data = np.load(
        file_path,
        allow_pickle=True,
    )
    return data

def main():
    args = get_parser().parse_args()

    # dimension
    n = args.n

    # nn architecture
    n_layers = args.n_layers
    d_layer = args.d_layer

    # training algorithm
    training_algorithm = args.train_alg

    # sgd
    n_iterations_lim = args.n_iterations_lim
    N_train = args.N_train
    lr = args.lr

    # approximation problem
    if not args.load:

        # start timer
        ct_initial = time.perf_counter()

        # initialize langevin sde object
        sde = LangevinSDE(
            problem_name='langevin_stop-t',
            potential_name='nd_2well',
            n=n,
            alpha=np.full(n, 1.),
            beta=args.beta,
        )

        # init gaussian ansatz
        ansatz = GaussianAnsatz(n, args.beta, normalized=False)

        # set gaussian
        means = -1. * np.ones((1, n))
        cov = 1 * np.eye(n)
        ansatz.set_given_ansatz_functions(means, cov)

        # set weights
        ansatz.theta = 1. * np.ones(1)

        # get function approximator
        d_innen_layers = [d_layer for i in range(n_layers)]
        d_layers = [sde.n] + d_innen_layers + [sde.n]
        model = FeedForwardNN(
            d_layers=d_layers,
            activation_type='tanh',
        )

        # initialize function approximation
        func = FunctionApproximation(
            target_function='control',
            model=model,
        )


        # classic training algorithm
        if training_algorithm == 'classic':
            losses_train, losses_test = classic_training_algorithm(sde, ansatz, func,
                                                                   n_iterations_lim, N_train, lr)
        # alternative training algorithm
        elif training_algorithm == 'alternative':
            losses_train = alternative_training_algorithm(sde, ansatz, func,
                                                          n_iterations_lim, N_train, lr)

        # inputs
        x = np.empty((6, sde.n))

        # x_1 = {-1, 0, 1}, x_2, ..., x_n = 0.
        x[:3, 1:] = 0.
        x[0, 0] = -1.
        x[1, 0] = 0.
        x[2, 0] = 1.

        # x_1, ..., x_n = {-1, 0, 1}.
        x[3, :] = -1.
        x[4, :] = 0.
        x[5, :] = 1.

        x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

        target_at_x = ansatz.control(x)
        model_at_x = model.forward(x_tensor).detach().numpy()

        # end timer
        ct = time.perf_counter() - ct_initial

        # npz files dictionary
        files_dict = {
            'n_iterations_lim': n_iterations_lim,
            'N_train': N_train,
            'lr': lr,
            'theta': model.get_parameters(),
            'losses_train': losses_train,
            'x': x,
            'target_at_x': target_at_x,
            'model_at_x': model_at_x,
            'ct': ct,
        }

        if training_algorithm == 'classic':
            files_dict['losses_test'] = losses_test

        dir_path = get_dir_path(training_algorithm)
        file_name = get_file_name(n, n_layers, d_layer, n_iterations_lim, N_train, lr)
        save(dir_path, file_name, files_dict)

    # load already computed approximation problem
    if args.load:

        # load npz files
        dir_path = get_dir_path(training_algorithm)
        file_name = get_file_name(n, n_layers, d_layer, n_iterations_lim, N_train, lr)
        data = load(dir_path, file_name)

        # get target function and model at x
        x = data['x']
        target_function = np.around(data['target_at_x'], 3)
        model = np.around(data['model_at_x'], 3)

        print('x:\n {}\n'.format(x))
        print('target function at x: \n{}\n'.format(target_function))
        print('model at x: \n{}\n'.format(model))

        # get time
        h, m, s = get_time_in_hms(data['ct'])
        print('Computational time: {:d}:{:02d}:{:02.2f}\n\n'.format(h, m, s))

        # plot training losses
        n_iterations_lim = data['n_iterations_lim']
        x = np.arange(n_iterations_lim)
        losses_train = data['losses_train']
        if training_algorithm == 'classic':
            losses_test = data['losses_test']

        plt.semilogy(x, losses_train, label='train')
        if training_algorithm == 'classic':
            plt.semilogy(x, losses_test, label='test')
        plt.legend()
        plt.show()


if __name__ == "__main__":
        main()
