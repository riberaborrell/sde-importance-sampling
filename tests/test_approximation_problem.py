import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from function_approximation.gaussian_ansatz import GaussianAnsatz
from function_approximation.models import FeedForwardNN, DenseNN
from function_approximation.function_approximation import FunctionApproximation
from sde.langevin_sde import LangevinSDE


class TestApproximationProblem:


    @pytest.fixture
    def sde(self, problem_name, potential_name, d, alpha_i, beta):
        ''' creates Langevin SDE object with the given setting.
        '''
        sde = LangevinSDE(
            problem_name=problem_name,
            potential_name=potential_name,
            d=d,
            alpha=np.full(d, alpha_i),
            beta=beta,
        )
        return sde

    @pytest.fixture
    def gaussian_ansatz(self, sde):
        '''initializes Gaussian Ansatz object with just one Gaussian function.
        '''

        # init object
        ansatz = GaussianAnsatz(sde, normalized=False)

        # set gaussian
        means = - 1. * np.ones((1, sde.d))
        sigma_i = 1.
        ansatz.set_given_ansatz_functions(means=means, sigma_i=sigma_i)

        # set weights
        ansatz.theta = 1. * np.ones(1)

        return ansatz


    @pytest.fixture
    def function_appr(self, sde, dense):

        # initialize feed-forward nn
        if not dense:
            model = FeedForwardNN(
                d_layers=[sde.d, 30, 30, sde.d],
                activation_type='tanh',
            )

        # initialize dense nn
        else:
            model = DenseNN(
                d_layers=[sde.d, 30, 30, sde.d],
                activation_type='tanh',
            )

        # initialize function approximation
        func = FunctionApproximation(
            target_function='control',
            model=model,
            initialization='random',
            training_algorithm='alternative',
        )

        return func

    def test_alternative_training_algorithm(self, sde, gaussian_ansatz, function_appr, K_train, lr):
        '''
        '''

        # set sgd parameters
        n_iterations_lim = 10**3

        # preallocate losses
        losses_train = np.empty(n_iterations_lim)

        # model
        model = function_appr.model

        # define mean square error loss
        loss = nn.MSELoss()

        # define optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
        )

        for i in np.arange(n_iterations_lim):

            # sample training data
            x = sde.sample_domain_uniformly(K=K_train)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # evaluate target function
            target = gaussian_ansatz.control(x)
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

        # evaluate at (-1, ..., -1)
        x_tensor = -1 * torch.ones(1, sde.d)
        x = x_tensor.detach().numpy()
        target_function_at_x = gaussian_ansatz.control(x)
        model_at_x = model.forward(x_tensor).detach().numpy()
        assert np.allclose(target_function_at_x, model_at_x, atol=10**-2)

        # evaluate at (0, ..., 0)
        x_tensor = 0 * torch.ones(1, sde.d)
        x = x_tensor.detach().numpy()
        target_function_at_x = gaussian_ansatz.control(x)
        model_at_x = model.forward(x_tensor).detach().numpy()
        assert np.allclose(target_function_at_x, model_at_x, atol=10**-2)

        # evaluate at (0.5, ..., 0.5)
        x_tensor = 0.5 * torch.ones(1, sde.d)
        x = x_tensor.detach().numpy()
        target_function_at_x = gaussian_ansatz.control(x)
        model_at_x = model.forward(x_tensor).detach().numpy()
        assert np.allclose(target_function_at_x, model_at_x, atol=10**-2)
