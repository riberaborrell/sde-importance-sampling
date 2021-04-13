from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils import get_nn_function_approximation_dir_path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class FunctionApproximation():

    def __init__(self, target_function, model, initialization='random'):
        assert target_function in ['value-f', 'control'], ''
        assert initialization in ['random', 'null', 'meta', 'hjb'], ''

        self.target_function = target_function
        self.model = model
        self.initialization = initialization

        self.dir_path = None

    def set_dir_path(self, settings_dir_path):
        self.dir_path = get_nn_function_approximation_dir_path(
            settings_dir_path,
            self.target_function,
            self.model.get_rel_path(),
            self.initialization,
        )

    def reset_parameters(self):
        self.initialization = 'random'
        for layer in self.model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

    def zero_parameters(self):
        self.initialization = 'null'
        for layer in self.model.children():
            for key in layer._parameters:
                layer._parameters[key] = torch.zeros_like(
                    layer._parameters[key], requires_grad=True
                )

    def fit_parameters_from_metadynamics(self, sde, iterations_lim=10000, epsilon=0.01):

        # parameters
        self.initialization = 'meta'

        # load meta bias potential
        meta_bias_pot = sde.get_meta_bias_potential(dt=0.001, sigma_i_meta=0.5, k=100, N_meta=1)
        meta_ms = meta_bias_pot['ms']
        meta_means = meta_bias_pot['means']
        meta_cov = meta_bias_pot['cov']
        meta_thetas = meta_bias_pot['thetas']

        # get means and thetas for trajectory i
        i = 0
        idx_i = slice(np.sum(meta_ms[:i]), np.sum(meta_ms[:i]) + meta_ms[i])
        meta_means_i = meta_means[idx_i]
        meta_thetas_i = meta_thetas[idx_i]

        # create ansatz functions from meta
        meta_ansatz = GaussianAnsatz(
            n=sde.n,
            potential_name=sde.potential_name,
            alpha=sde.alpha,
            beta=sde.beta,
        )
        meta_ansatz.set_given_ansatz_functions(
            means=meta_means_i,
            cov=meta_cov,
        )
        meta_ansatz.theta = meta_thetas_i

        # define optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01,
        )

        for i in np.arange(iterations_lim):

            # sample training data
            x = sde.sample_domain_uniformly(N=1000)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # ansatz functions evaluated at the grid
            if self.target_function == 'value-f':
                pass
                #target = meta_ansatz.value_function(x)
            elif self.target_function == 'control':
                target = meta_ansatz.control(x)
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

            # define loss
            inputs = self.model.forward(x_tensor)
            loss = nn.MSELoss()
            output = loss(inputs, target_tensor)
            #print('{:d}, {:2.3f}'.format(i, output))

            # stop if we have reached enough accuracy
            if output <= epsilon:
                break

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('{:d}, {:2.3f}'.format(i, output))
