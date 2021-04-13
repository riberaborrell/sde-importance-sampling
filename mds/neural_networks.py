from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils import get_two_layer_nn_dir_path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, d_1, d_out):
        super(TwoLayerNet, self).__init__()

        # define the two linear layers
        self.d_in = d_in
        self.d_1 = d_1
        self.d_out = d_out
        self.linear1 = nn.Linear(d_in, d_1, bias=True)
        self.linear2 = nn.Linear(d_1, d_out, bias=True)

        # dimension of flattened parameters
        self.d_flatten = self.d_1 * self.d_in + self.d_1 + self.d_out * self.d_1 + self.d_out

        # flattened parameters indices
        self.idx_A1 = slice(0, self.d_out * self.d_1)
        self.idx_b1 = slice(self.d_out * self.d_1, self.d_out * self.d_1 + self.d_1)
        self.idx_A2 = slice(self.d_out * self.d_1 + self.d_1, self.d_out * self.d_1 + self.d_1 + self.d_out * self.d_1)
        self.idx_b2 = slice(self.d_out * self.d_1 + self.d_1 + self.d_out * self.d_1, self.d_flatten)

        # parameters
        self.initialization = 'random'

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def reset_parameters(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

    def zero_parameters(self):
        for layer in self.children():
            for key in layer._parameters:
                layer._parameters[key] = torch.zeros_like(
                    layer._parameters[key], requires_grad=True
                )
        self.initialization = 'null'

    def get_flatten_parameters(self):

        # get nn parameters
        A1 = self._modules['linear1']._parameters['weight']
        b1 = self._modules['linear1']._parameters['bias']
        A2 = self._modules['linear2']._parameters['weight']
        b2 = self._modules['linear2']._parameters['bias']

        # preallocate flatten parameters
        flatten_theta = np.empty(self.d_flatten)

        # load parameters
        flatten_theta[self.idx_A1] = A1.detach().numpy().reshape(self.d_1 * self.d_in)
        flatten_theta[self.idx_b1] = b1.detach().numpy()
        flatten_theta[self.idx_A2] = A2.detach().numpy().reshape(self.d_out * self.d_1)
        flatten_theta[self.idx_b2] = b2.detach().numpy()
        return flatten_theta

    def load_parameters(self, theta):
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flatten, ''

        self.linear1._parameters['weight'] = torch.tensor(
            theta[self.idx_A1].reshape(self.d_1, self.d_in),
            requires_grad=True,
            dtype=torch.float,
        )
        self.linear1._parameters['bias'] = torch.tensor(
            theta[self.idx_b1],
            requires_grad=True,
            dtype=torch.float,
        )
        self.linear2._parameters['weight'] = torch.tensor(
            theta[self.idx_A2].reshape(self.d_out, self.d_1),
            requires_grad=True,
            dtype=torch.float,
        )
        self.linear2._parameters['bias'] = torch.tensor(
            theta[self.idx_b2],
            requires_grad=True,
            dtype=torch.float,
        )

    def set_dir_path(self, settings_dir_path):
        self.dir_path = get_two_layer_nn_dir_path(
            settings_dir_path,
            self.d_1,
            self.initialization,
        )

    def fit_parameters_from_metadynamics(self, sde, iterations_lim=10000, epsilon=0.01):

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
            self.parameters(),
            lr=0.01,
        )

        for i in np.arange(iterations_lim):

            # sample training data
            x = sde.sample_domain_uniformly(N=1000)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # ansatz functions evaluated at the grid
            target = meta_ansatz.control(x)
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

            # define loss
            inputs = self.forward(x_tensor)
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
        # parameters
        self.initialization = 'meta'

