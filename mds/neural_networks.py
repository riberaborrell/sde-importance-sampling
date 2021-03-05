import numpy as np

import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, d_1, d_out):
        super(TwoLayerNet, self).__init__()

        # define the two linear layers
        self.d_in = d_in
        self.d_1 = d_1
        self.d_out = d_out
        self.linear1 = nn.Linear(d_in, d_1, bias=True)
        self.linear2 = nn.Linear(d_1, d_out, bias=True)

        # flatten idx
        self.d_flatten = self.d_1 * self.d_in + self.d_1 + self.d_out * self.d_1 + self.d_out

        self.idx_A1 = slice(0, self.d_out * self.d_1)
        self.idx_b1 = slice(self.d_out * self.d_1, self.d_out * self.d_1 + self.d_1)
        self.idx_A2 = slice(self.d_out * self.d_1 + self.d_1, self.d_out * self.d_1 + self.d_1 + self.d_out * self.d_1)
        self.idx_b2 = slice(self.d_out * self.d_1 + self.d_1 + self.d_out * self.d_1, self.d_flatten)

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
