import numpy as np

import torch
import torch.nn as nn

import os

class GaussianAnsatz(nn.Module):
    def __init__(self, n, m, means, cov):
        super(GaussianAnsatz, self).__init__()

        # define the scalar product layer
        self.n = n
        self.m = m
        self.linear = nn.Linear(m, 1, bias=False)

        # dimension of flattened parameters
        self.d_flat = m

        # means and covariance
        self.means = means
        self.cov = cov

    def vec_mvn_pdf(self, x, means=None, cov=None):
        ''' Vectorized multivariate normal pdf (nd Gaussian) v(x; means, cov) with means evaluated at x
            x ((N, n)-tensor) : position
            means ((m, n)-tensor) : center of the gaussian
            cov ((n, n)-tensor) : covariance matrix
        '''
        assert x.ndim == 2, ''
        assert x.size(1) == self.n, ''
        if means is None:
            means = torch.rand(self.m, self.n)
        if cov is None:
            cov = torch.eye(self.n)
        assert means.ndim == 2, ''
        assert means.size(1) == self.n, ''
        assert cov.ndim == 2, ''
        assert cov.shape == (self.n, self.n), ''

        # get batch size
        N = x.size(0)

        # compute norm factor
        norm_factor = np.sqrt(((2 * np.pi) ** self.n) * torch.linalg.det(cov))
        inv_cov = torch.linalg.inv(cov)

        # unsqueeze x and mean
        x = torch.unsqueeze(x, 1)
        means = torch.unsqueeze(means, 0)

        # compute exponential term
        x_centered = (x - means).reshape(N * self.m, self.n)
        exp_term = torch.matmul(x_centered, inv_cov)
        exp_term = torch.sum(exp_term * x_centered, axis=1).reshape(N, self.m)
        exp_term *= - 0.5

        mvn_pdf = np.exp(exp_term) / norm_factor

        return mvn_pdf

    def forward(self, x):
        v = self.vec_mvn_pdf(x, self.means, self.cov)
        y_pred = self.linear(v)
        #y_pred = torch.squeeze(y_pred)
        return y_pred

    def get_rel_path(self):
        rel_path = os.path.join(
            'gaussian-ansatz-nn',
            'm_{}'.format(self.m),
        )
        return rel_path

    def get_flatten_parameters(self):
        A = self._modules['linear']._parameters['weight']
        theta = torch.squeeze(A).detach().numpy()
        return theta

    def load_parameters(self, theta):
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''
        self.linear._parameters['weight'] = torch.tensor(
            theta.reshape(1, self.m),
            requires_grad=True,
            dtype=torch.float,
        )


class TwoLayerNet(nn.Module):
    def __init__(self, d_in, d_1, d_out):
        super(TwoLayerNet, self).__init__()

        # model name
        self.name = 'two-layer-nn'

        # define the two linear layers
        self.d_in = d_in
        self.d_1 = d_1
        self.d_out = d_out
        self.linear1 = nn.Linear(d_in, d_1, bias=True)
        self.linear2 = nn.Linear(d_1, d_out, bias=True)

        # dimension of flattened parameters
        self.d_flat = self.d_1 * self.d_in + self.d_1 + self.d_out * self.d_1 + self.d_out

        # flattened parameters indices
        self.idx_A1 = slice(0, self.d_out * self.d_1)
        self.idx_b1 = slice(self.d_out * self.d_1, self.d_out * self.d_1 + self.d_1)
        self.idx_A2 = slice(self.d_out * self.d_1 + self.d_1, self.d_out * self.d_1 + self.d_1 + self.d_out * self.d_1)
        self.idx_b2 = slice(self.d_out * self.d_1 + self.d_1 + self.d_out * self.d_1, self.d_flat)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def get_flatten_parameters(self):

        # get nn parameters
        A1 = self._modules['linear1']._parameters['weight']
        b1 = self._modules['linear1']._parameters['bias']
        A2 = self._modules['linear2']._parameters['weight']
        b2 = self._modules['linear2']._parameters['bias']

        # preallocate flatten parameters
        flatten_theta = np.empty(self.d_flat)

        # load parameters
        flatten_theta[self.idx_A1] = A1.detach().numpy().reshape(self.d_1 * self.d_in)
        flatten_theta[self.idx_b1] = b1.detach().numpy()
        flatten_theta[self.idx_A2] = A2.detach().numpy().reshape(self.d_out * self.d_1)
        flatten_theta[self.idx_b2] = b2.detach().numpy()
        return flatten_theta

    def load_parameters(self, theta):
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''

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

    def get_rel_path(self):
        rel_path = os.path.join(
            self.name,
            'd1_{}'.format(self.d_1),
        )
        return rel_path

    def write_parameters(self, f):
        f.write('hidden layer dim: {:d}\n'.format(self.d_1))
        f.write('flattened parameters dim: {:d}\n'.format(self.d_flat))
