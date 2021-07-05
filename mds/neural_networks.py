import numpy as np

import torch
import torch.nn as nn

import os

class GaussianAnsatzNN(nn.Module):
    def __init__(self, n, m, means, cov):
        super(GaussianAnsatzNN, self).__init__()

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
        x = self.vec_mvn_pdf(x, self.means, self.cov)
        x = self.linear(x)
        return x

    def get_parameters(self):
        ''' get flattened parameters of the model
        '''
        A = self._modules['linear']._parameters['weight']
        theta = torch.squeeze(A).detach().numpy()
        return theta

    def load_parameters(self, theta):
        ''' load model parameters.
        '''
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''
        self.linear._parameters['weight'] = torch.tensor(
            theta.reshape(1, self.m),
            requires_grad=True,
            dtype=torch.float,
        )

    def get_rel_path(self):
        rel_path = os.path.join(
            'gaussian-ansatz-nn',
            'm_{}'.format(self.m),
        )
        return rel_path


class TwoLayerNN(nn.Module):
    def __init__(self, d_in, d_1, d_out):
        super(TwoLayerNN, self).__init__()

        # model name
        self.name = 'two-layer-nn'

        # define the two linear layers
        self.d_in = d_in
        self.d_1 = d_1
        self.d_out = d_out
        self.linear1 = nn.Linear(d_in, d_1, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_1, d_out, bias=True)

        # dimension of flattened parameters
        d_flat_A1 = self.linear1._parameters['weight'].flatten().shape[0]
        d_flat_b1 = self.linear1._parameters['bias'].shape[0]
        d_flat_A2 = self.linear2._parameters['weight'].flatten().shape[0]
        d_flat_b2 = self.linear2._parameters['bias'].shape[0]
        self.d_flat = d_flat_A1 + d_flat_b1 + d_flat_A2 + d_flat_b2

        # flattened parameters indices
        self.idx_A1 = slice(0, d_flat_A1)
        self.idx_b1 = slice(d_flat_A1, d_flat_A1 + d_flat_b1)
        self.idx_A2 = slice(d_flat_A1 + d_flat_b1, d_flat_A1 + d_flat_b1 + d_flat_A2)
        self.idx_b2 = slice(d_flat_A1 + d_flat_b1 + d_flat_A2, self.d_flat)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def get_parameters(self):
        ''' get flattened parameters of the model
        '''

        # get nn parameters
        A1 = self._modules['linear1']._parameters['weight']
        b1 = self._modules['linear1']._parameters['bias']
        A2 = self._modules['linear2']._parameters['weight']
        b2 = self._modules['linear2']._parameters['bias']

        # preallocate flatten parameters
        flatten_theta = np.empty(self.d_flat)

        # load parameters
        flatten_theta[self.idx_A1] = A1.detach().numpy().flatten()
        flatten_theta[self.idx_b1] = b1.detach().numpy()
        flatten_theta[self.idx_A2] = A2.detach().numpy().flatten()
        flatten_theta[self.idx_b2] = b2.detach().numpy()
        return flatten_theta

    def load_parameters(self, theta):
        ''' load model parameters.
        '''
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


class FeedForwardNN(nn.Module):
    def __init__(self, d_layers):
        super(FeedForwardNN, self).__init__()

        # model name
        self.name = 'feed-forward-nn'

        # layers and architecture
        assert type(d_layers) == list, ''
        assert len(d_layers) >= 2, ''
        self.d_layers = d_layers
        self.n_layers = len(d_layers) - 1
        self.d_in = d_layers[0]
        self.d_out = d_layers[-1]
        self.d_inner_layers = d_layers[1:-1]

        # flattened dimension of all parameters
        self.d_flat = 0

        # running idx
        idx = 0

        for i in range(self.n_layers):

            # define linear layers
            setattr(
                self,
                'linear{:d}'.format(i+1),
                nn.Linear(self.d_layers[i], self.d_layers[i+1], bias=True),
            )

            # dimension of flattened parameters
            linear = getattr(self, 'linear{:d}'.format(i+1))
            d_flat_A = linear._parameters['weight'].flatten().shape[0]
            d_flat_b = linear._parameters['bias'].shape[0]
            self.d_flat += d_flat_A + d_flat_b

            # flattened parameters indices
            setattr(
                self,
                'idx_A{:d}'.format(i+1),
                slice(idx, idx + d_flat_A),
            )
            setattr(
                self,
                'idx_b{:d}'.format(i+1),
                slice(idx + d_flat_A, idx + d_flat_A + d_flat_b),
            )
            idx += d_flat_A + d_flat_b

        # define relu as activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n_layers):

            # linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))
            x = linear(x)

            # activation function
            if i != self.n_layers -1:
                x = self.relu(x)

        return x

    def get_parameters(self):
        ''' get flattened parameters of the model.
        '''

        # preallocate flattened parameters
        flatten_theta = np.empty(self.d_flat)

        for i in range(self.n_layers):

            # get linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            # get indices of its weights and bias parameters
            idx_A = getattr(self, 'idx_A{:d}'.format(i+1))
            idx_b = getattr(self, 'idx_b{:d}'.format(i+1))

            # fill the flattened array
            flatten_theta[idx_A] = linear._parameters['weight'].detach().numpy().flatten()
            flatten_theta[idx_b] = linear._parameters['bias'].detach().numpy()

        return flatten_theta

    def load_parameters(self, theta):
        ''' load model parameters.
        '''
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''

        for i in range(self.n_layers):

            # get linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            # get indices of its weights and bias parameters
            idx_A = getattr(self, 'idx_A{:d}'.format(i+1))
            idx_b = getattr(self, 'idx_b{:d}'.format(i+1))

            # load weights
            linear._parameters['weight'] = torch.tensor(
                theta[idx_A].reshape(self.d_layers[i+1], self.d_layers[i]),
                requires_grad=True,
                dtype=torch.float32,
            )

            # load bias
            linear._parameters['bias'] = torch.tensor(
                theta[idx_b],
                requires_grad=True,
                dtype=torch.float32,
            )

    def get_rel_path(self):
        # make string from the dimensions of the inner layers
        arch_str = ''
        for d_layer in self.d_inner_layers:
            arch_str += '{:d}-'.format(d_layer)
        arch_str = arch_str[:-1]

        rel_path = os.path.join(
            self.name,
            'arch_' + arch_str,
        )
        return rel_path

    def write_parameters(self, f):
        #f.write('hidden layer dim: {:d}\n'.format(self.d_1))
        f.write('flattened parameters dim: {:d}\n'.format(self.d_flat))


class DenseNN(nn.Module):
    def __init__(self, d_layers):
        super(DenseNN, self).__init__()

        # model name
        self.name = 'dense-nn'

        # define the two linear layers
        assert type(d_layers) == list, ''
        assert len(d_layers) >= 2, ''
        self.d_layers = d_layers
        self.n_layers = len(d_layers) - 1
        self.d_in = d_layers[0]
        self.d_out = d_layers[-1]
        self.d_inner_layers = d_layers[1:-1]

        # flattened dimension of all parameters
        self.d_flat = 0

        # running idx
        idx = 0

        for i in range(self.n_layers):

            # define linear layers
            setattr(
                self,
                'linear{:d}'.format(i+1),
                nn.Linear(int(np.sum(self.d_layers[:i+1])), self.d_layers[i+1], bias=True),
            )

            # dimension of flattened parameters
            linear = getattr(self, 'linear{:d}'.format(i+1))
            d_flat_A = linear._parameters['weight'].flatten().shape[0]
            d_flat_b = linear._parameters['bias'].shape[0]
            self.d_flat += d_flat_A + d_flat_b

            # flattened parameters indices
            setattr(
                self,
                'idx_A{:d}'.format(i+1),
                slice(idx, idx + d_flat_A),
            )
            setattr(
                self,
                'idx_b{:d}'.format(i+1),
                slice(idx + d_flat_A, idx + d_flat_A + d_flat_b),
            )
            idx += d_flat_A + d_flat_b

        # define relu as activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n_layers):

            # linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            if i != self.n_layers - 1:
                x = torch.cat([x, self.relu(linear(x))], dim=1)
            else:
                x = linear(x)
        return x

    def get_parameters(self):
        ''' get flattened parameters of the model.
        '''

        # preallocate flattened parameters
        flatten_theta = np.empty(self.d_flat)

        for i in range(self.n_layers):

            # get linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            # get indices of its weights and bias parameters
            idx_A = getattr(self, 'idx_A{:d}'.format(i+1))
            idx_b = getattr(self, 'idx_b{:d}'.format(i+1))

            # fill the flattened array
            flatten_theta[idx_A] = linear._parameters['weight'].detach().numpy().flatten()
            flatten_theta[idx_b] = linear._parameters['bias'].detach().numpy()

        return flatten_theta

    def load_parameters(self, theta):
        ''' load model parameters.
        '''
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''

        for i in range(self.n_layers):

            # get linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            # get indices of its weights and bias parameters
            idx_A = getattr(self, 'idx_A{:d}'.format(i+1))
            idx_b = getattr(self, 'idx_b{:d}'.format(i+1))

            # load weights
            linear._parameters['weight'] = torch.tensor(
                theta[idx_A].reshape(self.d_layers[i+1], int(np.sum(self.d_layers[:i+1]))),
                requires_grad=True,
                dtype=torch.float32,
            )

            # load bias
            linear._parameters['bias'] = torch.tensor(
                theta[idx_b],
                requires_grad=True,
                dtype=torch.float32,
            )

    def get_rel_path(self):
        # make string from the dimensions of the inner layers
        arch_str = ''
        for d_layer in self.d_inner_layers:
            arch_str += '{:d}-'.format(d_layer)
        arch_str = arch_str[:-1]

        rel_path = os.path.join(
            self.name,
            'arch_' + arch_str,
        )
        return rel_path

    def write_parameters(self, f):
        #f.write('hidden layer dim: {:d}\n'.format(self.d_1))
        f.write('flattened parameters dim: {:d}\n'.format(self.d_flat))

