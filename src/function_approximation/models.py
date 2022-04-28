import numpy as np

import torch
import torch.nn as nn

import os

ACTIVATION_FUNCTION_TYPES = [
    'relu',
    'tanh',
]

class GaussianAnsatzNN(nn.Module):
    def __init__(self, sde, m_i, sigma_i, normalized=True, seed=None):
        super(GaussianAnsatzNN, self).__init__()

        ''' Gaussian ansatz NN model
            n (int): dimension
            beta (float) : inverse of the temperature
            m_i (int) : number of ansatz functions per axis
            means ((m, n)-tensor) : centers of the gaussian functions
            sigma_i (float): value in the diagonal entries of the covariance matrix
            cov ((n, n)-tensor) : covariance matrix
        '''

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

        # normalize Gaussian flag
        self.normalized = normalized

        # dimension and inverse of temperature
        self.d = sde.d

        # set gaussians uniformly in the domain
        self.set_unif_dist_ansatz_functions(sde.domain, m_i, sigma_i)

        # set parameters
        self.theta = torch.nn.Parameter(torch.randn(self.m))

        # dimension of flattened parameters
        self.d_flat = self.m

    def set_cov_matrix(self, sigma_i=None, cov=None):
        ''' sets the covariance matrix of the gaussian functions

        Parameters
        ----------
        sigma_i: float
            value in the diagonal entries of the covariance matrix.
        cov: tensor
            covariance matrix
        '''

        # scalar covariance matrix case
        if sigma_i is not None:

            # covariance matrix
            self.sigma_i = sigma_i
            self.cov = sigma_i * torch.eye(self.d)
            self.is_cov_scalar_matrix = True

            # compute inverse
            self.inv_cov = torch.eye(self.d) / sigma_i

            # compute determinant
            self.det_cov = sigma_i**self.d

        # general case
        if cov is not None:

            # check covariance matrix
            assert cov.ndim == 2, ''
            assert cov.shape == (self.d, self.d), ''

            # set covariance matrix
            self.cov = cov
            self.is_cov_scalar_matrix = False

            # compute inverse
            self.inv_cov = torch.linalg.inv(cov)

            # compute determinant
            self.det_cov = torch.linalg.det(cov)

    def set_unif_dist_ansatz_functions(self, domain, m_i, sigma_i):
        ''' sets the centers of the ansatz functions uniformly distributed along the domain
            with scalar covariance matrix

        Parameters
        ----------
        domain: array
            (d, 2)-array representing the boundaries of the domain
        m_i: float
            number of gaussian ansatz along each direction
        sigma_i: float
            value in the diagonal entries of the covariance matrix
        '''

        # set number of gaussians
        self.m_i = m_i
        self.m = m_i ** self.d

        # distribute centers of Gaussians uniformly
        mgrid_input = []
        for i in range(self.d):
            slice_i = slice(domain[i, 0], domain[i, 1], complex(0, m_i))
            mgrid_input.append(slice_i)
        means = np.mgrid[mgrid_input]
        means = np.moveaxis(means, 0, -1).reshape(self.m, self.d)
        self.means = torch.tensor(means, dtype=torch.float32)

        # set covariance matrix
        self.set_cov_matrix(sigma_i=sigma_i)


    def mvn_pdf_basis(self, x):
        ''' Multivariate normal pdf (nd Gaussian) basis v(x; means, cov) with different means
            but same covariance matrix evaluated at x

        Parameters
        ----------
        x: tensor
            position, (K, d)-tensor

        Returns:
            (K, m)-tensor
        '''
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.size(1) == self.d, ''
        K = x.size(0)

        # compute log of the basis

        # scalar covariance matrix
        if self.is_cov_scalar_matrix:
            log_mvn_pdf_basis = - 0.5 * torch.sum(
                (x.view(K, 1, self.d) - self.means.view(1, self.m, self.d))**2,
                axis=2,
            ) / self.sigma_i

            # add normalization factor
            if self.normalized:
                log_mvn_pdf_basis -= torch.log(2 * torch.tensor(np.pi) * self.sigma_i) \
                                   * self.d / 2

        # general covariance matrix
        else:
            #TODO! test

            # prepare position and means for broadcasting
            x = torch.unsqueeze(x, 1)
            means = torch.unsqueeze(self.means, 0)

            # compute exponential term
            x_centered = (x - means).reshape(K * self.m, self.d)
            exp_term = torch.matmul(x_centered, inv_cov)
            exp_term = torch.sum(exp_term * x_centered, axis=1).reshape(K, self.m)
            exp_term *= - 0.5

            # add normalization factor
            if self.normalized:

                log_mvn_pdf_basis -= torch.log((2 * torch.tensor(np.pi)) ** self.d * self.det_cov) / 2

        return torch.exp(log_mvn_pdf_basis)

    def mvn_pdf_gradient_basis(self, x):
        ''' Gradient of the multivariate normal pdf (nd Gaussian) \nabla v(x; means, cov)
        with means evaluated at x

        Parameters
        ----------
        x: tensor
            position, (K, d)-tensor

        Returns:
            (K, m, d)-tensor
        '''
        # assume shape of x array to be (K, d)
        assert x.ndim == 2, ''
        assert x.size(1) == self.d, ''
        K = x.size(0)

        # get nd gaussian basis
        mvn_pdf_basis = self.mvn_pdf_basis(x)

        # compute gradient of the exponential term
        if self.is_cov_scalar_matrix:
            grad_exp_term = (
                x.view(K, 1, self.d) - self.means.view(1, self.m, self.d)
            ) / self.sigma_i

        # general covariance matrix
        else:
            #TODO! test
            grad_exp_term = 0.5 * np.matmul(
                x.view(K, 1, self.d) - self.means.view(1, self.m, self.d),
                inv_cov + inv_cov.T,
            )

        # compute gaussian gradients basis
        return - grad_exp_term * mvn_pdf_basis[:, :, np.newaxis]

    def forward(self, x):
        x = self.mvn_pdf_gradient_basis(x)
        x = torch.tensordot(x, self.theta, dims=([1], [0]))
        return x

    def get_parameters(self):
        ''' get parameters of the model
        '''
        return self._parameters['theta'].detach().numpy()

    def load_parameters(self, theta):
        ''' load model parameters.
        '''
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''
        self._parameters['theta'] = torch.tensor(
            theta,
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
    def __init__(self, d_in, d_1, d_out, seed=None):
        super(TwoLayerNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

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
    def __init__(self, d_layers, activation_type='relu', seed=None):
        super(FeedForwardNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

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

        # type of activation function
        assert activation_type in ACTIVATION_FUNCTION_TYPES, ''
        self.activation_type = activation_type

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
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()

        # define tanh as activation function
        elif self.activation_type == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x):
        for i in range(self.n_layers):

            # linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))
            x = linear(x)

            # activation function
            if i != self.n_layers -1:
                x = self.activation(x)

        return x

    def get_gradient_parameters(self):
        '''
        '''

        # preallocate gradient of the parameters
        flatten_theta_grad = np.empty(self.d_flat)

        for i in range(self.n_layers):

            # get linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            # get indices of its weights and bias parameters
            idx_A = getattr(self, 'idx_A{:d}'.format(i+1))
            idx_b = getattr(self, 'idx_b{:d}'.format(i+1))

            # fill the flattened array
            flatten_theta_grad[idx_A] = linear._parameters['weight'].grad.detach().numpy().flatten()
            flatten_theta_grad[idx_b] = linear._parameters['bias'].grad.detach().numpy()

        return flatten_theta_grad

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
            'act_' + self.activation_type,
        )
        return rel_path

    def write_parameters(self, f):
        f.write('\nFeedforward NN\n')
        f.write('architecture: {}\n'.format(self.d_layers))
        f.write('activation function: {}\n'.format(self.activation_type))
        f.write('flattened parameters dim: {:d}\n'.format(self.d_flat))


class DenseNN(nn.Module):
    def __init__(self, d_layers, activation_type='relu', seed=None):
        super(DenseNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

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

        # type of activation function
        assert activation_type in ACTIVATION_FUNCTION_TYPES, ''
        self.activation_type = activation_type

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
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()

        # define tanh as activation function
        elif self.activation_type == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(self.n_layers):

            # linear layer
            linear = getattr(self, 'linear{:d}'.format(i+1))

            if i != self.n_layers - 1:
                x = torch.cat([x, self.activation(linear(x))], dim=1)
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
            'act_' + self.activation_type,
        )
        return rel_path

    def write_parameters(self, f):
        f.write('\nDense NN\n')
        f.write('architecture: {}\n'.format(self.d_layers))
        f.write('activation function: {}\n'.format(self.activation_type))
        f.write('flattened parameters dim: {:d}\n'.format(self.d_flat))

class SequentialNN(nn.Module):
    def __init__(self, N, d_layers, activation_type='relu', is_dense=False, seed=None):
        super(SequentialNN, self).__init__()

        # set seed
        if seed is not None:
            torch.manual_seed(seed)

        # model name
        self.name = 'sequential-nn'

        # number of models
        assert N >= 1, ''
        self.N = N

        # initialize models for each time step
        for i in range(N):
            if not is_dense:
                setattr(
                    self,
                    'model_{:d}'.format(i),
                    FeedForwardNN(d_layers, activation_type),
                )
            else:
                setattr(
                    self,
                    'model_{:d}'.format(i),
                    DenseNN(d_layers, activation_type),
                )

        # flattened dimension of each model
        self.d_flat_k = self.model_0.d_flat

        # flattened dimension of all parameters
        self.d_flat = self.d_flat_k * N

    def forward(self, k, x):
        model_k = getattr(self, 'model_{:d}'.format(k))
        return model_k.forward(x)

    def get_parameters(self):
        '''
        '''
        # preallocate flattened parameters
        flatten_theta = np.empty(self.d_flat)

        # get flattened parameters for each model
        for k in range(self.N):
            model_k = getattr(self, 'model_{:d}'.format(k))
            flatten_theta_k = model_k.get_parameters()
            flatten_theta[k * self.d_flat_k: (k + 1) * self.d_flat_k] = flatten_theta_k

        return flatten_theta

    def load_parameters(self, theta):
        ''' load model parameters
        '''
        assert theta.ndim == 1, ''
        assert theta.shape[0] == self.d_flat, ''

        # reshape theta
        theta = theta.reshape(self.N, self.d_flat_k)

        # load parameters for each model
        for k in range(self.N):
            model_k = getattr(self, 'model_{:d}'.format(k))
            model_k.load_parameters(theta[k])

    def get_rel_path(self):
        return self.model_0.get_rel_path()
