from sde_importance_sampling.neural_networks import GaussianAnsatzNN

import numpy as np
import pytest
import torch

class TestGaussianAnsatzNN:

    @pytest.fixture
    def random_inputs(self, N, n):
        '''generates random input data'''
        return torch.rand(N, n)

    @pytest.fixture
    def random_distr_gaussian_ansatz_nn(self, n, m):
        '''initializes GaussianAnsatz'''
        return GaussianAnsatzNN(n, m)

    def test_mvn_pdf_basis_size(self, random_distr_gaussian_ansatz_nn, random_inputs):
        '''evaluates vectorized mvn pdf'''
        model = random_distr_gaussian_ansatz_nn
        basis = model.mvn_pdf_basis(random_inputs)

        N = random_inputs.size(0)
        assert basis.size() == (N, model.m)

    def test_model_size(self, random_distr_gaussian_ansatz_nn, random_inputs):
        '''evaluates gaussian ansatz nn'''
        model = random_distr_gaussian_ansatz_nn
        output = model.forward(random_inputs)

        N = random_inputs.size(0)
        assert output.size() == (N, 1)

    def mvn_pdf_gradient_basis(self, model, x):
        ''' Gradient of the multivariate normal pdf (nd Gaussian) \nabla v(x; means, cov)
        with means evaluated at x
            x ((N, n)-array) : posicion
        '''
        # this method is not used

        # assume shape of x array to be (N, n)
        assert x.ndim == 2, ''
        assert x.size(1) == model.n, ''
        N = x.size(0)

        # get nd gaussian basis
        mvn_pdf_basis = model.mvn_pdf_basis(x)

        # covariance matrix inverse
        inv_cov = np.linalg.inv(model.cov)

        # prepare position and means for broadcasting
        x = x[:, np.newaxis, :]
        means = model.means[np.newaxis, :, :]

        # compute gradient of the exponential term
        exp_term_gradient = - 0.5 * np.matmul(x - means, inv_cov + inv_cov.T)

        # compute gaussian gradients basis
        mvn_pdf_gradient_basis = exp_term_gradient * mvn_pdf_basis[:, :, np.newaxis]

        return mvn_pdf_gradient_basis

    @pytest.mark.skip()
    def test_mvn_pdf_gradient_basis(self, random_distr_gaussian_ansatz_nn, random_inputs):

        # evaluates gaussian ansatz nn model
        model = random_distr_gaussian_ansatz_nn
        output = model.forward(random_inputs)

        # preallocate Jacobian matrix
        N = random_inputs.size(0)
        Jac = torch.empty(N, model.m)

        # get nn parameters
        A = model._modules['linear']._parameters['weight']

        for i in range(N):

            # backward with vector-Jacobian product
            v = torch.eye(N)[i].reshape(N, 1)
            output.backward(v, retain_graph=True)

            # save gradients
            Jac[i, :] = A.grad[0, :]

            # reset gradients
            A.grad.zero_()

