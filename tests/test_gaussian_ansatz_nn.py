from mds.neural_networks import GaussianAnsatzNN

import numpy as np
import pytest
import torch

class TestGaussianAnsatzNN:

    @pytest.fixture
    def random_inputs(self, N, n):
        '''generates random input data'''
        return torch.rand(N, n)

    @pytest.fixture
    def random_distr_gaussian_ansatz(self, n, m):
        '''initializes GaussianAnsatz'''
        return GaussianAnsatzNN(n, m, means=None, cov=None)

    def test_vec_mvn_pdf(self, random_distr_gaussian_ansatz, random_inputs):
        '''evaluates vectorized mvn pdf'''
        ansatz = random_distr_gaussian_ansatz
        v = ansatz.mvn_pdf_basis(random_inputs)

        N = random_inputs.size(0)
        assert v.size() == (N, ansatz.m)

    def test_gaussian_ansatz(self, random_distr_gaussian_ansatz, random_inputs):
        '''evaluates gaussian ansatz nn'''
        ansatz = random_distr_gaussian_ansatz
        output = ansatz.forward(random_inputs)

        N = random_inputs.size(0)
        assert output.size() == (N, 1)

    def test_mvn_pdf_gradient_basis(self, random_distr_gaussian_ansatz, random_inputs):

        # evaluates gaussian ansatz
        ansatz = random_distr_gaussian_ansatz
        output = ansatz.forward(random_inputs)

        # preallocate Jacobian matrix
        N = random_inputs.size(0)
        Jac = torch.empty(N, ansatz.m)

        # get nn parameters
        A = ansatz._modules['linear']._parameters['weight']

        for i in range(N):

            # backward with vector-Jacobian product
            v = torch.eye(N)[i].reshape(N, 1)
            output.backward(v, retain_graph=True)

            # save gradients
            Jac[i, :] = A.grad[0, :]

            # reset gradients
            A.grad.zero_()
