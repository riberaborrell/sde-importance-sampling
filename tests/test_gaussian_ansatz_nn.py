from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.neural_networks import GaussianAnsatzNN

import numpy as np
import pytest
import torch

import time

class TestGaussianAnsatzNN:

    @pytest.fixture
    def sde(self, problem_name, potential_name, n, alpha_i, beta):
        ''' creates Langevin SDE object with the given setting.
        '''
        sde = LangevinSDE(
            problem_name=problem_name,
            potential_name=potential_name,
            n=n,
            alpha=np.full(n, alpha_i),
            beta=beta,
        )
        return sde

    @pytest.fixture
    def random_inputs(self, N, n):
        '''generates random input data'''
        return torch.rand(N, n)

    @pytest.fixture
    def gaussian_ansatz(self, sde, m_i, sigma_i):
        '''initializes GaussianAnsatzNN model and distributes uniformly the means'''
        return GaussianAnsatzNN(sde.n, sde.beta, sde.domain, m_i, sigma_i)

    def test_mvn_pdf_basis_size(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        basis = model.mvn_pdf_basis(random_inputs)

        N = random_inputs.size(0)
        assert basis.size() == (N, model.m)

    def test_mvn_pdf_gradient_basis_size(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        basis = model.mvn_pdf_gradient_basis(random_inputs)

        N = random_inputs.size(0)
        assert basis.size() == (N, model.m, model.n)

    def test_model_size(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        output = model.forward(random_inputs)

        N = random_inputs.size(0)
        assert output.size() == (N, model.n)

    def test_model_get_parameters(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        theta = model.get_parameters()

        assert theta.ndim == 1
        assert theta.shape[0] == model.m

    def test_model_load_parameters(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        theta_old = model.get_parameters()

        theta_new = np.random.rand(model.m)
        model.load_parameters(theta_new)

        theta = model.get_parameters()
        assert np.isclose(theta_new, theta).all()
        assert not np.isclose(theta_old, theta).all()

    def test_mvn_pdf_basis_ct(self, random_inputs, gaussian_ansatz):

        # start timer
        ct_initial = time.perf_counter()

        # compute mvn basis
        basis = gaussian_ansatz.mvn_pdf_basis(random_inputs)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_mvn_pdf_gradient_basis_ct(self, random_inputs, gaussian_ansatz):

        # start timer
        ct_initial = time.perf_counter()

        # compute mvn basis
        basis = gaussian_ansatz.mvn_pdf_gradient_basis(random_inputs)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    @pytest.mark.skip()
    def test_mvn_pdf_gradient_basis(self, random_distr_gaussian_ansatz_nn, random_inputs):
        '''
        '''
        #TODO:revise!

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

