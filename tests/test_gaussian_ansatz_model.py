import time

import numpy as np
import pytest
import torch

from sde.langevin_sde import LangevinSDE
from function_approximation.models import GaussianAnsatzModel


class TestGaussianAnsatzModel:

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
    def random_inputs(self, K, d):
        ''' generates random input data
        '''
        return torch.rand(K, d)

    @pytest.fixture
    def gaussian_ansatz(self, sde, m_i, sigma_i):
        ''' initializes GaussianAnsatzModel and distributes uniformly the means
        '''
        return GaussianAnsatzModel(sde, m_i, sigma_i)

    def test_mvn_pdf_basis_size(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        basis = model.mvn_pdf_basis(random_inputs)

        K = random_inputs.size(0)
        assert basis.size() == (K, model.m)

    def test_mvn_pdf_gradient_basis_size(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        basis = model.mvn_pdf_gradient_basis(random_inputs)

        K = random_inputs.size(0)
        assert basis.size() == (K, model.m, model.d)

    def test_model_size(self, gaussian_ansatz, random_inputs):
        '''
        '''
        model = gaussian_ansatz
        output = model.forward(random_inputs)

        K = random_inputs.size(0)
        assert output.size() == (K, model.d)

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
        K = random_inputs.size(0)
        Jac = torch.empty(K, model.m)

        # get nn parameters
        A = model._modules['linear']._parameters['weight']

        for i in range(K):

            # backward with vector-Jacobian product
            v = torch.eye(K)[i].reshape(K, 1)
            output.backward(v, retain_graph=True)

            # save gradients
            Jac[i, :] = A.grad[0, :]

            # reset gradients
            A.grad.zero_()

