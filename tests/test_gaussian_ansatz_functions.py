from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz

import numpy as np
import pytest

class TestGaussianAnsatzFunctions:

    @pytest.fixture
    def random_inputs(self, N, n):
        '''generates random input data
        '''
        return np.rand(N, n)

    @pytest.fixture
    def gaussian_ansatz(self, n, beta):
        '''initializes GaussianAnsatz
        '''
        return GaussianAnsatz(n, beta)

    def normal_pdf(self, x, mu=0, sigma=1):
        norm_factor = np.sqrt(2 * np.pi) * sigma
        return np.exp(-0.5 * ((x - mu) / sigma) **2 ) / norm_factor

    def derivative_normal_pdf(self, x, mu=0, sigma=1):
        return stats.norm.pdf(x, mu, sigma) * (mu - x) / sigma**2

    def test_1d_gaussian_ansatz(self, gaussian_ansatz):
        '''
        '''
        # create 1d grid
        h = 0.1
        x = np.arange(-3, 3 + h, h)
        x_exp = np.expand_dims(x, axis=1)

        # evaluate 1d multivariate normal probability distribution function
        means = np.array([[0.]])
        cov = 1 * np.eye(1)
        Z1 = gaussian_ansatz.mvn_pdf_basis(x_exp, means, cov)[:, 0]

        # evaluate 1d normal pdf
        Z2 = self.normal_pdf(x, mu=0, sigma=1)

        assert Z1.shape == Z2.shape
        assert np.isclose(Z1, Z2).all()

