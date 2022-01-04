from sde_importance_sampling.functions import mvn_pdf, mvn_pdf_gradient
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz

import numpy as np
import scipy.stats as stats
import pytest

class TestGaussianAnsatzFunctions:

    @pytest.fixture
    def random_inputs(self, N, n):
        '''generates random input data
        '''
        return np.random.rand(N, n)

    @pytest.fixture
    def gaussian_ansatz(self, n, beta):
        '''initializes GaussianAnsatz
        '''
        return GaussianAnsatz(n, beta)

    def normal_pdf(self, x, mu, sigma):
        norm_factor = np.sqrt(2 * np.pi) * sigma
        return np.exp(-0.5 * ((x - mu) / sigma) **2 ) / norm_factor

    def normal_pdf_derivative(self, x, mu, sigma):
        return stats.norm.pdf(x, mu, sigma) * (mu - x) / sigma**2

    def test_1d_mvn_pdf(self, N):

        # get N 1-dimensional points 
        x = np.random.rand(N, 1)

        # general mvn pdf evaluated at x
        mu = 0.
        sigma = 1.
        mean = np.array([mu])
        cov = np.array([[sigma**2]])
        nu = mvn_pdf(x, mean, cov)

        # 1d gaussian evaluated at x
        nu_test = self.normal_pdf(x, mu, sigma).squeeze()

        assert nu.shape == nu_test.shape
        assert np.isclose(nu, nu_test).all()

    def test_1d_mvn_pdf_gradient(self, N):

        # get N 1-dimensional points 
        x = np.random.rand(N, 1)

        # gradient of the general mvn pdf evaluated at x
        mu = 0.
        sigma = 1.
        mean = np.array([mu])
        cov = np.array([[sigma**2]])
        kappa = mvn_pdf_gradient(x, mean, cov)

        # derivative of 1d gaussian evaluated at x
        kappa_test = self.normal_pdf_derivative(x, mu, sigma)

        assert kappa.shape == kappa_test.shape
        assert np.isclose(kappa, kappa_test).all()

    def test_vectorized_mvn_pdf_gradient(self, N, n):

        # get N n-dimensional points 
        x = np.random.rand(N, n)

        # gradient of the general mvn pdf evaluated at x using vectorization
        mean = np.full(n, 1)
        cov = 2. * np.eye(n)
        kappa = mvn_pdf_gradient(x, mean, cov)

        # random variable with multivariate distribution
        rv = stats.multivariate_normal(mean, cov, allow_singular=False)

        # mvn pdf
        mvn_pdf = rv.pdf(x)

        # covariance matrix inverse
        inv_cov = np.linalg.inv(cov)

        # gradient of the exponential term of the pdf
        grad_exp_term = np.zeros((N, n))
        for i in range(n):
            for j in range(n):
                grad_exp_term[:, i] += (x[:, i] - mean[i]) * (inv_cov[i, j] + inv_cov[j, i])
        grad_exp_term *= - 0.5

        grad_mvn_pdf = grad_exp_term * mvn_pdf[:, np.newaxis]

        assert kappa.shape == grad_mvn_pdf.shape
        assert np.isclose(kappa, grad_mvn_pdf).all()

    @pytest.mark.skip()
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

