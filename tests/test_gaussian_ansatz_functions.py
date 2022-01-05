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

    def test_mvn_pdf_gradient_broadcasting(self, N, n):

        # get N n-dimensional points 
        x = np.random.rand(N, n)

        # gradient of the general mvn pdf evaluated at x (using broadcasting)
        mean = np.full(n, 1)
        cov = 2. * np.eye(n)
        kappa = mvn_pdf_gradient(x, mean, cov)

        # gradient of the general mvn pdf evaluated at x (without broadcasting)

        # random variable with multivariate distribution
        rv = stats.multivariate_normal(mean, cov, allow_singular=False)

        # mvn pdf
        mvn_pdf = rv.pdf(x)

        # covariance matrix inverse
        inv_cov = np.linalg.inv(cov)

        # gradient of the exponential term of the pdf
        grad_exp_term = np.zeros((N, n))
        for i in range(N):
            for j in range(n):
                for k in range(n):
                    grad_exp_term[i, j] += (x[i, j] - mean[j]) * (inv_cov[j, k] + inv_cov[k, j])
        grad_exp_term *= - 0.5

        grad_mvn_pdf = grad_exp_term * mvn_pdf[:, np.newaxis]

        assert kappa.shape == grad_mvn_pdf.shape
        assert np.isclose(kappa, grad_mvn_pdf).all()

    #@pytest.mark.skip()
    def test_mvn_pdf_basis_1d_1m(self, N, beta):
        '''
        '''
        # get N 1-dimensional points 
        n = 1
        x = np.random.rand(N, n)

        # get 1 centers of the gaussian
        m = 1
        means = np.random.rand(m, n)

        # fix covariance matrix
        cov = 0.5 * np.eye(n)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta)

        # set gaussians
        ansatz.set_given_ansatz_functions(means, cov)

        # get basis of gaussian functions evaluated at x
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x, means, cov)

        # since our basis just have one gaussian
        gaussian = mvn_pdf_basis[:, 0]

        # evaluate 1d gaussian at x
        gaussian_test = self.normal_pdf(x, mu=means[0, 0], sigma=np.sqrt(cov[0, 0]))[:, 0]

        assert gaussian.shape == gaussian_test.shape
        assert np.isclose(gaussian, gaussian_test).all()

    def test_mvn_pdf_basis_broadcasting(self, N, n, m, beta):

        # get N n-dimensional points 
        x = np.random.rand(N, n)

        # get m diferent centers of the gaussians
        means = np.random.rand(m, n)

        # fix same covariance matrix
        cov = 2 * np.eye(n)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta)

        # set gaussians
        ansatz.set_given_ansatz_functions(means, cov)

        # get basis of gaussian functions evaluated at x (using broadcasting)
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x, means, cov)

        # get basis of gaussian functions evaluated at x (without broadcasting)

        # covariance matrix inverse
        inv_cov = np.linalg.inv(cov)

        # compute exp term
        exp_term = np.zeros((N, m))
        for i in range(N):
            for l in range(m):
                for j in range(n):
                    for k in range(n):
                        exp_term[i, l] += (x[i, j] - means[l, j]) \
                                        * inv_cov[j, k] \
                                        * (x[i, k] - means[l, k])
        exp_term *= - 0.5


        # compute norm factor
        norm_factor = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(cov))

        # normalize
        mvn_pdf_basis_test = np.exp(exp_term) / norm_factor

        assert mvn_pdf_basis.shape == mvn_pdf_basis_test.shape
        assert np.isclose(mvn_pdf_basis, mvn_pdf_basis_test).all()


    def test_mvn_pdf_gradient_basis_broadcasting(self, N, n, m, beta):

        # get N n-dimensional points 
        x = np.random.rand(N, n)

        # get m diferent centers of the gaussians
        means = np.random.rand(m, n)

        # fix same covariance matrix
        cov = 2 * np.eye(n)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta)

        # set gaussians
        ansatz.set_given_ansatz_functions(means, cov)

        # get basis of gradient of gaussian functions evaluated at x (using broadcasting)
        mvn_pdf_gradient_basis = ansatz.mvn_pdf_gradient_basis(x, means, cov)

        # get basis of gradient of gaussian functions evaluated at x (without broadcasting)

        # get nd gaussian basis
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x, means, cov)

        # covariance matrix inverse
        inv_cov = np.linalg.inv(cov)

        # prepare position and means for broadcasting

        # compute gradient of the exponential term
        exp_term_gradient = np.zeros((N, m, n))
        for i in range(N):
            for l in range(m):
                for j in range(n):
                    for k in range(n):
                            exp_term_gradient[i, l, j] += (x[i, j] - means[l, j]) \
                                                        * (inv_cov[j, k] + inv_cov[k, j])
        exp_term_gradient *= - 0.5

        # compute gaussian gradients basis
        mvn_pdf_gradient_basis_test = exp_term_gradient * mvn_pdf_basis[:, :, np.newaxis]

        assert mvn_pdf_gradient_basis.shape == mvn_pdf_gradient_basis_test.shape
        assert np.isclose(mvn_pdf_gradient_basis, mvn_pdf_gradient_basis_test).all()

