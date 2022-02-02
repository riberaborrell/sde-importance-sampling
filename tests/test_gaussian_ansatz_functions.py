from sde_importance_sampling.functions import mvn_pdf, mvn_pdf_gradient
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz
from sde_importance_sampling.utils_path import get_tests_plots_dir

import numpy as np
import scipy.stats as stats
import torch
import pytest

import time

class TestGaussianAnsatzFunctions:

    @pytest.fixture
    def dir_path(self):
        ''' returns dir path for the test plots
        '''
        return get_tests_plots_dir()


    @pytest.fixture
    def random_inputs(self, N, n):
        '''generates random input data
        '''
        return np.random.rand(N, n)

    @pytest.fixture
    def random_means(self, m, n):
        '''generates centers for the Gaussian ansatz
        '''
        return np.random.rand(m, n)

    @pytest.fixture
    def ansatz(self, n, beta, random_means):
        '''initializes GaussianAnsatz with the given centers
        '''

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta)

        # scalar covariance matrix
        sigma_i = 1.

        # covariance matrix
        cov = 1 * np.eye(n)

        # set gaussians
        #ansatz.set_given_ansatz_functions(means=random_means, cov=cov)
        ansatz.set_given_ansatz_functions(means=random_means, sigma_i=sigma_i)

        # set weights
        ansatz.theta = np.random.rand(ansatz.m)

        return ansatz


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
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # get basis of gaussian functions evaluated at x
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x)

        # since our basis just have one gaussian
        gaussian = mvn_pdf_basis[:, 0]

        # evaluate 1d gaussian at x
        gaussian_test = self.normal_pdf(x, mu=means[0, 0], sigma=np.sqrt(cov[0, 0]))[:, 0]

        assert gaussian.shape == gaussian_test.shape
        assert np.isclose(gaussian, gaussian_test).all()


    def test_mvn_pdf_basis_broadcasting(self, random_inputs, ansatz):

        #  N n-dimensional points 
        x = random_inputs
        N, n = random_inputs.shape

        # get basis of gaussian functions evaluated at x (using broadcasting)
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x)

        # get basis of gaussian functions evaluated at x (without broadcasting)

        # m centers of the gaussians
        means = ansatz.means
        m = means.shape[0]

        # covariance matrix
        cov = ansatz.cov

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


    def test_mvn_pdf_gradient_basis_broadcasting(self, random_inputs, ansatz):

        #  N n-dimensional points 
        x = random_inputs
        N, n = random_inputs.shape

        # get basis of gradient of gaussian functions evaluated at x (using broadcasting)
        mvn_pdf_gradient_basis = ansatz.mvn_pdf_gradient_basis(x)

        # get basis of gradient of gaussian functions evaluated at x (without broadcasting)

        # get m diferent centers of the gaussians
        means = ansatz.means
        m = means.shape[0]

        # covariance matrix
        cov = ansatz.cov

        # get nd gaussian basis
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x)

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

    def test_mvn_pdf_basis_ct(self, random_inputs, ansatz):

        # start timer
        ct_initial = time.perf_counter()

        # compute mvn basis
        basis = ansatz.mvn_pdf_basis(random_inputs)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_mvn_pdf_basis_torch_ct(self, n, N, m):

        # start timer
        ct_initial = time.perf_counter()

        x = torch.randn([N, n])
        means = torch.randn([m, n])
        sigma_i = 1.0

        log_p = - 0.5 * torch.sum((means.view(1, m, n) - x.view(N, 1, n))**2, 2) / sigma_i - torch.log(2 * torch.tensor(np.pi) * sigma_i) * n / 2
        p_evaluated = torch.sum(torch.exp(log_p), 1)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_mvn_pdf_basis_numpy_ct(self, n, N, m):

        # start timer
        ct_initial = time.perf_counter()

        x = np.random.randn(N, n)
        means = np.random.randn(m, n)
        sigma_i = 1.0

        # prepare position and means for broadcasting
        #x = x[:, np.newaxis, :]
        #means = means[np.newaxis, :, :]

        # log p
        exp_term = - 0.5 * np.sum((x[:, np.newaxis, :] - means[np.newaxis, :, :])**2, axis=2) / sigma_i

        # normalize
        norm_factor = np.sqrt(((2 * np.pi * sigma_i) ** n))
        mvn_pdf_basis = np.exp(exp_term) / norm_factor

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_ansatz_value_function_ct(self, random_inputs, ansatz):

        # start timer
        ct_initial = time.perf_counter()

        # compute value function
        ansatz.set_value_function_constant_to_zero()
        value_function = ansatz.value_function(random_inputs)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_mvn_pdf_gradient_basis_ct(self, random_inputs, ansatz):

        # start timer
        ct_initial = time.perf_counter()

        # compute mvn basis
        basis = ansatz.mvn_pdf_gradient_basis(random_inputs)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_ansatz_control_ct(self, random_inputs, ansatz):

        # start timer
        ct_initial = time.perf_counter()

        # compute control
        control = ansatz.control(random_inputs)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))


    def test_mvn_pdf_1d_plot(self, dir_path, beta):

        # set dimension
        n = 1

        # set 1 center
        means = np.zeros((1, n))

        # covariance matrix
        cov = 0.5 * np.eye(n)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta, normalized=False)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # plot
        ansatz.dir_path = dir_path
        ansatz.plot_1d_multivariate_normal_pdf(domain=np.array([[-3, 3]]))


    def test_mvn_pdf_2d_plot(self, dir_path, beta):

        # set dimension
        n = 2

        # set 1 center
        means = np.zeros((1, n))

        # covariance matrix
        cov = 0.5 * np.eye(n)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta, normalized=False)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # plot
        ansatz.dir_path = dir_path
        ansatz.plot_2d_multivariate_normal_pdf(domain=np.array([[-3, 3], [-3, 3]]))


    def test_mvn_pdf_nd_plot(self, dir_path, beta):

        # set dimension
        n = 10

        # set 1 center

        # set 1 center
        means = np.zeros((1, n))

        # covariance matrix
        cov = 0.5 * np.eye(n)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(n, beta, normalized=False)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # plot
        ansatz.dir_path = dir_path
        ansatz.plot_nd_multivariate_normal_pdf(i=0)
        ansatz.plot_nd_multivariate_normal_pdf(i=1)
