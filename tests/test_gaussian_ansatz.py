import time

import numpy as np
import pytest
import scipy.stats as stats
import torch

from sde.functions import mvn_pdf, mvn_pdf_gradient
from sde.langevin_sde import LangevinSDE
from function_approximation.gaussian_ansatz import GaussianAnsatz
from utils.paths import get_tests_plots_dir


class TestGaussianAnsatzFunctions:

    @pytest.fixture
    def dir_path(self):
        ''' returns dir path for the test plots
        '''
        return get_tests_plots_dir()

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
        '''generates random input data
        '''
        return np.random.rand(K, d)

    @pytest.fixture
    def random_means(self, m, d):
        '''generates centers for the Gaussian ansatz
        '''
        return np.random.rand(m, d)

    @pytest.fixture
    def ansatz(self, sde, random_means):
        '''initializes GaussianAnsatz with the given centers
        '''

        # initialize gaussian ansatz
        #ansatz = GaussianAnsatz(sde, normalized=True)
        ansatz = GaussianAnsatz(sde, normalized=False)

        # scalar covariance matrix
        sigma_i = 1.

        # covariance matrix
        cov = 1 * np.eye(sde.d)

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


    def test_1d_mvn_pdf(self, K):

        # get N 1-dimensional points 
        x = np.random.rand(K, 1)

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


    def test_1d_mvn_pdf_gradient(self, K):

        # get N 1-dimensional points 
        x = np.random.rand(K, 1)

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


    def test_mvn_pdf_gradient_broadcasting(self, K, d):

        # get N n-dimensional points 
        x = np.random.rand(K, d)

        # gradient of the general mvn pdf evaluated at x (using broadcasting)
        mean = np.full(d, 1)
        cov = 2. * np.eye(d)
        kappa = mvn_pdf_gradient(x, mean, cov)

        # gradient of the general mvn pdf evaluated at x (without broadcasting)

        # random variable with multivariate distribution
        rv = stats.multivariate_normal(mean, cov, allow_singular=False)

        # mvn pdf
        mvn_pdf = rv.pdf(x)

        # covariance matrix inverse
        inv_cov = np.linalg.inv(cov)

        # gradient of the exponential term of the pdf
        grad_exp_term = np.zeros((K, d))
        for i in range(K):
            for j in range(d):
                for k in range(d):
                    grad_exp_term[i, j] += (x[i, j] - mean[j]) * (inv_cov[j, k] + inv_cov[k, j])
        grad_exp_term *= - 0.5

        grad_mvn_pdf = grad_exp_term * mvn_pdf[:, np.newaxis]

        assert kappa.shape == grad_mvn_pdf.shape
        assert np.isclose(kappa, grad_mvn_pdf).all()


    @pytest.mark.skip()
    def test_mvn_pdf_basis_1d_1m(self, sde, K):
        '''
        '''

        # set dimension
        assert sde.d == 1

        # get N 1-dimensional points 
        x = np.random.rand(K, sde.d)

        # get 1 centers of the gaussian
        m = 1
        means = np.random.rand(m, sde.d)

        # scalar covariance matrix
        sigma_i = 0.5

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(sde)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, sigma_i=sigma_i)

        # get basis of gaussian functions evaluated at x
        mvn_pdf_basis = ansatz.mvn_pdf_basis(x)

        # since our basis just have one gaussian
        gaussian = mvn_pdf_basis[:, 0]

        # evaluate 1d gaussian at x
        gaussian_test = self.normal_pdf(x, mu=means[0, 0], sigma=np.sqrt(sigma_i))[:, 0]

        assert gaussian.shape == gaussian_test.shape
        assert np.isclose(gaussian, gaussian_test).all()


    def test_mvn_pdf_basis_broadcasting(self, random_inputs, ansatz):

        #  K d-dimensional points 
        x = random_inputs
        K, d = random_inputs.shape

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
        exp_term = np.zeros((K, m))
        for i in range(K):
            for l in range(m):
                for j in range(d):
                    for k in range(d):
                        exp_term[i, l] += (x[i, j] - means[l, j]) \
                                        * inv_cov[j, k] \
                                        * (x[i, k] - means[l, k])
        exp_term *= - 0.5


        # compute norm factor
        if ansatz.normalized:
            norm_factor = np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov))

            # normalize
            mvn_pdf_basis_test = np.exp(exp_term) / norm_factor

        else:
            mvn_pdf_basis_test = np.exp(exp_term)

        assert mvn_pdf_basis.shape == mvn_pdf_basis_test.shape
        assert np.isclose(mvn_pdf_basis, mvn_pdf_basis_test).all()


    def test_mvn_pdf_gradient_basis_broadcasting(self, random_inputs, ansatz):

        #  K d-dimensional points 
        x = random_inputs
        K, d = random_inputs.shape

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
        exp_term_gradient = np.zeros((K, m, d))
        for i in range(K):
            for l in range(m):
                for j in range(d):
                    for k in range(d):
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

    def test_mvn_pdf_basis_torch_ct(self, d, K, m):

        # start timer
        ct_initial = time.perf_counter()

        x = torch.randn([K, d])
        means = torch.randn([m, d])
        sigma_i = 1.0

        log_p = - 0.5 * torch.sum((means.view(1, m, d) - x.view(K, 1, d))**2, 2) / sigma_i - torch.log(2 * torch.tensor(np.pi) * sigma_i) * d / 2
        p_evaluated = torch.sum(torch.exp(log_p), 1)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.3f}'.format(ct_final - ct_initial))

    def test_mvn_pdf_basis_numpy_ct(self, d, K, m):

        # start timer
        ct_initial = time.perf_counter()

        x = np.random.randn(K, d)
        means = np.random.randn(m, d)
        sigma_i = 1.0

        # prepare position and means for broadcasting
        #x = x[:, np.newaxis, :]
        #means = means[np.newaxis, :, :]

        # log p
        exp_term = - 0.5 * np.sum((x[:, np.newaxis, :] - means[np.newaxis, :, :])**2, axis=2) / sigma_i

        # normalize
        norm_factor = np.sqrt(((2 * np.pi * sigma_i) ** d))
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


    @pytest.mark.skip()
    def test_mvn_pdf_1d_plot(self, sde, dir_path):

        # set dimension
        assert sde.d == 1

        # set 1 center
        means = np.zeros((1, sde.d))

        # covariance matrix
        cov = 0.5 * np.eye(sde.d)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(sde, normalized=False)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # plot
        ansatz.dir_path = dir_path
        ansatz.plot_1d_multivariate_normal_pdf(domain=np.array([[-3, 3]]))


    @pytest.mark.skip()
    def test_mvn_pdf_2d_plot(self, sde, dir_path):

        # set dimension
        assert sde.d == 2

        # set 1 center
        means = np.zeros((1, sde.d))

        # covariance matrix
        cov = 0.5 * np.eye(sde.d)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(sde, normalized=False)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # plot
        ansatz.dir_path = dir_path
        ansatz.plot_2d_multivariate_normal_pdf(domain=np.array([[-3, 3], [-3, 3]]))


    @pytest.mark.skip()
    def test_mvn_pdf_nd_plot(self, sde, dir_path):

        # set dimension
        d = 10

        # set 1 center

        # set 1 center
        means = np.zeros((1, sde.d))

        # covariance matrix
        cov = 0.5 * np.eye(sde.d)

        # initialize gaussian ansatz
        ansatz = GaussianAnsatz(sde, normalized=False)

        # set gaussians
        ansatz.set_given_ansatz_functions(means=means, cov=cov)

        # plot
        ansatz.dir_path = dir_path
        ansatz.plot_nd_multivariate_normal_pdf(i=0)
        ansatz.plot_nd_multivariate_normal_pdf(i=1)
