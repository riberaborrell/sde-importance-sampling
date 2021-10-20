from mds.functions import constant, quadratic_one_well, double_well, double_well_gradient

import functools
import numpy as np
import pytest

class TestQuadraticOneWell:

    def test_1d_quadratic_one_well(self):

        # get function
        nu = 2.
        f = functools.partial(quadratic_one_well, nu=nu)

        # test points
        x = -1.
        assert f(x) == nu * 4
        x = 0.
        assert f(x) == nu
        x = 1.
        assert f(x) == 0

    def test_nd_quadratic_one_well(self):

        # set n
        n = 5

        # get function
        nu = np.random.rand(n)
        f = functools.partial(quadratic_one_well, nu=nu)

        # test points
        x = -1 * np.ones(n)
        assert f(x) == 4 * np.sum(nu)
        x = np.zeros(n)
        assert f(x) == np.sum(nu)
        x = np.ones(n)
        assert f(x) == 0

        # test vectorization
        x = np.random.rand(n)
        y_test = 0
        for i in range(n):
            y_test += nu[i] * (x[i] - 1)**2
        assert f(x) == y_test

    def test_nd_vec_quadratic_one_well(self):

        # set n nd N
        n = 3
        N = 10

        # get function
        nu = np.random.rand(n)
        f = functools.partial(quadratic_one_well, nu=nu)

        # test points
        x = np.random.rand(N, n)
        x[0] = np.ones(n)
        x[1] = np.zeros(n)
        x[2] = -1 * np.ones(n)
        y = f(x)
        assert y.ndim == 1
        assert y.shape[0] == N

        assert y[0] == 0
        assert y[1] == np.sum(nu)
        assert y[2] == 4 * np.sum(nu)

        # test vectorization
        x = np.random.rand(N, n)
        y_test = np.zeros(N)
        for i in range(n):
            y_test += nu[i] * (x[:, i] - 1)**2
        assert (f(x) == y_test).all(), ''

class TestDoubleWell:

    def test_1d_double_well(self):

        # get function
        alpha = 2.
        f = functools.partial(double_well, alpha=alpha)

        # test points
        x = - 1.
        assert f(x) == 0
        x = 0.
        assert f(x) == alpha
        x = 1.
        assert f(x) == 0

    def test_nd_double_well(self):

        # set n
        n = 5

        # get function
        alpha = np.random.rand(n)
        f = functools.partial(double_well, alpha=alpha)

        # test points
        x = -1 * np.ones(n)
        assert f(x) == 0
        x = np.zeros(n)
        assert f(x) == np.sum(alpha)
        x = np.ones(n)
        assert f(x) == 0

        # test vectorization
        x = np.random.rand(n)
        y_test = 0
        for i in range(n):
            y_test += alpha[i] * (x[i]**2 - 1)**2
        assert f(x) == y_test

    def test_nd_vec_double_well(self):

        # set n nd N
        n = 3
        N = 10

        # get function
        alpha = np.random.rand(n)
        f = functools.partial(double_well, alpha=alpha)

        # test points
        x = np.random.rand(N, n)
        x[0] = -1 * np.ones(n)
        x[1] = np.zeros(n)
        x[2] = np.ones(n)
        y = f(x)
        assert y.ndim == 1
        assert y.shape[0] == N

        assert y[0] == 0
        assert y[1] == np.sum(alpha)
        assert y[2] == 0

        # test vectorization
        x = np.random.rand(N, n)
        y_test = np.zeros(N)
        for i in range(n):
            y_test += alpha[i] * (x[:, i]**2 - 1)**2
        assert (f(x) == y_test).all()

class TestDoubleWellGradient:

    def test_1d_double_well_gradient(self):

        # get function
        alpha = 2.
        f = functools.partial(double_well_gradient, alpha=alpha)

        # test points
        x = - 1.
        assert f(x) == 0
        x = - 0.5
        assert f(x) == alpha * 3 / 2
        x = 0.
        assert f(x) == 0
        x = 0.5
        assert f(x) == - alpha * 3 / 2
        x = 1.
        assert f(x) == 0

    def test_nd_double_well_gradient(self):

        # set n
        n = 5

        # get function
        alpha = np.random.rand(n)
        f = functools.partial(double_well_gradient, alpha=alpha)

        # test points
        x = -1 * np.ones(n)
        assert (f(x) == 0).all()
        x = - 0.5 * np.ones(n)
        assert (f(x) == alpha * 3 / 2).all()
        x = np.zeros(n)
        assert (f(x) == 0).all()
        x = 0.5 * np.ones(n)
        assert (f(x) == - alpha * 3 / 2).all()
        x = 1. * np.ones(n)
        assert (f(x) == 0).all()

        # test vectorization
        x = np.random.rand(n)
        y = f(x)
        assert y.ndim == 1
        assert y.shape[0] == n
        y_test = np.empty(n)
        for i in range(n):
            y_test[i] = 4 * alpha[i] * x[i] * (x[i]**2 - 1)
        assert (y == y_test).all()

    def test_nd_vec_double_well_gradient(self):

        # set n nd N
        n = 3
        N = 10

        # get function
        alpha = np.random.rand(n)
        f = functools.partial(double_well_gradient, alpha=alpha)

        # test points
        x = np.random.rand(N, n)
        x[0] = - np.ones(n)
        x[1] = -0.5 * np.ones(n)
        x[2] = np.zeros(n)
        x[3] = 0.5 * np.ones(n)
        x[4] = np.ones(n)
        y = f(x)
        assert y.ndim == 2
        assert y.shape[0] == N
        assert y.shape[1] == n

        assert (y[0] == 0).all()
        assert (y[1] == alpha * 3 / 2).all()
        assert (y[2] == 0).all()
        assert (y[3] == - alpha * 3 / 2).all()
        assert (y[4] == 0).all()

        # test vectorization
        x = np.random.rand(N, n)
        y_test = np.empty((N, n))
        for i in range(n):
            y_test[:, i] = 4 * alpha[i] * x[:, i] * (x[:, i]**2 - 1)
        assert (f(x) == y_test).all(), ''
