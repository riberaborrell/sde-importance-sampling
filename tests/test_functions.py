from mds.functions import constant, quadratic_one_well

import functools
import numpy as np
import pytest

class TestFunctions:

    def test_1d_quadratic_one_well(self):

        # test points
        nu = 2.
        f = functools.partial(quadratic_one_well, nu=nu)

        x = 1.
        y = f(x)
        assert y == 0, ''

        x = 0.
        y = f(x)
        assert y == nu, ''

        x = -1.
        y = f(x)
        assert y == nu * 4, ''

    def test_nd_quadratic_one_well(self):

        # set n
        n = 5

        # test points
        nu = np.ones(n)
        f = functools.partial(quadratic_one_well, nu=nu)

        x = np.ones(n)
        y = f(x)
        assert y == 0, ''

        x = np.zeros(n)
        y = f(x)
        assert y == np.sum(nu), ''

        x = -1 * np.ones(n)
        y = f(x)
        assert y == 4 * np.sum(nu), ''


        # test vectorization
        nu = np.random.rand(n)
        f = functools.partial(quadratic_one_well, nu=nu)
        x = np.random.rand(n)
        y = f(x)
        y_test = 0
        for i in range(n):
            y_test += nu[i] * (x[i] - 1)**2
        assert y == y_test, ''

    def test_nd_vec_quadratic_one_well(self):

        # set n nd N
        n = 3
        N = 10

        # test points
        nu = np.ones(n)
        f = functools.partial(quadratic_one_well, nu=nu)
        x = np.random.rand(N, n)
        x[0] = np.ones(n)
        x[1] = np.zeros(n)
        x[2] = -1 * np.ones(n)
        y = f(x)
        assert y.ndim == 1, ''
        assert y.shape[0] == N, ''

        assert y[0] == 0, ''
        assert y[1] == np.sum(nu), ''
        assert y[2] == 4 * np.sum(nu), ''

        # test vectorization
        nu = np.random.rand(n)
        f = functools.partial(quadratic_one_well, nu=nu)
        x = np.random.rand(N, n)
        y = f(x)
        y_test = np.zeros(N)
        for i in range(n):
            y_test += nu[i] * (x[:, i] - 1)**2
        assert (y == y_test).all(), ''
