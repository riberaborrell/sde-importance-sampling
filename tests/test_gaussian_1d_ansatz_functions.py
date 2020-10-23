from mds.gaussian_1d_ansatz_functions import GaussianAnsatz

import pytest
import unittest
import numpy as np

class TestGaussianAnsatz(unittest.TestCase):
    def _makeOne(self, domain, m, sigma):
        ansatz = GaussianAnsatz(domain, m)
        ansatz.set_unif_dist_ansatz_functions(sigma)
        return ansatz

    def test_vectorization(self):
        m = 50
        ansatz = self._makeOne(
            domain=np.array([-3, 3]),
            m=m,
            sigma=1,
        )
        M = 100
        x = np.random.uniform(-3, 3, M)
        v = ansatz.basis_control(x)
        self.assertEqual(v.shape, (M, m))
