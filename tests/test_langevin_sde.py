import pytest
import time

import numpy as np

from sde_importance_sampling.langevin_sde import LangevinSDE


class TestLangevinSDE:

    @pytest.fixture
    def sde_1d(self, problem_name, potential_name, alpha_i, beta):
        ''' creates Langevin SDE object with the given setting.
        '''
        d = 1
        sde = LangevinSDE(
            problem_name=problem_name,
            potential_name=potential_name,
            d=d,
            alpha=np.full(d, alpha_i),
            beta=beta,
        )
        return sde

    @pytest.fixture
    def sde_2d(self, problem_name, potential_name, alpha_i, beta):
        ''' creates Langevin SDE object with the given setting.
        '''
        d = 1
        sde = LangevinSDE(
            problem_name=problem_name,
            potential_name=potential_name,
            d=d,
            alpha=np.full(d, alpha_i),
            beta=beta,
        )
        return sde

    @pytest.fixture
    def sde_nd(self, problem_name, potential_name, d, alpha_i, beta):
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

    def test_discretize_domain_h_1d(self, sde_1d, h):
        ''' test domain discretization.
        '''
        sde = sde_1d

        # discretize domain
        sde.discretize_domain(h)

        # compute size of domain
        domain_h_size = sde.domain_h.nbytes
        domain_h_size = np.around(domain_h_size / 1024 / 1024, 2)

        assert sde.Nh <=10**6

    def test_discretize_domain_h_2d(self, sde_2d, h):
        ''' test domain discretization.
        '''
        sde = sde_2d

        # discretize domain
        sde.discretize_domain(h)

        # compute size of domain
        domain_h_size = sde.domain_h.nbytes
        domain_h_size = np.around(domain_h_size / 1024 / 1024, 2)

        assert sde.Nh <=10**6

    def test_get_flat_domain_h(self, sde_nd):
        sde = sde_nd
        sde.discretize_domain(h=0.1)
        flat_domain_h = sde.get_flat_domain_h()
        assert flat_domain_h.shape == (sde.Nh, sde.d)

    def test_get_idx_target_set(self, sde_nd):
        sde = sde_nd
        sde.discretize_domain(h=0.1)
        flat_domain_h = sde.get_flat_domain_h()
        idx_ts = sde.get_idx_target_set()
        print(flat_domain_h[idx_ts])


    def test_get_idx_ct(self, sde_nd):
        sde = sde_nd

        # sample point uniformly
        x = sde.sample_domain_uniformly(K=1)[0, :]

        # discretize domain
        sde.discretize_domain(h=0.005)

        # start timer
        ct_initial = time.perf_counter()

        # get index
        idx = sde.get_index(x)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.5f}'.format(ct_final - ct_initial))

    def test_get_idx_vectorized(self, sde_nd, K):
        sde = sde_nd

        # sample point uniformly
        x = sde.sample_domain_uniformly(K=K)

        # discretize domain
        sde.discretize_domain_ith_coordinate(h=0.005)

        # get index
        idx = sde.get_index_vectorized(x)


    def test_get_idx_vectorized_ct(self, sde_nd, K):
        sde = sde_nd

        # sample point uniformly
        x = sde.sample_domain_uniformly(K=K)

        # discretize domain
        sde.discretize_domain_ith_coordinate(h=0.005)

        # start timer
        ct_initial = time.perf_counter()

        # get index
        idx = sde.get_index_vectorized(x)

        # stop timer
        ct_final = time.perf_counter()

        print('ct: {:.5f}'.format(ct_final - ct_initial))
