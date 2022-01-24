from sde_importance_sampling.langevin_sde import LangevinSDE

import pytest

import numpy as np

class TestLangevinSDE:

    @pytest.fixture
    def sde(self, problem_name, potential_name, n, alpha_i, beta):
        ''' creates Langevin SDE object with the given setting.
        '''
        sde = LangevinSDE(
            problem_name=problem_name,
            potential_name=potential_name,
            n=n,
            alpha=np.full(n, alpha_i),
            beta=beta,
        )
        return sde

    def test_discretize_domain_h(self, sde, h):
        ''' test domain discretization.
        '''

        # discretize domain
        sde.discretize_domain(h)

        # compute size of domain
        domain_h_size = sde.domain_h.nbytes
        domain_h_size = np.around(domain_h_size / 1024 / 1024, 2)

        assert sde.Nh <=10**6

    def test_get_flat_domain_h(self, sde):
        sde.discretize_domain(h=0.1)
        flat_domain_h = sde.get_flat_domain_h()
        assert flat_domain_h.shape == (sde.Nh, sde.n)

    def test_get_idx_target_set(self, sde):
        sde.discretize_domain(h=0.1)
        flat_domain_h = sde.get_flat_domain_h()
        idx_ts = sde.get_idx_target_set()
        print(flat_domain_h[idx_ts])
