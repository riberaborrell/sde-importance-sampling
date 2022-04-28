from copy import copy, deepcopy

import numpy as np
import pytest

from sampling.importance_sampling import Sampling
from sde.langevin_sde import LangevinSDE


class TestSamplingFht:

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
    def sample(self, sde, dt, k_lim, K):

        # initialize sampling object
        sample = Sampling(sde, is_controlled=False)

        # set sampling and Euler-Marujama parameters
        sample.set_sampling_parameters(
            dt=dt,
            k_lim=k_lim,
            xzero=np.full(sde.d, -1),
            K=K,
        )
        return sample

    #@pytest.mark.skip(reason='')
    def test_not_controlled_sampling(self, sample, seed):
        ''' trajectories are sampled vectorized together until the last time step of the slowest
            trajectory. The corresponding statistics are saved for the fht
        '''
        # fix seed
        np.random.seed(seed)

        sample.start_timer()
        sample.preallocate_fht()

        # initialize xt
        xt = sample.initial_position()

        for k in np.arange(1, sample.k_lim + 1):

            # get indices from the trajectories which are new in target
            idx = sample.get_idx_new_in_target_set(xt)

            # save first hitting time
            if idx.shape[0] != 0:
                sample.fht[idx] = k * sample.dt

            # if all trajectories have arrived in the target set
            if sample.been_in_target_set.all() == True:

                # save number of time steps used
                sample.k = k
                break

            # compute gradient
            gradient = sample.sde.gradient(xt)

            # get Brownian increment
            dbt = sample.brownian_increment()

            # sde update
            xt = sample.sde_update(xt, gradient, dbt)

        sample.stop_timer()

        # report
        msg = 'mean fht: {:2.2f}, ct: {:2.2f}' \
              ''.format(np.mean(sample.fht), sample.ct)
        print(msg)

    #@pytest.mark.skip(reason='')
    def test_not_controlled_sampling_stopped_process(self, sample, seed):
        ''' trajectories are sampled vectorized together but they are not updated
            anymore when they reach the target set
        '''
        # fix seed
        np.random.seed(seed)

        sample.start_timer()

        # initialize fht array
        sample.fht = np.zeros(sample.K)

        # initialize xt
        xt = sample.initial_position()

        for k in np.arange(1, sample.k_lim + 1):

            # not in the target set
            idx = (
                (xt < sample.sde.target_set[:, 0]) |
                (xt > sample.sde.target_set[:, 1])
            ).any(axis=1).reshape(sample.K,)

            # number of trajectory which are not in the target set
            K_idx = np.sum(idx)

            # if all trajectories are in the target set
            if K_idx == 0:

                # save number of time steps used
                sample.k = k
                break

            # compute gradient
            gradient = sample.sde.gradient(xt[idx])

            # get Brownian increment
            dbt = np.sqrt(sample.dt) \
                * np.random.normal(0, 1, K_idx * sample.sde.d).reshape(K_idx, sample.sde.d)

            # sde update
            xt[idx] = sample.sde_update(xt[idx], gradient, dbt)

            # update first hitting time
            sample.fht[idx] += sample.dt

        sample.stop_timer()

        # report
        msg = 'mean fht: {:2.2f}, ct: {:2.2f}' \
              ''.format(np.mean(sample.fht), sample.ct)
        print(msg)
