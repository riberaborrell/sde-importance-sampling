from sde_importance_sampling.importance_sampling import Sampling

import pytest
from copy import copy, deepcopy

import numpy as np

class TestSamplingFht:

    @pytest.fixture
    def sample(self, n, alpha_i, beta, dt, k_lim, N):

        # initialize sampling object
        sample = Sampling(
            problem_name='langevin_stop-t',
            potential_name='nd_2well',
            n=n,
            alpha=np.full(n, alpha_i),
            beta=beta,
            is_controlled=False,
        )

        # set sampling and Euler-Marujama parameters
        sample.set_sampling_parameters(
            dt=dt,
            k_lim=k_lim,
            xzero=np.full(n, -1),
            N=N,
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
            gradient = sample.gradient(xt)

            # get Brownian increment
            dB = sample.brownian_increment()

            # sde update
            xt = sample.sde_update(xt, gradient, dB)

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
        sample.fht = np.zeros(sample.N)

        # initialize xt
        xt = sample.initial_position()

        for k in np.arange(1, sample.k_lim + 1):

            # not in the target set
            idx = (
                (xt < sample.target_set[:, 0]) |
                (xt > sample.target_set[:, 1])
            ).any(axis=1).reshape(sample.N,)

            # number of trajectory which are not in the target set
            N_idx = np.sum(idx)

            # if all trajectories are in the target set
            if N_idx == 0:

                # save number of time steps used
                sample.k = k
                break

            # compute gradient
            gradient = sample.gradient(xt[idx])

            # get Brownian increment
            dB = np.sqrt(sample.dt) \
               * np.random.normal(0, 1, N_idx * sample.n).reshape(N_idx, sample.n)

            # sde update
            xt[idx] = sample.sde_update(xt[idx], gradient, dB)

            # update first hitting time
            sample.fht[idx] += sample.dt

        sample.stop_timer()

        # report
        msg = 'mean fht: {:2.2f}, ct: {:2.2f}' \
              ''.format(np.mean(sample.fht), sample.ct)
        print(msg)
