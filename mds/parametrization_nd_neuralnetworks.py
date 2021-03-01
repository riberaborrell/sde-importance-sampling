from mds.langevin_nd_sde import LangevinSDE
from mds.neural_networks import TwoLayerNet

import numpy as np
import scipy.stats as stats

class GaussianAnsatz(LangevinSDE):
    '''
    '''
    def __init__(self, n, potential_name, alpha, beta,
                 target_set=None, domain=None, h=None):
        '''
        '''
        super().__init__(n, potential_name, alpha, beta,
                         target_set, domain, h)

