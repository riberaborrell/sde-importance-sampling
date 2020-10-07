import numpy as np

def is_1d_valid_domain(domain):
    '''
    '''
    assert domain.ndim == 1, ''
    assert domain.shape[0] == 2, ''
    assert domain[0] < domain[1], ''

def is_1d_valid_target_set(domain, target_set):
    '''
    '''
    assert target_set.ndim == 1, ''
    assert target_set.shape[0] == 2, ''
    assert target_set[0] < target_set[1], ''
    assert domain[0] <= target_set[0], ''
    assert domain[1] >= target_set[1], ''

def is_1d_valid_control(u, lower_bound, upper_bound):
    '''
    '''
    if type(u) == np.ndarray and np.min(u) < lower_bound:
        return False
    elif type(u) == np.ndarray and np.max(u) > upper_bound:
        return False
    elif type(u) == float and u < lower_bound:
        return False
    elif type(u) == float and u > upper_bound:
        return False
    return True
