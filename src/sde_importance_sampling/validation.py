import numpy as np

def is_1d_valid_interval(domain):
    '''
    '''
    assert domain.ndim == 1, ''
    assert domain.shape[0] == 2, ''
    assert domain[0] < domain[1], ''

def is_1d_valid_target_set(domain, target_set):
    '''
    '''
    is_1d_valid_interval(target_set)
    assert domain[0] <= target_set[0], ''
    assert domain[1] >= target_set[1], ''

def is_1d_valid_control(u, lower_bound, upper_bound):
    '''
        Args:
            lower_bound (float): lower bound
            upper_bound (float): upper bound
            u ((M,)-array) : position
    '''
    assert u.ndim == 1, ''
    if np.min(u) < lower_bound:
        return False
    elif np.max(u) > upper_bound:
        return False
    return True

def is_2d_valid_interval(domain):
    '''
    '''
    assert domain.ndim == 2, ''
    assert domain.shape == (2, 2), ''
    assert domain[0][0] < domain[0][1], ''
    assert domain[1][0] < domain[1][1], ''

def is_2d_valid_target_set(domain, target_set):
    '''
    '''
    is_2d_valid_interval(target_set)
    assert domain[0][0] <= target_set[0][0], ''
    assert domain[0][1] >= target_set[0][1], ''
    assert domain[1][0] <= target_set[1][0], ''
    assert domain[1][1] >= target_set[1][1], ''

def is_2d_valid_control(u, lower_bound, upper_bound):
    '''
        Args:
            lower_bound ((2,)-array): lower bound
            upper_bound ((2,)-array): upper bound
            u ((M,2)-array) : position
    '''
    assert u.ndim == 2, ''
    assert u.shape[1] == 2, ''
    if (np.min(u, axis=0) < lower_bound).any():
        return False
    elif (np.max(u, axis=0) > upper_bound).any():
        return False
    return True
