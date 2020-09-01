import numpy as np

def is_valid_1d_target_set(target_set):
    '''
    '''
    target_set_min, target_set_max = target_set
    if target_set_min >= target_set_max:
        return False
    return True

def is_valid_control(u, lower_bound, upper_bound):
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
