import numpy as np

def is_valid_1d_target_set(target_set):
    '''
    '''
    target_set_min, target_set_max = target_set
    if target_set_min >= target_set_max:
        return False
    return True
