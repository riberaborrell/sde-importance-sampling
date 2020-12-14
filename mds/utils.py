from datetime import datetime

import numpy as np
import os
import shutil

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

def make_dir_path(dir_path):
    ''' Create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def empty_dir(dir_path):
    ''' Remove all files in the directory from the given path
    '''
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Reason: {}'.format((file_path, e)))

def get_example_dir_path(potential=None, n=None, alpha=None, beta=None, target_set=None, subdirectory=None):
    ''' Get example data path and create its directories
    '''
    # get dir path
    if (potential is not None and
        n is not None and
        alpha is not None and
        beta is not None and
        target_set is not None and
        subdirectory is not None):

        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_n_stamp(n),
            get_alpha_stamp(alpha),
            get_beta_stamp(beta),
            get_target_set_stamp(target_set),
            subdirectory,
        )

    elif (potential is not None and
          n is not None and
          alpha is not None and
          beta is not None and
          target_set is not None):

        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_n_stamp(n),
            get_alpha_stamp(alpha),
            get_beta_stamp(beta),
            get_target_set_stamp(target_set),
        )

    elif (potential is not None and
          n is not None and
          alpha is not None):

        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_n_stamp(n),
            get_alpha_stamp(alpha),
        )

    elif (potential is not None and n is not None):
        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_n_stamp(n),
        )

    else:
        dir_path = DATA_PATH

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path


def get_example_data_path(potential=None, alpha=None, beta=None, target_set=None, subdirectory=None):
    ''' Get example data path and create its directories
    '''
    # get dir path
    if potential and alpha is not None and beta and target_set is not None and subdirectory:
        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_alpha_stamp(alpha),
            get_beta_stamp(beta),
            get_target_set_stamp(target_set),
            subdirectory,
        )
    elif potential and alpha is not None and beta and target_set is not None:
        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_alpha_stamp(alpha),
            get_beta_stamp(beta),
            get_target_set_stamp(target_set),
        )
    elif potential and alpha is not None:
        dir_path = os.path.join(
            DATA_PATH,
            potential,
            get_alpha_stamp(alpha),
        )
    elif potential:
        dir_path = os.path.join(
            DATA_PATH,
            potential,
        )
    else:
        dir_path = DATA_PATH

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_ansatz_data_path(example_data_path, ansatz_type, m, sigma):
    ''' Get ansatz data path and create its directories
    '''
    # get dir path
    dir_path = os.path.join(
        example_data_path,
        ansatz_type,
        get_m_stamp(m),
        get_sigma_stamp(sigma),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_gd_data_path(ansatz_data_path, gd_type, theta_init, lr):
    ''' Get gd data path and create its directories
    '''
    # get dir path
    dir_path = os.path.join(
        ansatz_data_path,
        gd_type,
        'theta_init_{}'.format(theta_init),
        get_lr_stamp(lr),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_n_stamp(n):
    return 'n_{:d}'.format(n)

def get_alpha_stamp(alpha):
    assert alpha.ndim == 1, ''
    alpha_stamp = 'alpha'
    for alpha_i in alpha:
        alpha_stamp += '_{}'.format(float(alpha_i))
    return alpha_stamp

def get_beta_stamp(beta):
    return 'beta_{}'.format(float(beta))

def get_sde_stamp(alpha, beta):
    sde_stamp = get_alpha_stamp(alpha)
    sde_stamp += get_beta_stamp(beta)
    return sde_stamp

def get_target_set_stamp(target_set):
    if type(target_set) == str:
        return 'ts_' + target_set
    if type(target_set) == np.ndarray and target_set.ndim > 1:
        if target_set.ndim == 2 and target_set.shape == (2, 2):
            target_set = target_set.reshape((target_set.shape[0] * target_set.shape[1]))
    target_set_stamp = 'target_set'
    for entry in target_set:
        target_set_stamp += '_{}'.format(float(entry))
    return target_set_stamp

def get_m_stamp(m):
    return 'm_{}'.format(m)

def get_sigma_stamp(sigma):
    return 'sigma_{}'.format(float(sigma))

def get_lr_stamp(lr):
    return 'lr_{}'.format(float(lr))

def get_datetime_stamp():
    time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    return time_stamp

def get_trajectories_stamp(M):
    assert type(M) == int, 'Error:'
    trajectories_stamp = 'M{:.0e}'.format(M)
    return sde_stamp

def get_time_in_hms(dt):
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s
