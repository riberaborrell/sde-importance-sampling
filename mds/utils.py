import numpy as np

from datetime import datetime
from pathlib import Path
import shutil

import os

MDS_PATH = Path(os.path.dirname(__file__))
PROJECT_PATH = MDS_PATH.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

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

def get_settings_dir_path(potential=None, n=None, alpha_i=None,
                         beta=None, target_set=None):
    ''' Get absolute path of the directory for the chosen settings and create its directories
    '''
    # get dir path
    if (potential is not None and
        n is not None and
        alpha_i is not None and
        beta is not None and
        target_set is not None):

        dir_path = os.path.join(
            DATA_PATH,
            potential,
            'n_{:d}'.format(n),
            'alpha_i_{}'.format(float(alpha_i)),
            'beta_{}'.format(float(beta)),
            get_target_set_str(target_set),
        )

    elif (potential is not None and
          n is not None and
          alpha_i is not None):

        dir_path = os.path.join(
            DATA_PATH,
            potential,
            'n_{:d}'.format(n),
            'alpha_i_{}'.format(float(alpha_i)),
        )

    elif (potential is not None and n is not None):
        dir_path = os.path.join(
            DATA_PATH,
            potential,
            'n_{:d}'.format(n),
        )

    else:
        dir_path = DATA_PATH

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_hjb_solution_dir_path(settings_dir_path, h):
    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'hjb-solution',
        'h_{:.0e}'.format(h),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_not_controlled_dir_path(settings_dir_path, dt, N):
    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'mc-sampling',
        'dt_{}'.format(dt),
        'N_{:.0e}'.format(N),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_controlled_dir_path(parent_dir_path, dt, N):
    # get dir path
    dir_path = os.path.join(
        parent_dir_path,
        'is',
        'dt_{}'.format(dt),
        'N_{:.0e}'.format(N),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_metadynamics_dir_path(settings_dir_path, dt, sigma_i, k, N):
    ''' Get metadynamics dir path and create its directories
    '''
    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'metadynamics',
        'dt_{}'.format(dt),
        'sigma_i_{}'.format(sigma_i),
        'k_{}'.format(k),
        'N_{}'.format(N),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_gaussian_ansatz_dir_path(settings_dir_path, distributed, theta, m_i=None,
                                 sigma_i=None, sigma_i_meta=None, k=None, N_meta=None, h=None):
    ''' Get gaussian ansatz dir path and create its directories
    '''
    # get dir path
    if distributed == 'uniform':
        dir_path = os.path.join(
            settings_dir_path,
            'gaussian-value-f',
            distributed,
            'm_i_{}'.format(m_i),
            'sigma_i_{}'.format(float(sigma_i)),
        )
    elif distributed == 'meta':
        dir_path = os.path.join(
            settings_dir_path,
            'gaussian-value-f',
            distributed,
            'sigma_i_meta_{}'.format(float(sigma_i_meta)),
            'k_{}'.format(k),
            'N_meta_{}'.format(N_meta),
        )
    if theta == 'null':
        dir_path = os.path.join(dir_path, 'theta_' + theta)
    elif distributed == 'uniform' and theta == 'meta':
        dir_path = os.path.join(
            dir_path,
            'theta_' + theta,
            'sigma_i_meta_{}'.format(float(sigma_i_meta)),
            'k_{}'.format(k),
            'N_meta_{}'.format(N_meta),
        )
    elif distributed == 'meta' and theta == 'meta':
        dir_path = os.path.join(dir_path, 'theta_' + theta)
    elif theta == 'optimal':
        dir_path = os.path.join(
            dir_path,
            'theta_' + theta,
            'h_{:.0e}'.format(h),
        )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_nn_function_approximation_dir_path(settings_dir_path, target_function,
                                           nn_rel_path, initialization):
    dir_path = os.path.join(
        settings_dir_path,
        'appr_{}'.format(target_function),
        nn_rel_path,
        'theta_{}'.format(initialization),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path


def get_som_dir_path(func_appr_dir_path, loss_type, optimizer, lr, dt, N):
    ''' Get stochastic optimization method absolute dir path and create its directories
    '''
    # get dir path
    dir_path = os.path.join(
        func_appr_dir_path,
        'loss_{}'.format(loss_type),
        optimizer,
        'lr_{}'.format(float(lr)),
        'dt_{}'.format(dt),
        'N_{}'.format(N),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

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

def get_target_set_str(target_set):
    if type(target_set) == str:
        return 'ts_' + target_set
    if type(target_set) == np.ndarray and target_set.ndim > 1:
        if target_set.ndim == 2 and target_set.shape == (2, 2):
            target_set = target_set.reshape((target_set.shape[0] * target_set.shape[1]))
    target_set_str = 'target_set'
    for entry in target_set:
        target_set_str += '_{}'.format(float(entry))
    return target_set_str

def get_datetime_stamp():
    time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    return time_stamp

def get_trajectories_stamp(N):
    assert type(N) == int, 'Error:'
    trajectories_stamp = 'N{:.0e}'.format(N)
    return sde_stamp

def get_time_in_hms(dt):
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s
