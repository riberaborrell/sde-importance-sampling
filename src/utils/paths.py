from datetime import datetime
import os
import shutil

import numpy as np

from utils.config import PROJECT_ROOT_DIR, DATA_ROOT_DIR

def get_project_dir():
    ''' returns the absolute path of the repository's directory
    '''
    return PROJECT_ROOT_DIR

def get_data_dir():
    ''' returns the absolute path of the repository's data directory
    '''
    return DATA_ROOT_DIR

def get_tests_plots_dir():
    ''' returns the absolute path of the directory for the tests plots
    '''
    return os.path.join(get_data_dir(), 'tests')

def make_dir_path(dir_path):
    ''' create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def empty_dir(dir_path):
    ''' remove all files in the directory from the given path
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

def get_fig_notebooks_dir_path():
    ''' returns the absolute path of the notebooks directory figures
    '''

    # get dir path
    dir_path = os.path.join(
        get_data_dir(),
        'notebooks',
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_hjb_solution_dir_path(settings_dir_path, h):
    ''' returns the absolute path of hjb-solution directory
    '''
    dir_path = os.path.join(
        settings_dir_path,
        'hjb-solution',
        'h_{:.0e}'.format(h),
    )
    return dir_path

def get_not_controlled_dir_path(settings_dir_path, dt, K, seed):
    ''' returns the absolute path of mc sampling directory
    '''
    # set seed string
    if seed is not None:
        seed_str = 'seed_{:d}'.format(seed)
    else:
        seed_str = ''

    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'mc-sampling',
        'dt_{}'.format(dt),
        'K_{:.0e}'.format(K),
        seed_str,
    )
    return dir_path

def get_controlled_dir_path(parent_dir_path, dt, K, seed):
    ''' returns the relative path of an importance sampling
    '''
    # set seed string
    if seed is not None:
        seed_str = 'seed_{:d}'.format(seed)
    else:
        seed_str = ''

    # get dir path
    dir_path = os.path.join(
        parent_dir_path,
        'is',
        'dt_{}'.format(dt),
        'K_{:.0e}'.format(K),
        seed_str,
    )
    return dir_path

def get_metadynamics_dir_path(meta_type, weights_type, omega_0, sigma_i, dt, delta, K, seed):
    ''' returns relative path of the metadynamics algorithm
    '''
    # set seed string
    if seed is not None:
        seed_str = 'seed_{:d}'.format(seed)
    else:
        seed_str = ''

    # get dir path
    dir_path = os.path.join(
        'metadynamics-{}'.format(meta_type),
        'weights-{}'.format(weights_type),
        'omega0_{}'.format(omega_0),
        'sigma-i_{}'.format(sigma_i),
        'dt_{}'.format(dt),
        'delta_{:2.2f}'.format(delta),
        'K_{}'.format(K),
        seed_str,
    )
    return dir_path

def get_metadynamics_nn_dir_path(settings_dir_path, dt, sigma_i, meta_type, k, N):
    #TODO! debug metadynamics with nn
    ''' Get metadynamics dir path and create its directories
    '''
    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'metadynamics-nn',
        'dt_{}'.format(dt),
        'sigma-i_{}'.format(sigma_i),
        meta_type,
        'k_{}'.format(k),
        'N_{}'.format(N),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_flat_bias_dir_path(settings_dir_path, dt, k_lim, N):
    #TODO! debug flat bias potential
    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'flat-bias-potential',
        'dt_{}'.format(dt),
        'k_lim_{}'.format(k_lim),
        'N_{}'.format(N),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_gaussian_ansatz_dir_path(settings_dir_path, distributed, theta, m_i=None,
                                 sigma_i=None, sigma_i_meta=None, delta_meta=None, K_meta=None,
                                 seed_meta=None, h=None):
    ''' Get gaussian ansatz dir path and create its directories
    '''
    # set seed string
    if seed_meta is not None:
        seed_str = 'seed_{:d}'.format(seed_meta)
    else:
        seed_str = ''

    # get dir path
    dir_path = os.path.join(
        settings_dir_path,
        'appr_value-f',
        'gaussian-ansatz',
    )

    if distributed == 'uniform':
        dir_path = os.path.join(
            dir_path,
            distributed,
            'm-i_{}'.format(m_i),
            'sigma-i_{}'.format(float(sigma_i)),
        )
    elif distributed == 'meta':
        dir_path = os.path.join(
            dir_path,
            distributed,
            'sigma-i_meta_{}'.format(float(sigma_i_meta)),
            'delta_{}'.format(delta),
            'K_meta_{}'.format(K_meta),
        )
    if theta == 'null':
        dir_path = os.path.join(dir_path, 'theta_' + theta)
    elif distributed == 'uniform' and theta == 'meta':
        dir_path = os.path.join(
            dir_path,
            'theta_' + theta,
            'sigma-i_meta_{}'.format(float(sigma_i_meta)),
            'delta_{}'.format(delta),
            'K_meta_{}'.format(K_meta),
            seed_str,
        )
    elif distributed == 'meta' and theta == 'meta':
        dir_path = os.path.join(dir_path, 'theta_' + theta)
    elif theta == 'hjb':
        dir_path = os.path.join(
            dir_path,
            'theta_' + theta,
            'h_{:.0e}'.format(h),
        )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path



def get_som_dir_path(func_appr_dir_path, grad_estimator, optimizer, lr, dt, K, seed):
    ''' Get stochastic optimization method absolute dir path and create its directories
    '''

    # set seed string
    if seed is not None:
        seed_str = 'seed_{:d}'.format(seed)
    else:
        seed_str = ''

    # get dir path
    dir_path = os.path.join(
        func_appr_dir_path,
        'grad_{}'.format(grad_estimator),
        optimizer,
        'lr_{}'.format(float(lr)),
        'dt_{}'.format(dt),
        'K_{}'.format(K),
        seed_str,
    )
    return dir_path


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

def get_trajectories_stamp(K):
    assert type(K) == int, 'Error:'
    trajectories_stamp = 'K{:.0e}'.format(K)
    return sde_stamp

def get_time_in_hms(dt):
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s
