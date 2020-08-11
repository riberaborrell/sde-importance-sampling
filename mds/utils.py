from datetime import datetime

import os
import shutil

MDS_PATH = os.path.abspath(os.path.dirname(__file__))

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

def get_data_path(potential=None, alpha=None, beta=None, target_set=None, subdirectory=None):
    ''' Get data path and create its directories
    '''
    # get dir path
    if potential and alpha and beta and target_set and subdirectory:
        target_set_min, target_set_max = target_set
        dir_path = os.path.join(
            MDS_PATH,
            'data',
            potential,
            'alpha_{}'.format(float(alpha)),
            'beta_{}'.format(float(beta)),
            'target_set_{}_{}'.format(target_set_min, target_set_max),
            subdirectory,
        )
    elif potential and alpha and beta and target_set:
        target_set_min, target_set_max = target_set
        dir_path = os.path.join(
            MDS_PATH,
            'data',
            potential,
            'alpha_{}'.format(float(alpha)),
            'beta_{}'.format(float(beta)),
            'target_set_{}_{}'.format(target_set_min, target_set_max),
        )
    elif potential and alpha:
        dir_path = os.path.join(
            MDS_PATH,
            'data',
            potential,
            'alpha_{}'.format(float(alpha)),
        )
    elif potential:
        dir_path = os.path.join(
            MDS_PATH,
            'data',
            potential,
        )
    else:
        dir_path = os.path.join(MDS_PATH, 'data')

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_datetime_stamp():
    time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    return time_stamp

def get_sde_stamp(alpha, beta):
    sde_stamp = 'alpha{}_beta{}'.format(float(alpha), float(beta))
    return sde_stamp

def get_trajectories_stamp(M):
    assert type(M) == int, 'Error:'
    trajectories_stamp = 'M{:.0e}'.format(M)
    return sde_stamp

def get_time_in_hms(dt):
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s

def get_time_in_hms(dt):
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s
