import os
import shutil

MDS_PATH = os.path.abspath(os.path.dirname(__file__))

def make_dir_path(dir_path):
    ''' Create directories of the give path if they do not already exist
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

def get_data_path(potential=None, beta=None, target_set=None):
    # get dir path
    if potential and beta and target_set:
        target_set_min, target_set_max = target_set
        dir_path = os.path.join(
            MDS_PATH,
            'data',
            potential,
            'beta_{}'.format(float(beta)),
            'target_set_{}_{}'.format(target_set_min, target_set_max),
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
