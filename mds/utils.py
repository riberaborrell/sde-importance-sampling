import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))

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
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path
