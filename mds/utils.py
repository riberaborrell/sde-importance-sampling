import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))

def get_figures_path(potential=None, target_set=None, beta=None):
    # get dir path
    if potential and target_set and beta:
        target_set_min, target_set_max = target_set
        dir_path = os.path.join(
            MDS_PATH,
            'figures',
            potential,
            'target_set_{}_{}'.format(target_set_min, target_set_max),
            'beta_{}'.format(beta)
        )
    elif potential:
        dir_path = os.path.join(
            MDS_PATH,
            'figures',
            potential,
        )
    else:
        dir_path = os.path.join(MDS_PATH, 'figures')

    # create dir path if not exists
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    return dir_path

def get_data_path(potential=None, target_set=None, beta=None):
    # get dir path
    if potential and target_set and beta:
        target_set_min, target_set_max = target_set
        dir_path = os.path.join(
            MDS_PATH,
            'data',
            potential,
            'target_set_{}_{}'.format(target_set_min, target_set_max),
            'beta_{}'.format(beta)
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
