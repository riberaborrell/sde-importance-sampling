from mds.utils import make_dir_path
from mds.plots import Plot

import numpy as np
import os

def double_well_1d_gradient(x):
    alpha = 1
    return 4 * alpha * x * (x**2 - 1)

def get_idx_new_in_ts(x, been_in_target_set):
    is_in_target_set = x > 1

    idx = np.where(
            (is_in_target_set == True) &
            (been_in_target_set == False)
    )[0]

    been_in_target_set[idx] = True

    return idx

def save_som(dir_path, files_dict):
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'som.npz')
    np.savez(file_path, **files_dict)

def load_som(dir_path):
    make_dir_path(dir_path)
    file_path = os.path.join(dir_path, 'som.npz')
    data = np.load(
        file_path,
        allow_pickle=True,
    )
    return data

def plot_1d_control(x, control, dir_path):
    plt = Plot(dir_path, 'control')
    plt.xlabel = 'x'
    plt.set_ylim(- 5, 5)
    plt.one_line_plot(x, control)
