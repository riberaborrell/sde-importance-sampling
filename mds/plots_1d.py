from mds.utils import get_example_data_path

import numpy as np
import matplotlib.pyplot as plt

import os

class Plot1d:
    dir_path = get_example_data_path()
    def __init__(self, dir_path=dir_path, file_name=None, file_type='png'):
        self.file_name = file_name
        self.file_type = file_type
        self.dir_path = dir_path

        # title and label
        self.title = ''
        self.label = ''

        # axes labels
        self.xlabel = 'x'
        self.ylabel = None

        # axes bounds
        self.xmin = None
        self.xmax= None
        self.ymin = None
        self.ymax= None
        self.yticks = None

    @property
    def file_path(self):
        if self.file_name:
            return os.path.join(
                self.dir_path, self.file_name + '.' + self.file_type
            )
        else:
            return None

    def set_title(self, title):
        self.title = title

    def set_xlim(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def set_ylim(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax
        tick_sep = (ymax - ymin) / 10
        self.yticks = np.arange(ymin, ymax + tick_sep, tick_sep)

    def one_line_plot(self, x, y, color=None, label=None):
        assert x.ndim == y.ndim == 1, ''
        assert x.shape[0] == y.shape[0], ''

        plt.title(self.title)
        plt.plot(x, y, color=color, linestyle='-', label=label)
        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.grid(True)
        if label is not None:
            plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def multiple_lines_plot(self, x, ys, colors=None, linestyles=None, labels=None):
        assert x.ndim == 1, ''
        assert ys.ndim == 2, ''
        assert x.shape[0] == ys.shape[1], ''

        num_plots = ys.shape[0]
        if colors is not None:
            assert num_plots == len(colors), ''
        else:
            colors = [None for i in range(num_plots)]
        if linestyles is not None:
            assert num_plots == len(linestyles), ''
        else:
            linestyles = ['-' for i in range(num_plots)]
        if labels is not None:
            assert num_plots == len(labels), ''
        else:
            labels = [None for i in range(num_plots)]

        plt.title(self.title)
        for i in range(num_plots):
            plt.plot(x, ys[i], color=colors[i],
                     linestyle=linestyles[i], label=labels[i])

        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.yticks(self.yticks)
        plt.grid(True)
        if any(label is not None for label in labels):
            plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()

    def one_bar_plot(self, x, height, color=None, label=None):
        assert x.ndim == height.ndim == 1, ''
        assert x.shape[0] == height.shape[0], ''

        plt.title(self.title)
        plt.bar(x, height, width=0.8, color=color, label=label, align='center')
        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        #plt.xlim(left=epochs[0] -0.8, right=epochs[-1] + 0.8)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        #plt.set_yticks([-1,0,1])
        if label is not None:
            plt.legend(loc='upper left', fontsize=8)
        plt.grid(True)
        plt.savefig(self.file_path)
        plt.close()

    def trajectory(self, x, y):
        plt.title(self.title)
        plt.plot(x, y, 'r', label='EM solution path')
        plt.xlabel('t', fontsize=16)
        plt.ylabel('X', fontsize=16)
        plt.ylim(-1.8, 1.8)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(self.file_path)
        plt.close()
