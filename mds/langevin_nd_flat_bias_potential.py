from mds.utils import get_flat_bias_dir_path

import numpy as np
import time

import os

class GetFlatBiasPotential:
    '''
    '''

    def __init__(self, sample, seed=None):

        # sampling object
        self.sample = sample

        # seed
        self.seed = seed

        # training data
        self.x = None
        self.u = None

        # computational time
        self.t_initial = None
        self.t_final = None

        # set path
        self.dir_path = None

    def set_dir_path(self):
        self.dir_path = get_flat_bias_dir_path(
            self.sample.settings_dir_path,
            self.sample.dt,
            self.sample.k_lim,
            self.sample.N,
        )

    def start_timer(self):
        self.t_initial = time.perf_counter()

    def stop_timer(self):
        self.t_final = time.perf_counter()

    def get_training_data(self):
        self.start_timer()

        # preallocate data
        x = np.empty((self.sample.k_lim + 1, self.sample.N, self.sample.n))
        u = np.empty((self.sample.k_lim + 1, self.sample.N, self.sample.n))

        # initialize x
        x[0] = self.sample.sample_domain_uniformly(N=self.sample.N)
        #x[0] = np.full((self.sample.N, self.sample.n), self.sample.xzero)
        gradient = self.sample.gradient(x[0])
        u[0] = gradient / np.sqrt(2)

        for k in np.arange(1, self.sample.k_lim + 1):
            # Brownian increment
            dB = np.sqrt(self.sample.dt) \
               * np.random.normal(0, 1, self.sample.N * self.sample.n).reshape(self.sample.N, self.sample.n)

            # sde update
            x[k] = self.sample.sde_update(x[k -1], gradient, dB)
            gradient = self.sample.gradient(x[k])
            u[k] = gradient / np.sqrt(2)

        # save flattened
        self.x = x.reshape((self.sample.k_lim + 1) * self.sample.N, self.sample.n)
        self.u = u.reshape((self.sample.k_lim + 1) * self.sample.N, self.sample.n)

        self.stop_timer()

    def save(self):
        np.savez(
            os.path.join(self.dir_path, 'flat-bias-potential.npz'),
            x=self.x,
            u=self.u,
            t_initial=self.t_initial,
            t_final=self.t_final,
        )

    def load(self):
        try:
            bias_pot = np.load(
                os.path.join(self.dir_path, 'flat-bias-potential.npz'),
                allow_pickle=True,
            )
            self.x = bias_pot['x']
            self.u = bias_pot['u']
            self.t_initial = bias_pot['t_initial']
            self.t_final = bias_pot['t_final']
            return True

        except:
            msg = 'no training data for a flat bias potential found with dt={:.4f}, N={:.0e}' \
                  ''.format(self.sample.dt, self.sample.N)
            print(msg)
            return False

    def plot_2d_training_data(self):
        from figures.myfigure import MyFigure

        x, y = np.moveaxis(self.x, -1, 0)

        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=self.dir_path,
            file_name='training-data',
        )
        plt.scatter(x, y)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.savefig(fig.file_path)
        plt.close()
