from plotting import Plot
import sampling

import numpy as np
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
GD_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/gradient_descent_greedy')

class gradient_descent:
    '''
    '''
    def __init__(self, lr, epochs, M):
        '''
        '''
        self.lr = lr
        self.epochs = epochs
        self.M = M

        self.losses = np.zeros(epochs + 1)
        self.a_s = None
        
        self.sample = None

        self.do_plots = False

    def set_sample(self):
        # initialize langevin_1d object
        sample = sampling.langevin_1d(beta=1)

        # set sampling and Euler-Majurama parameters
        sample.set_sampling_parameters(
            xzero=-1,
            M=self.M,
            target_set=[0.9, 1.1],
            dt=0.001,
            N=100000,
        )
        sample.set_a_optimal()
        self.sample = sample

    def set_ansatz_functions_greedy(self, m, sigma):
        epochs = self.epochs

        # set uniformly distributed ansatz functions
        self.sample.set_uniformly_dist_ansatz_functions(m, sigma)
        
        # preallocate coefficient a of the gd
        self.a_s = np.zeros((epochs + 1, m))

    def set_a_from_metadynamics_greedy(self):
        self.sample.set_a_from_metadynamics()
        self.a_s[0] = self.sample.a

    def set_a_optimal_greedy(self):
        self.sample.is_drifted = True
        self.a_s[0] = self.sample.a_opt

    def set_a_null_greedy(self):
        self.sample.is_drifted = True
        m = self.sample.m
        self.a_s[0] = np.zeros(m)

    def train_step(self, a, epoch):
        lr = self.lr
        M = self.M
        sample = self.sample
        a_opt = sample.a_opt

        # set a coefficients
        sample.a = a

        # plot potential and gradient
        if self.do_plots:
            file_name = 'potential_and_gradient_gd_greedy_epoch{:d}'.format(epoch)
            sample.plot_potential_and_gradient(
                file_name=file_name,
                dir_path=GD_FIGURES_PATH,
            )

        # compute loss function and its gradient
        sample.sample_soc()
        sample.compute_statistics()

        loss = sample.mean_J
        grad_loss = sample.mean_gradJ

        a_dif = a_opt - a
        print(epoch, loss, np.mean(grad_loss))
        print(a)

        # Update parameters
        a -= lr * grad_loss 
        
        # Returns the parameters and the loss
        return a, loss

    def gradient_descent_greedy(self):
        epochs = self.epochs
        losses = self.losses
        
        a = self.a_s[0]
        for epoch in np.arange(1, epochs + 1):

            # compute loss and do update parameters
            a, loss = self.train_step(a, epoch)

            # save loss 
            self.losses[epoch - 1] = loss

            # save parameters
            self.a_s[epoch] = a


    def save_statistics(self):
        # save as and losses
        np.savez(
            os.path.join(DATA_PATH, 'langevin1d_gd_greedy.npz'),
            lr=self.lr,
            epochs=self.epochs,
            M=self.M,
            a_opt=self.sample.a_opt,
            mus=self.sample.mus,
            sigmas=self.sample.sigmas,
            a_s=self.a_s,
            losses=self.losses,
        )

    def plot_tilted_potentials(self):
        sample = self.sample
        epochs = self.epochs
        a_s = self.a_s

        X = np.linspace(-2, 2, 100)

        # compute tilted potentials
        Vs = np.zeros((epochs + 1, 100))
        for epoch in np.arange(epochs + 1):
            sample.a = a_s[epoch]
            Vs[epoch, :] = sample.tilted_potential(X)
        
        # compute optimal tilted potential
        sample.a = sample.a_opt
        Voptimal = sample.tilted_potential(X)

        pl = Plot(file_name='tilted_potentials_gradient_descent')
        pl.gradient_descent_tilted_potentials(X, Vs, Voptimal)
