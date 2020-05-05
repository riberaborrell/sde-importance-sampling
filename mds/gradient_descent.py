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
        
        self.m = None
        self.a_opt = None
        self.mus = None
        self.sigmas = None
        
        self.a_s = None

        self.do_plots = False

    def set_parameters_greedy(self, m):
        epochs = self.epochs

        self.m = m
        self.a_s = np.zeros((epochs + 1, m))
        
        # initialize langevin_1d object
        samp = sampling.langevin_1d(beta=1)

        # set a and coefficients of the ansatz functions from metadynamics
        samp.set_uniformly_dist_ansatz_functions(
            m=10,
            sigma=0.2,
        )
        samp.set_a_from_metadynamics()
        
        # get optimal solution
        samp.set_a_optimal()
    
        self.a_opt = samp.a_opt
        self.a_s[0] = samp.a
        self.mus = samp.mus
        self.sigmas = samp.sigmas 

    def train_step(self, a, epoch):
        lr = self.lr
        M = self.M
        mus = self.mus
        sigmas = self.sigmas

        # initialize langevin_1d object
        samp = sampling.langevin_1d(beta=1)

        # set a and coefficients of the ansatz functions
        samp.set_bias_potential(a, mus, sigmas)

        # plot potential and gradient
        if self.do_plots:
            file_name = 'potential_and_gradient_gd_greedy_epoch{:d}'.format(epoch)
            samp.plot_potential_and_gradient(
                file_name=file_name,
                dir_path=GD_FIGURES_PATH,
            )


        # set sampling and Euler-Majurama parameters
        samp.set_sampling_parameters(
            xzero=-1,
            M=M,
            target_set=[0.9, 1.1],
            dt=0.001,
            N=100000,
        )
        
        # compute loss function and its gradient
        samp.sample_soc()

        samp.compute_statistics()
        loss = samp.mean_J
        grad_loss = samp.mean_gradJ

        a_dif = self.a_opt - a
        print(epoch, loss, np.mean(grad_loss))
        print(a)

        # Update parameters
        a = a - lr * grad_loss 
        
        # Returns the parameters and the loss
        return a, loss

    def gradient_descent(self):
        m = self.m
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
            os.path.join(DATA_PATH, 'losts_gd_greedy.npz'),
            a=self.a_s,
            losses=self.losses,
        )

        # save optimal bias potential
        np.savez(
            os.path.join(DATA_PATH, 'langevin1d_bias_potential_gd_greedy.npz'),
            a=self.a_s[-1],
            mus=self.mus,
            sigmas=self.sigmas,
        )

    def plot_tilted_potentials(self):
        epochs = self.epochs
        mus = self.mus
        sigmas = self.sigmas

        X = np.linspace(-2, 2, 100)

        # compute tilted potentials
        Vs = np.zeros((epochs + 1, 100))
        for epoch in np.arange(epochs + 1):
            samp = sampling.langevin_1d(beta=1)
            a = self.a_s[epoch]
            samp.set_bias_potential(a, mus, sigmas)
            Vs[epoch, :] = samp.tilted_potential(X)
        
        # compute optimal tilted potential
        Voptimal = np.zeros(100)

        pl = Plot(file_name='tilted_potentials_gradient_descent')
        pl.gradient_descent_tilted_potentials(X, Vs, Voptimal)
