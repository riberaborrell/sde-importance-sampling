import sampling

import numpy as np
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

class gradient_descent:
    '''
    '''
    def __init__(self, lr, epochs, M):
        '''
        '''
        self._lr = lr
        self._epochs = epochs
        self._M = M

        self._losses = np.zeros(epochs + 1)
        
        self._m = None
        self._as = None
        self._mus = None
        self._sigmas = None

    def set_parameters_greedy(self, m):
        epochs = self._epochs

        self._m = m
        self._as = np.zeros((epochs + 1, m))
        
        # initialize langevin_1d object
        samp = sampling.langevin_1d(
            beta=2,
            is_drifted=True,
        )

        # set a and coefficients of the ansatz functions from metadynamics
        samp.set_bias_potential_from_metadynamics(
            m=m,
            J_min=-1.9,
            J_max=0.9,
        )
        self._as[0] = samp._a
        self._mus = samp._mus
        self._sigmas = samp._sigmas 

    def train_step(self, a):
        lr = self._lr
        M = self._M
        mus = self._mus
        sigmas = self._sigmas

        # initialize langevin_1d object
        samp = sampling.langevin_1d(
            beta=2,
            is_drifted=True,
        )

        # set a and coefficients of the ansatz functions
        samp.set_bias_potential(a, mus, sigmas)

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
        #samp.sample_soc_vectorized()

        samp.compute_statistics()
        loss = samp._mean_J
        grad_loss = samp._mean_gradJ

        # Update parameters
        a = a - lr * grad_loss 
        
        # Returns the parameters and the loss
        return a, loss

    def gradient_descent(self):
        m = self._m
        epochs = self._epochs
        losses = self._losses
        
        a = self._as[0]
        for epoch in np.arange(1, epochs + 1):
            print(epoch)

            # compute loss and do update parameters
            a, loss = self.train_step(a)

            # save loss 
            self._losses[epoch - 1] = loss

            # save parameters
            self._as[epoch] = a


    def save_statistics(self):
        # save as and losses
        np.savez(
            os.path.join(DATA_PATH, 'losts_gd_greedy.npz'),
            a=self._as,
            losses=self._losses,
        )

        # save optimal bias potential
        np.savez(
            os.path.join(DATA_PATH, 'langevin1d_bias_potential_gd_greedy.npz'),
            a=self._as[-1],
            mus=self._mus,
            sigmas=self._sigmas,
        )
