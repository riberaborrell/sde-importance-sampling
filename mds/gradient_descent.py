import sampling

from metadynamics import get_a_from_metadynamics

import numpy as np

class gradient_descent:
    '''
    '''
    def __init__(self, lr, epochs, m, M):
        '''
        '''
        self._m = m
        self._a = np.zeros((epochs + 1, m))
        self._mus = np.zeros(m)
        self._sigmas = np.zeros(m)

        self._lr = lr
        self._epochs = epochs

        self._M = M
        self._losses = np.zeros(epochs + 1)

    def train_step(self, a):
        lr = self._lr
        mus = self._mus
        sigmas = self._sigmas

        samp = sampling.langevin_1d(
            seed=1, 
            beta=5,
            xzero=-1,
            target_set=[0.9, 1.1],
            num_trajectories=1000, 
            is_drifted=True,
            do_reweighting=False,
            is_sampling_problem=False,
            is_soc_problem=True,
        )

        # set a and coefficients of the ansatz functions
        samp.get_a(a)
        samp.get_v(mus, sigmas)
        
        # compute loss function and its gradient
        samp.sample()
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

        a, mus, sigmas = get_a_from_metadynamics(
            beta=5,
            m=m,
            J_min=-1.9,
            J_max=0.9,
        )
        self._a[0] = a
        self._mus = mus
        self._sigmas = sigmas 

        for epoch in np.arange(1, epochs + 1):
            print(epoch)
            # compute loss and do update parameters
            a, loss = self.train_step(a)

            # save loss 
            self._losses[epoche - 1]

            # save parameters
            self._a[epoche]
        
