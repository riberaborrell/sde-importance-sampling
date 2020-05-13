from plotting import Plot
import sampling

from datetime import datetime
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

        self.losses = np.zeros(epochs)
        self.grad_losses = None
        self.a_s = None
        self.a_dif = None
        
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
        self.sample = sample

    def set_ansatz_functions_greedy(self, m, sigma):
        epochs = self.epochs

        # set uniformly distributed ansatz functions
        self.sample.set_uniformly_dist_ansatz_functions(m, sigma)

        # set optimal solution in the ansatz function basis
        self.sample.set_a_optimal()
        
        # preallocate coefficient a of the gd
        self.a_s = np.zeros((epochs + 1, m))
        
        # preallocate grad of the losses 
        self.grad_losses = np.zeros((epochs, m))

    def set_a_from_metadynamics_greedy(self):
        self.sample.set_a_from_metadynamics()
        self.a_s[0] = self.sample.a

    def set_a_optimal_greedy(self):
        self.sample.is_drifted = True
        self.a_s[0] = self.sample.a_opt

    def set_a_null_greedy(self):
        self.sample.is_drifted = True
        self.a_s[0] = np.zeros(self.sample.m)

    def train_step(self, a):
        lr = self.lr
        M = self.M
        sample = self.sample

        # set a coefficients
        sample.a = a

        # compute loss function and its gradient
        sample.sample_soc()
        sample.compute_statistics()
        loss = sample.mean_J
        grad_loss = sample.mean_gradJ

        # Update parameters
        a_updated = a - lr * grad_loss 
        
        # Returns the parameters, the loss and the gradient
        return a_updated, loss, grad_loss

    def gradient_descent_greedy(self):
        epochs = self.epochs
        
        for epoch in np.arange(epochs):
            
            # plot potential and gradient
            if self.do_plots:
                self.sample.a = self.a_s[epoch]
                file_name = 'potential_and_gradient_gd_greedy_epoch{:d}'.format(epoch)
                self.sample.plot_potential_and_gradient(
                    file_name=file_name,
                    dir_path=GD_FIGURES_PATH,
                )
            # compute loss and its gradient and update parameters
            a, loss, grad_loss = self.train_step(self.a_s[epoch])

            # save loss and its gradient 
            self.losses[epoch] = loss
            self.grad_losses[epoch] = grad_loss

            # save parameters
            self.a_s[epoch + 1] = a

            # print 
            print(epoch)
            print(self.a_s[epoch])  
            print(loss)
            print(grad_loss)
            print(self.sample.a_opt - self.a_s[epoch])  
        print(epoch + 1)
        print(self.a_s[epoch + 1])  
        print(self.sample.a_opt - self.a_s[epoch + 1])  

        # compute eucl dist between a and a opt
        self.a_dif = np.linalg.norm(self.a_s - self.sample.a_opt, axis=1)

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
            a_dif=self.a_dif,
            losses=self.losses,
            grad_losses=self.grad_losses,
        )

    def write_statistics(self):
        # set path
        time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(
            DATA_PATH, 
            'langevin1d_gd_greedy' + time_stamp + '.txt',
        )

        # write in file
        f = open(file_path, "w")
        
        f.write('m: {:d}\n'.format(self.sample.m))
        for j in np.arange(self.sample.m):
            f.write('a_opt_j: {:2.4e}\n'.format(self.sample.a_opt[j]))
        f.write('\n')

        for epoch in np.arange(self.epochs):
            f.write('epoch = {:d}\n'.format(epoch))
            f.write('loss = {:2.4e}\n'.format(self.losses[epoch]))
            f.write('a_dif = {:2.4e}\n\n'.format(self.a_dif[epoch]))

        f.write('epoch = {:d}\n'.format(self.epochs))
        f.write('a_dif = {:2.4e}\n'.format(self.a_dif[self.epochs]))
            
        f.close()


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
