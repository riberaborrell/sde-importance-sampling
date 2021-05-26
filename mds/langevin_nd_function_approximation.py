from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.utils import get_nn_function_approximation_dir_path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class FunctionApproximation():

    def __init__(self, target_function, model, initialization='random'):
        assert target_function in ['value-f', 'control'], ''
        assert initialization in ['random', 'null', 'meta', 'hjb'], ''

        self.target_function = target_function
        self.model = model
        self.initialization = initialization

        self.dir_path = None

    def set_dir_path(self, settings_dir_path):
        self.dir_path = get_nn_function_approximation_dir_path(
            settings_dir_path,
            self.target_function,
            self.model.get_rel_path(),
            self.initialization,
        )

    def reset_parameters(self):
        self.initialization = 'random'
        for layer in self.model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

    def zero_parameters(self):
        self.initialization = 'null'
        for layer in self.model.children():
            for key in layer._parameters:
                layer._parameters[key] = torch.zeros_like(
                    layer._parameters[key], requires_grad=True
                )

    def fit_parameters_from_metadynamics(self, sde, iterations_lim=10000, N=1000, epsilon=0.01):

        # parameters
        self.initialization = 'meta'

        # load meta bias potential
        meta = sde.get_metadynamics_sampling(dt=0.001, sigma_i_meta=0.5, k=100, N_meta=10)

        # create ansatz functions from meta
        meta.sample.ansatz = GaussianAnsatz(n=sde.n)
        meta.set_ansatz_all_trajectories()

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        for i in np.arange(iterations_lim):

            # sample training data
            x = sde.sample_domain_uniformly(N=N)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # ansatz functions evaluated at the grid
            if self.target_function == 'value-f':
                pass
                #target = meta_ansatz.value_function(x)
            elif self.target_function == 'control':
                target = meta.sample.ansatz.control(x)
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

            # define loss
            inputs = self.model.forward(x_tensor)
            loss = nn.MSELoss()
            output = loss(inputs, target_tensor)
            #print('{:d}, {:2.3f}'.format(i, output))

            # stop if we have reached enough accuracy
            if output <= epsilon:
                break

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('nn fitted from metadynamics!')
        print('{:d}, {:2.3f}'.format(i, output))

    def fit_parameters_flat_controlled_potential(self, sde, iterations_lim=10000, N=1000, epsilon=0.01):
        '''
        '''
        # parameters
        self.initialization = 'flat'

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        for i in np.arange(iterations_lim):

            # sample training data
            x = sde.sample_domain_uniformly(N=N)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # ansatz functions evaluated at the grid
            if self.target_function == 'value-f':
                pass
            elif self.target_function == 'control':
                target = sde.gradient(x) / np.sqrt(2)
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

            # define loss
            inputs = self.model.forward(x_tensor)
            loss = nn.MSELoss()
            output = loss(inputs, target_tensor)

            print('{:d}, {:2.3f}'.format(i, output))

            # stop if we have reached enough accuracy
            if output <= epsilon:
                break

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('nn fitted from flat potential!')
        print('{:d}, {:2.3f}'.format(i, output))

    def fit_parameters_semiflat_controlled_potential(self, sde, iterations_lim=10000, epsilon=0.01):
        '''
        '''
        # load flat bias potential
        flatbias = sde.get_flat_bias_sampling(dt=0.01, k_lim=100, N=1000)

        # parameters
        self.initialization = 'semi-flat'

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        x = flatbias.x
        x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

        # ansatz functions evaluated at the grid
        if self.target_function == 'value-f':
            pass
        elif self.target_function == 'control':
            target = flatbias.u
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

        for i in np.arange(iterations_lim):


            # define loss
            inputs = self.model.forward(x_tensor)
            loss = nn.MSELoss()
            output = loss(inputs, target_tensor)

            print('{:d}, {:2.3f}'.format(i, output))

            # stop if we have reached enough accuracy
            if output <= epsilon:
                break

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('nn fitted from flat potential!')
        print('{:d}, {:2.3f}'.format(i, output))


    def write_parameters(self, f):
        f.write('\nFunction approximation via nn\n')
        f.write('target function: {}\n'.format(self.target_function))
        f.write('model architecture: {}\n'.format(self.model.name))
        self.model.write_parameters(f)
        f.write('initialization: {}\n'.format(self.initialization))

