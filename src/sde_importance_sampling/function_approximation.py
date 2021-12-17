from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import os


class FunctionApproximation():

    def __init__(self, target_function, model, initialization='random'):
        assert target_function in ['value-f', 'control'], ''
        assert initialization in ['random', 'null', 'not-controlled', 'meta', 'hjb'], ''

        self.target_function = target_function
        self.model = model
        self.initialization = initialization

        self.dir_path = None

    def set_dir_path(self, root_dir_path):

        if self.initialization == 'meta':
            initialization_str = ''
        else:
            initialization_str = 'theta_{}'.format(self.initialization)

        self.dir_path = os.path.join(
            root_dir_path,
            'appr_{}'.format(self.target_function),
            self.model.get_rel_path(),
            initialization_str,
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

    def train_parameters_classic(self, sde=None, sol_hjb=None, meta=None):

        # assume type of initialization
        assert self.initialization in ['not-controlled', 'meta', 'hjb'], ''

        # load trained parameters if already computed
        succ = self.load_trained_parameters(self.dir_path)
        if succ:
            print('\nnn already trained with {}.\n'.format(self.initialization))
            return

        # set sgd parameters
        self.n_iterations_lim = 10**3
        self.losses_train = np.empty(self.n_iterations_lim)
        self.losses_test = np.empty(self.n_iterations_lim)

        # define mean square error loss
        loss = nn.MSELoss()

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        # get dimension of the problem
        if self.initialization == 'not-controlled':
            assert sde is not None, ''
            n = sde.n
        elif self.initialization == 'meta':
            assert meta is not None, ''
            n = meta.sample.n
        elif self.initialization == 'hjb':
            assert sol_hjb is not None, ''
            n = sol_hjb.n

        # choose discretization size depending on n 
        if n == 1:
            h = 0.001
        elif n == 2:
            h = 0.005
        elif n == 3:
            h = 0.1
        elif n == 4:
            h = 0.5
        else:
            msg = '\n this approximation method is not implemented for n = {:d}.\n' \
                  ''.format(n)
            print(msg)
            return

        # get data points
        if self.initialization == 'not-controlled':
            sde.discretize_domain(h)
            data = sde.get_flat_domain_h()
        elif self.initialization == 'meta':
            meta.sample.discretize_domain(h)
            data = meta.sample.get_flat_domain_h()
        elif self.initialization == 'hjb':
            sol_hjb.discretize_domain(h)
            data = sol_hjb.get_flat_domain_h()
        self.N_data = data.shape[0]

        # split into training and test data
        self.N_train = int(0.75 * self.N_data)
        np.random.shuffle(data)
        x_train = data[:self.N_train]
        x_test = data[self.N_train:]
        self.N_test = x_test.shape[0]

        # convert to tensors
        x_train_tensor = torch.tensor(x_train, requires_grad=False, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, requires_grad=False, dtype=torch.float32)

        # evaluate target function at the training data
        if self.initialization == 'not-controlled':

            # zero control
            target_train = np.zeros((self.N_train, n))
            target_test = np.zeros((self.N_test, n))

        elif self.initialization == 'meta':

            # create ansatz functions with meta parameters
            meta.sample.ansatz = GaussianAnsatz(n=n, beta=meta.sample.beta, normalized=False)
            meta.set_ansatz()

            # evaluation
            target_train = meta.sample.ansatz.control(x_train)
            target_test = meta.sample.ansatz.control(x_test)

        elif self.initialization == 'hjb':
            #TODO: write method in the hjb solver which given an input returns the control
            pass

        # tensorize target function evaluation
        target_train_tensor = torch.tensor(
            target_train,
            requires_grad=False,
            dtype=torch.float32,
        )
        target_test_tensor = torch.tensor(
            target_test,
            requires_grad=False,
            dtype=torch.float32,
        )

        for i in np.arange(self.n_iterations_lim):

            # evaluate nn control at the training data
            inputs_train = self.model.forward(x_train_tensor)
            inputs_test = self.model.forward(x_test_tensor)

            # compute loss
            output_train = loss(inputs_train, target_train_tensor)
            output_test = loss(inputs_test, target_test_tensor).detach().numpy()
            if i % 100 == 0:
                msg = 'it.: {:d}, loss (train): {:2.3e}, loss (test): {:2.3e}' \
                      ''.format(i, output_train, output_test)
                print(msg)

            # save loss
            self.losses_train[i] = output_train.detach().numpy()
            self.losses_test[i] = output_test

            # compute gradients
            output_train.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('\nnn trained with {}!'.format(self.initialization))
        print('it.: {:d}, loss (train): {:2.3e}, loss (test): {:2.3e} \n' \
              ''.format(i, output_train, output_test))

        # number of iterations used
        self.n_iterations = i

        # save parameters
        self.theta = self.model.get_parameters()

        # save nn training
        self.save_trained_parameters(self.dir_path)

    def train_parameters_alternative(self, sde=None, sol_hjb=None, meta=None):

        # assume type of initialization
        assert self.initialization in ['not-controlled', 'meta', 'hjb'], ''

        # load trained parameters if already computed
        succ = self.load_trained_parameters(self.dir_path)
        if succ:
            print('\nnn already trained with metadynamics.\n')
            return

        # get dimension of the problem
        if self.initialization == 'not-controlled':
            assert sde is not None, ''
            n = sde.n
        elif self.initialization == 'meta':
            assert meta is not None, ''
            n = meta.sample.n
        elif self.initialization == 'hjb':
            assert sol_hjb is not None, ''
            n = sol_hjb.n

        # define loss
        loss = nn.MSELoss()

        # set sgd parameters
        self.n_iterations_lim = 10**3
        self.N_train = 5 * 10**2
        self.losses_train = np.empty(self.n_iterations_lim)

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        if self.initialization == 'meta':

            # create ansatz functions from meta
            meta.sample.ansatz = GaussianAnsatz(n=n, normalized=False, beta=meta.sample.beta)
            meta.set_ansatz()

        # compute target function if target function is zero
        if self.initialization == 'not-controlled':
            target = np.zeros((self.N_train, n))

        for i in np.arange(self.n_iterations_lim):

            # sample training data
            if self.initialization == 'not-controlled':
                x = sde.sample_domain_uniformly(N=self.N_train)
            elif self.initialization == 'meta':
                x = meta.sample.sample_domain_uniformly(N=self.N_train)
            elif self.initialization == 'hjb':
                x = sol_hjb.sample_domain_uniformly(N=self.N_train)

            # tensorize
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # evaluate target function at the training data
            if self.initialization == 'not-controlled':
                pass

            elif self.initialization == 'meta':
                target = meta.sample.ansatz.control(x)

            elif self.initialization == 'hjb':
                pass

            # tensorize target function evaluation
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

            # compute loss
            inputs = self.model.forward(x_tensor)
            output = loss(inputs, target_tensor)
            if i % 1000 == 0:
                print('it.: {:d}, loss: {:2.3e}'.format(i, output))

            # save loss
            self.losses_train[i] = output.detach().numpy()

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('\nnn trained with {}!'.format(self.initialization))
        print('it.: {:d}, loss: {:2.3e}\n'.format(i, output))

        # number of iterations used
        self.n_iterations = i

        # save parameters
        self.theta = self.model.get_parameters()

        # save nn training
        self.save_trained_parameters(self.dir_path)

    def save_trained_parameters(self, dir_path):
        # create directory if dir_path does not exist
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        #TODO: adapt method to save parameters of the alternative approach
        # save npz file
        file_path = os.path.join(dir_path, 'nn.npz')
        np.savez(
            file_path,
            n_iterations_lim=self.n_iterations_lim,
            #n_iterations=self.n_iterations,
            #N_data=self.N_data,
            #N_train=self.N_train,
            #N_test=self.N_test,
            #epsilon=self.epsilon,
            theta=self.theta,
            losses_train=self.losses_train,
            #losses_test=self.losses_test,
        )

    def load_trained_parameters(self, dir_path):
        try:
            # load arrrays of the npz file
            data = np.load(
                  os.path.join(dir_path, 'nn.npz'),
                  allow_pickle=True,
            )
            for file_name in data.files:
                setattr(self, file_name, data[file_name])

            # load parameters in the model
            self.model.load_parameters(self.theta)
            return True

        except:
            print('no trained nn found')
            return False


    def write_parameters(self, f):
        f.write('\nFunction approximation via nn\n')
        f.write('target function: {}\n'.format(self.target_function))
        f.write('model architecture: {}\n'.format(self.model.name))
        self.model.write_parameters(f)
        f.write('initialization: {}\n'.format(self.initialization))
