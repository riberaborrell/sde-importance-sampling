from sde_importance_sampling.gaussian_nd_ansatz_functions import GaussianAnsatz

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import os


class FunctionApproximation():

    def __init__(self, target_function, model, initialization='random'):
        assert target_function in ['value-f', 'control'], ''
        assert initialization in ['random', 'null', 'meta', 'hjb'], ''

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

    def set_sgd_parameters(self, n):
        self.n_iterations_lim = 10**3
        self.N_train = 5 * 10**2
        self.losses = np.empty(self.n_iterations_lim)

    def train_parameters_with_not_controlled_potential(self, sde):
        # load trained parameters if already computed
        succ = self.load_trained_parameters(self.dir_path)
        if succ:
            print('\nnn already trained with not controlled potential.\n')
            return

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        # define loss
        loss = nn.MSELoss()

        # set sgd parameters
        self.set_sgd_parameters(sde.n)

        # target function
        target_tensor = torch.zeros(self.N_train, sde.n)

        for i in np.arange(self.n_iterations_lim):

            # sample training data
            x = sde.sample_domain_uniformly(self.N_train)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # compute loss
            inputs = self.model.forward(x_tensor)
            output = loss(inputs, target_tensor)
            if i % 1000 == 0:
                print('{:d}, {:2.3e}'.format(i, output))

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('nn trained with not controlled potential!')
        print('it.: {:d}, loss: {:2.3e}\n'.format(i, output))

        # number of iterations used
        self.n_iterations = i

        # save parameters
        self.theta = self.model.get_parameters()

        # save nn training
        self.save_trained_parameters(self.dir_path)

    def train_parameters_with_metadynamics(self, meta):

        # load trained parameters if already computed
        succ = self.load_trained_parameters(self.dir_path)
        if succ:
            print('\nnn already trained with metadynamics.\n')
            return

        # set sgd parameters

        self.n_iterations_lim = 10**3
        self.losses_train = np.empty(self.n_iterations_lim)
        self.losses_test = np.empty(self.n_iterations_lim)

        # define loss
        loss = nn.MSELoss()

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        # create ansatz functions from meta
        n = meta.sample.n
        meta.sample.ansatz = GaussianAnsatz(n=n, beta=meta.sample.beta, normalized=False)
        meta.set_ansatz()

        # get data point from uniform grid 
        if meta.sample.n == 1:
            h = 0.001
        elif meta.sample.n == 2:
            h = 0.005
        elif meta.sample.n == 3:
            h = 0.1
        elif meta.sample.n == 4:
            h = 0.5
        meta.sample.discretize_domain(h)
        data = meta.sample.get_flat_domain_h()
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

        # evaluate meta control at the training data
        target_train = meta.sample.ansatz.control(x_train)
        target_test = meta.sample.ansatz.control(x_test)
        target_train_tensor = torch.tensor(target_train, requires_grad=False, dtype=torch.float32)
        target_test_tensor = torch.tensor(target_test, requires_grad=False, dtype=torch.float32)

        for i in np.arange(self.n_iterations_lim):

            # evaluate nn control at the training data
            inputs_train = self.model.forward(x_train_tensor)
            inputs_test = self.model.forward(x_test_tensor)

            # compute loss
            output_train = loss(inputs_train, target_train_tensor)
            output_test = loss(inputs_test, target_test_tensor).detach().numpy()
            if i % 100 == 0:
                print('it.: {:d}, loss (train): {:2.3e}, loss (test): {:2.3e}' \
                      ''.format(i, output_train, output_test))

            # save loss
            self.losses_train[i] = output_train.detach().numpy()
            self.losses_test[i] = output_test

            # compute gradients
            output_train.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('nn trained from metadynamics!')
        print('it.: {:d}, loss (train): {:2.3e}, loss (test): {:2.3e}' \
              ''.format(i, output_train, output_test))

        # number of iterations used
        self.n_iterations = i

        # save parameters
        self.theta = self.model.get_parameters()

        # save nn training
        self.save_trained_parameters(self.dir_path)

    def train_parameters_with_metadynamics_2(self, meta):

        # load trained parameters if already computed
        succ = self.load_trained_parameters(self.dir_path)
        if succ:
            print('\nnn already trained with metadynamics.\n')
            return

        # create ansatz functions from meta
        n = meta.sample.n
        meta.sample.ansatz = GaussianAnsatz(n=n, normalized=False, beta=meta.sample.beta)
        meta.set_ansatz_cumulative()

        # define loss
        loss = nn.MSELoss()

        # set sgd parameters
        self.set_sgd_parameters(meta.sample.n)

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        for i in np.arange(self.n_iterations_lim):

            # sample training data
            x = meta.sample.sample_domain_uniformly(N=self.N_train)
            x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)

            # ansatz functions evaluated at the grid
            if self.target_function == 'value-f':
                pass
                #target = meta_ansatz.value_function(x)
            elif self.target_function == 'control':
                target = meta.sample.ansatz.control(x)
            target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

            inputs = self.model.forward(x_tensor)

            # compute loss
            output = loss(inputs, target_tensor)
            if i % 1000 == 0:
                print('it.: {:d}, loss: {:2.3e}'.format(i, output))

            # save loss
            self.losses[i] = output.detach().numpy()

            # compute gradients
            output.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        print('nn trained from metadynamics!')
        print('it.: {:d}, loss: {:2.3e}\n'.format(i, self.losses[i]))

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

        # save npz file
        file_path = os.path.join(dir_path, 'nn.npz')
        np.savez(
            file_path,
            n_iterations_lim=self.n_iterations_lim,
            #n_iterations=self.n_iterations,
            N_data=self.N_data,
            N_train=self.N_train,
            N_test=self.N_test,
            #epsilon=self.epsilon,
            theta=self.theta,
            losses_train=self.losses_train,
            losses_test=self.losses_test,
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


    def fit_parameters_flat_controlled_potential(self, sde, n_iterations_lim=10000, N=1000, epsilon=0.01):
        '''
        '''
        # parameters
        self.initialization = 'flat'

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        for i in np.arange(n_iterations_lim):

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

    def fit_parameters_semiflat_controlled_potential(self, sde, n_iterations_lim=10000, epsilon=0.01):
        '''
        '''
        # load flat bias potential
        flatbias = sde.get_flat_bias_sampling(dt=0.01, k_lim=100, N=1000)
        N_flat = flatbias.x.shape[0]

        # add boundary condition to training data
        x_boundary = sde.sample_domain_boundary_uniformly_vec(N_flat // 10)
        x = np.vstack((flatbias.x, x_boundary))

        if self.target_function == 'value-f':
            pass
        elif self.target_function == 'control':
            u_boundary = np.zeros((N_flat // 10, sde.n))
            target = np.vstack((flatbias.u, u_boundary))

        # tensorize variables
        x_tensor = torch.tensor(x, requires_grad=False, dtype=torch.float32)
        target_tensor = torch.tensor(target, requires_grad=False, dtype=torch.float32)

        # parameters
        self.initialization = 'semi-flat'

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        for i in np.arange(n_iterations_lim):

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

