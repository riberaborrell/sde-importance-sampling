import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz


class FunctionApproximation(object):
    '''

    Attributes
    ----------
    target_function: str
        function that we want to approximate: control or value function
    model: Pytorch nn model object
        model which aims to approximate the target function
    initialization: str
        type of parameters initialization
    training_algorithm: str
        type of training algorithm

    # stochastic gradient based optimization
    n_iterations_lim: int
        maximal number of iterations
    K_data: int
        number of points used in a batch
    K_train: int
        number of points used in a batch for training
    K_test: int
        number of points used in a batch for testing

    losses_train: array
        loss function at the train points in the classic training
    losses_test: array
        loss function at the test points in the classic training

    # computational time
    ct_initial: float
        initial computational time
    ct_time: float
        final computational time
    ct: float

    # directory path
    dir_path: str
        absolute path of the directory for function approximation

    Methods
    -------
    set_dir_path(root_dir_path)

    reset_parameters()

    zero_parameters()

    start_timer()

    stop_timer()

    train_parameters(sde=None, sol_hjb=None, meta=None)

    train_parameters_classic(sde, sol_hjb=None, meta=None)

    train_parameters_alternative(sde=None, sol_hjb=None, meta=None)

    save_trained_parameters(dir_path)

    load_trained_parameters(dir_path)

    write_parameters(f)

    plot_mse_loss(dir_path=None, file_name='mse-loss')

    '''

    def __init__(self, target_function, model, initialization='random', training_algorithm=None):
        ''' init method

        Parameters
        ----------
        target_function: str
            function that we want to approximate: control or value function
        model: Pytorch nn model object
            model which aims to approximate the target function
        initialization: str
            type of parameters initialization
        training_algorithm: str
            type of training algorithm
        '''

        assert target_function in ['value-f', 'control'], ''
        assert initialization in ['random', 'null', 'not-controlled', 'meta', 'hjb'], ''

        self.target_function = target_function
        self.model = model
        self.initialization = initialization
        self.training_algorithm = training_algorithm

    def set_dir_path(self, root_dir_path):
        ''' sets directory path for the function approximation
        '''

        # initialization
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

    def start_timer(self):
        ''' start timer
        '''
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        ''' stop timer
        '''
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def reset_parameters(self):
        ''' reset parameters of the chosen model. The weights are chosen "random". See
            pytorch documentation for details.
        '''
        self.initialization = 'random'
        for layer in self.model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

    def zero_parameters(self):
        ''' set parameters of the chosen model to zero.
        '''
        self.initialization = 'null'
        for layer in self.model.children():
            for key in layer._parameters:
                layer._parameters[key] = torch.zeros_like(
                    layer._parameters[key], requires_grad=True
                )

    def train_parameters(self, sde=None, sol_hjb=None, meta=None):
        ''' train parameters such that our model approximates the chosen target function.

        Parameters
        ----------
        sde: LangevinSDE object, optional
            langevin sde object
        sol_hjb: HJBSolver object, optional
            hjb solver object
        meta: Metadynamics object, optional
            metadynamics object
        '''

        # assume type of initialization
        assert self.initialization in ['not-controlled', 'meta', 'hjb'], ''

        # load trained parameters if already computed
        succ = self.load_trained_parameters(self.dir_path)
        if succ:
            print('\nnn already trained with {}.\n'.format(self.initialization))
            return

        # start timer
        self.start_timer()

        # get dimension of the problem
        if self.initialization == 'not-controlled':
            assert sde is not None, ''
        elif self.initialization == 'meta':
            assert meta is not None, ''
            sde = meta.sample.sde
        elif self.initialization == 'hjb':
            assert sol_hjb is not None, ''
            sde = sol_hjb.sde

        # choose training algorithm
        if self.training_algorithm == 'classic':
            self.train_parameters_classic(sde, sol_hjb, meta)
        elif self.training_algorithm == 'alternative':
            self.train_parameters_alternative(sde, sol_hjb, meta)
        else:
            return

        # stop timer
        self.stop_timer()

        # save nn training parameters and statistics
        self.save_trained_parameters(self.dir_path)

    def train_parameters_classic(self, sde, sol_hjb=None, meta=None):
        ''' trains parameters in a supervised fashion. The same data points set is used in every
            iteration of the gradient based optimization.

        Parameters
        ----------
        sde: LangevinSDE object
            langevin sde object
        sol_hjb: HJBSolver object, optional
            hjb solver object
        meta: Metadynamics object, optional
            metadynamics object
        '''

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
        # choose discretization size depending on d 
        if sde.d == 1:
            h = 0.001
        elif sde.d == 2:
            h = 0.005
        elif sde.d == 3:
            h = 0.05
        elif sde.d == 4:
            h = 0.1
        else:
            msg = '\n this approximation method is not implemented for n = {:d}.\n' \
                  ''.format(n)
            print(msg)
            return

        # get data points
        sde.discretize_domain(h)
        data = sde.get_flat_domain_h()
        self.K_data = data.shape[0]

        # split into training and test data
        self.K_train = int(0.75 * self.K_data)
        np.random.shuffle(data)
        x_train = data[:self.K_train]
        x_test = data[self.K_train:]
        self.K_test = x_test.shape[0]

        # convert to tensors
        x_train_tensor = torch.tensor(x_train, requires_grad=False, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, requires_grad=False, dtype=torch.float32)

        # evaluate target function at the training data

        # zero control
        if self.initialization == 'not-controlled':
            target_train = np.zeros((self.K_train, sde.d))
            target_test = np.zeros((self.K_test, sde.d))

        # metadynamics
        elif self.initialization == 'meta':

            # create ansatz functions with meta parameters
            meta.sample.ansatz = GaussianAnsatz(sde, normalized=False)
            meta.set_ansatz()

            # evaluation
            target_train = meta.sample.ansatz.control(x_train)
            target_test = meta.sample.ansatz.control(x_test)

        # hjb control
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

    def train_parameters_alternative(self, sde, sol_hjb=None, meta=None):
        ''' trains parameters in a non supervised fashion. The data points set is newly sampled
            at every iteration.

        Parameters
        ----------
        sde: LangevinSDE object
            langevin sde object
        sol_hjb: HJBSolver object, optional
            hjb solver object
        meta: Metadynamics object, optional
            metadynamics object
        '''

        # set sgd parameters

        # not controlled
        if self.initialization == 'not-controlled':
            self.n_iterations_lim = 10**4
            self.K_train = 10**3

        # meta
        elif self.initialization == 'meta':
            self.n_iterations_lim = 10**4
            self.K_train = 10**3

            K_centers = int(0.95 * self.K_train)
            m = meta.ms.sum()
            K_gauss = K_centers // m
            K_uniform = self.K_train - K_centers + K_centers % m

        # hjb
        elif self.initialization == 'hjb':
            self.n_iterations_lim = 10**5
            self.K_train = 10**3

        self.losses_train = np.empty(self.n_iterations_lim)

        # define mean square error loss
        loss = nn.MSELoss()

        # define optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )

        if self.initialization == 'meta':

            # create ansatz functions from meta
            meta.sample.ansatz = GaussianAnsatz(sde, normalized=False)
            meta.set_ansatz()

        # compute target function if target function is zero
        if self.initialization == 'not-controlled':
            target = np.zeros((self.K_train, sde.d))

        for i in np.arange(self.n_iterations_lim):

            # sample training data

            if self.initialization == 'not-controlled':
                x = sde.sample_domain_uniformly(K=self.K_train)

            elif self.initialization == 'meta':

                # preallocate training points
                x = np.empty((self.K_train, sde.d))

                # sample normal distributed for each gaussian
                for l in range(m):
                    x[K_gauss*l:K_gauss*(l+1), :] = sde.sample_multivariate_normal(
                        mean=meta.means[l],
                        cov=0.1 * np.eye(sde.d),
                        K=K_gauss,
                    )

                # sample uniform distributed in the whole domain
                x[(l+1)*K_gauss:, :] = sde.sample_domain_uniformly(
                    K=K_uniform,
                    subset=np.full((sde.d, 2), [-2, 2]),
                )

            elif self.initialization == 'hjb':
                x = sol_hjb.sample_domain_uniformly(K=self.K_train)

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
            if i % 1 == 0:
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


    def save_trained_parameters(self, dir_path):
        ''' save model parameters and training process

        Parameters
        ----------
        dir_path: str
            directory path
        '''
        # set file path
        file_path = os.path.join(dir_path, 'nn.npz')

        # create directory if dir_path does not exist
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        # model parameters after training
        self.theta = self.model.get_parameters()

        # create dictionary
        files_dict = {}

        files_dict['training_algorithm'] = self.training_algorithm
        files_dict['theta'] = self.theta
        files_dict['n_iterations_lim'] = self.n_iterations_lim

        if self.training_algorithm == 'classic':
            files_dict['K_data'] = self.K_data
            files_dict['K_train'] = self.K_train
            files_dict['K_test'] = self.K_test
            files_dict['losses_train'] = self.losses_train
            files_dict['losses_test'] = self.losses_test

        else: # self.training_algorithm == 'alternative':
            files_dict['K_train'] = self.K_train
            files_dict['losses_train'] = self.losses_train

        # computational time
        files_dict['ct'] = self.ct

        # accouracy is fixed
        # files_dict['n_iterations' ] = self.n_iterations
        # files_dict['epsilon'] = self.epsilon

        # save npz file
        np.savez(file_path, **files_dict)

    def load_trained_parameters(self, dir_path):
        ''' load model parameters and training process

        Parameters
        ----------
        dir_path: str
            directory path
        '''
        try:
            # load arrrays of the npz file
            data = np.load(
                  os.path.join(dir_path, 'nn.npz'),
                  allow_pickle=True,
            )
            for npz_file_name in data.files:
                if data[npz_file_name].ndim == 0:
                    setattr(self, npz_file_name, data[npz_file_name][()])
                else:
                    setattr(self, npz_file_name, data[npz_file_name])

            # load parameters in the model
            self.model.load_parameters(self.theta)
            return True

        except:
            print('no trained nn found')
            return False


    def write_parameters(self, f):
        ''' writes in the given file the relevant information of the iteration

        Parameters
        ----------
        f: pyObject
            file where we write on
        '''

        f.write('\nFunction approximation via nn\n')
        f.write('target function: {}\n'.format(self.target_function))
        f.write('model architecture: {}\n'.format(self.model.name))
        self.model.write_parameters(f)
        f.write('initialization: {}\n'.format(self.initialization))

        if hasattr(self, 'training_algorithm'):
            f.write('training-algorithm: {}\n'.format(self.training_algorithm))

    def plot_mse_loss(self, dir_path=None, file_name='mse-loss'):
        ''' for the classic training approach plots the training and the test loss. For
            the alternative training approach plots just the training loss.

        Parameters
        ----------
        dir_path: str, optional
            directory for the plot
        file_name: str, optional
            name of the plot file
        '''
        import matplotlib.pyplot as plt
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        # initialize figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )

        # set number of iterations
        x = np.arange(self.n_iterations_lim)

        if self.training_algorithm == 'classic':
            fig.ax.semilogy(x, self.losses_train, alpha=0.5, label='training')
            fig.ax.semilogy(x, self.losses_test, alpha=0.5, label='test')
        elif self.training_algorithm == 'alternative':
            fig.ax.semilogy(x, self.losses_train, alpha=0.5, label='training')

        fig.set_title('MSE loss')
        fig.set_xlabel('SGD iterations')
        #fig.set_legend_location('upper right')
        fig.ax.legend()
        plt.show()
