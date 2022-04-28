import numpy as np
import pytest
import torch

from function_approximation.models import TwoLayerNN, \
                                          FeedForwardNN, \
                                          DenseNN, \
                                          SequentialNN


class TestTwoLayerNN:

    @pytest.fixture
    def random_inputs(self, K, d):
        ''' generates random input data
        '''
        return torch.rand(K, d, dtype=torch.float32)

    @pytest.fixture
    def random_two_layer_nn(self, d, d1):
        ''' initializes TwoLayerNN with random weights
        '''
        return TwoLayerNN(d, d1, d)

    def test_initialization(self, random_inputs, random_two_layer_nn):
        '''
        '''
        model = random_two_layer_nn
        output = model.forward(random_inputs)

        K, d = random_inputs.size()
        assert output.size() == (K, d)


class TestFeedForwardNN:
    @pytest.fixture
    def random_inputs(self, K, d):
        ''' generates random input data
        '''
        return torch.rand(K, d, dtype=torch.float32)

    @pytest.fixture
    def random_feed_forward_nn(self, d):
        ''' initializes FeedForwardNN with random weights
        '''
        d_layers = [d, 30, 30, d]
        return FeedForwardNN(d_layers)

    def test_initialization(self, random_inputs, random_feed_forward_nn):
        '''
        '''
        model = random_feed_forward_nn
        output = model.forward(random_inputs)

        K, d = random_inputs.size()
        assert output.size() == (K, d)

    def test_get_parameters(self, random_inputs, random_feed_forward_nn):
        model = random_feed_forward_nn
        flattened_theta = model.get_parameters()
        assert flattened_theta.ndim == 1, ''
        assert flattened_theta.shape[0] == model.d_flat, ''

class TestSequentialNN:
    @pytest.fixture
    def random_inputs(self, K, d):
        ''' generates random input data
        '''
        return torch.rand(K, d, dtype=torch.float32)

    @pytest.fixture
    def sequential_nn(self, d):
        ''' initializes SequentialNN with K feed forward models withrandom weights
        '''
        d_layers = [d, 30, 30, d]
        activation_type = 'tanh'
        K = 201
        return SequentialNN(K, d_layers, activation_type)

    def test_initialization(self, random_inputs, sequential_nn):
        '''
        '''
        model = sequential_nn
        output = model.forward(0, random_inputs)

        K, d = random_inputs.size()
        assert output.size() == (K, d)

    def test_get_parameters(self, random_inputs, sequential_nn):
        model = sequential_nn
        flattened_theta = model.get_parameters()
        assert flattened_theta.ndim == 1, ''
        assert flattened_theta.shape[0] == model.d_flat, ''
