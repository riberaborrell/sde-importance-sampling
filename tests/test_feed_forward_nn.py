from mds.neural_networks import TwoLayerNN, FeedForwardNN

import numpy as np
import pytest
import torch

class TestTwoLayerNN:

    @pytest.fixture
    def random_inputs(self, N, n):
        ''' generates random input data
        '''
        return torch.rand(N, n)

    @pytest.fixture
    def random_two_layer_nn(self, n, d1):
        ''' initializes TwoLayerNN with random weights
        '''
        return TwoLayerNN(n, d1, n)

    def test_initialization(self, random_inputs, random_two_layer_nn):
        '''
        '''
        model = random_two_layer_nn
        output = model.forward(random_inputs)

        N, n = random_inputs.size()
        assert output.size() == (N, n)


class TestFeedForwardNN:
    @pytest.fixture
    def random_inputs(self, N, n):
        ''' generates random input data
        '''
        return torch.rand(N, n)

    @pytest.fixture
    def random_feed_forward_nn(self, n):
        ''' initializes FeedForwardNN with random weights
        '''
        d_layers = [n, 30, 30, n]
        return FeedForwardNN(d_layers)

    def test_initialization(self, random_inputs, random_feed_forward_nn):
        '''
        '''
        model = random_feed_forward_nn
        breakpoint()
        output = model.forward(random_inputs)

        N, n = random_inputs.size()
        assert output.size() == (N, n)

    def test_get_parameters(self, random_inputs, random_feed_forward_nn):
        model = random_feed_forward_nn
        flattened_theta = model.get_parameters()
        assert flattened_theta.ndim == 1, ''
        assert flattened_theta.shape[0] == model.d_flat, ''
