from mds.neural_networks import TwoLayerNet

import numpy as np
import pytest
import torch

class TestTwoLayerNet:

    @pytest.fixture
    def random_inputs(self, N, n):
        ''' generates random input data
        '''
        return torch.rand(N, n)

    @pytest.fixture
    def random_two_layer_nn(self, n, d1):
        ''' initializes TwoLayerNet with random weights
        '''
        return TwoLayerNet(n, d1, n)

    def test_initialization(self, random_inputs, random_two_layer_nn):
        '''
        '''
        model = random_two_layer_nn
        output = model.forward(random_inputs)

        N, n = random_inputs.size()
        assert output.size() == (N, n)
