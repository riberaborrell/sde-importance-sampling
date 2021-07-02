from mds.potentials_and_gradients_nd import POTENTIAL_NAMES

import pytest

import numpy as np

def pytest_addoption(parser):
    parser.addoption(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.addoption(
        '--n',
        dest='n',
        type=int,
        default=1,
        help='Set the dimension n. Default: 1',
    )
    parser.addoption(
        '--potential-name',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='nd_2well',
        help='Set type of potential. Default: double well',
    )
    parser.addoption(
        '--alpha-i',
        dest='alpha_i',
        type=float,
        default=1,
        help='Set nd double well barrier height. Default: 1',
    )
    parser.addoption(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the beta parameter. Default: 1',
    )
    parser.addoption(
        '--xzero-i',
        dest='xzero_i',
        type=float,
        default=-1,
        help='Set the initial posicion of the process at each axis. Default: -1',
    )
    parser.addoption(
        '--m',
        dest='m',
        type=int,
        default=30,
        help='Set number of ansatz functions. Default: 30',
    )
    parser.addoption(
        '--N',
        dest='N',
        type=int,
        default=100,
        help='Set number of batch size. Default: 100',
    )
    parser.addoption(
        '--d1',
        dest='d1',
        type=int,
        default=10,
        help='Set dimmension of the first layer of the nn. Default: 10',
    )

@pytest.fixture(scope='session')
def seed(request):
    return request.config.getoption('seed')

@pytest.fixture(scope='session')
def n(request):
    return request.config.getoption('n')

@pytest.fixture(scope='session')
def potential_name(request):
    return request.config.getoption('potential_name')

@pytest.fixture(scope='session')
def alpha_i(request):
    return request.config.getoption('alpha_i')

@pytest.fixture(scope='session')
def beta(request):
    return request.config.getoption('beta')

@pytest.fixture(scope='session')
def xzero_i(request):
    return request.config.getoption('xzero_i')

@pytest.fixture(scope='session')
def m(request):
    return request.config.getoption('m')

@pytest.fixture(scope='session')
def N(request):
    return request.config.getoption('N')

@pytest.fixture(scope='session')
def d1(request):
    return request.config.getoption('d1')
