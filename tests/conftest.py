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

@pytest.fixture(scope='session')
def seed(request):
    return request.config.getoption('seed')

@pytest.fixture(scope='session')
def n(request):
    return request.config.getoption('n')

@pytest.fixture(scope='session')
def m(request):
    return request.config.getoption('m')

@pytest.fixture(scope='session')
def N(request):
    return request.config.getoption('N')
