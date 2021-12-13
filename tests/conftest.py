from sde_importance_sampling.langevin_nd_sde import POTENTIAL_NAMES

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
        '--problem-name',
        dest='problem_name',
        choices=['langevin_det-t', 'langevin_stop-t'],
        default='langevin_stop-t',
        help='Set type of problem. Default: overdamped langevin with stopping times',
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
        '--dt',
        dest='dt',
        type=float,
        default=0.001,
        help='Set dt. Default: 0.001',
    )
    parser.addoption(
        '--k-lim',
        dest='k_lim',
        type=int,
        default=10**8,
        help='Set maximal number of time steps. Default: 100.000.000',
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
def problem_name(request):
    return request.config.getoption('problem_name')

@pytest.fixture(scope='session')
def potential_name(request):
    return request.config.getoption('potential_name')

@pytest.fixture(scope='session')
def seed(request):
    return request.config.getoption('seed')

@pytest.fixture(scope='session')
def n(request):
    return request.config.getoption('n')

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
def dt(request):
    return request.config.getoption('dt')

@pytest.fixture(scope='session')
def k_lim(request):
    return request.config.getoption('k_lim')

@pytest.fixture(scope='session')
def m(request):
    return request.config.getoption('m')

@pytest.fixture(scope='session')
def N(request):
    return request.config.getoption('N')

@pytest.fixture(scope='session')
def d1(request):
    return request.config.getoption('d1')
