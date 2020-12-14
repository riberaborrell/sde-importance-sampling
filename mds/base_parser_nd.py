from mds.potentials_and_gradients_nd import POTENTIAL_NAMES

import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--n',
        dest='n',
        type=int,
        default=3,
        help='Set the dimension n. Default: 3',
    )
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='nd_2well',
        help='Set type of potential. Default: double well',
    )
    parser.add_argument(
        '--alpha_i',
        dest='alpha_i',
        type=float,
        default=1,
        help='Set nd double well barrier height. Default: 1',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the beta parameter. Default: 1',
    )
    parser.add_argument(
        '--xzero_i',
        dest='xzero_i',
        type=float,
        default=-1,
        help='Set the initial posicion of the process at each axis. Default: -1',
    )
    parser.add_argument(
        '--h',
        dest='h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--N',
        dest='N',
        type=int,
        default=10**4,
        help='Set number of trajectories to sample. Default: 10.000',
    )
    parser.add_argument(
        '--dt',
        dest='dt',
        type=float,
        default=0.001,
        help='Set dt. Default: 0.001',
    )
    parser.add_argument(
        '--k-lim',
        dest='k_lim',
        type=int,
        default=10**8,
        help='Set maximal number of time steps. Default: 100.000.000',
    )
    parser.add_argument(
        '--m',
        dest='m',
        type=int,
        default=30,
        help='Set number of uniformly distributed ansatz functions per axis \
              that you want to use. Default: 30',
    )
    parser.add_argument(
        '--sigma',
        dest='sigma',
        type=float,
        default=0.2,
        help='Set the covariance matrix of the ansatz functions. Default: 0.2',
    )
    parser.add_argument(
        '--theta',
        dest='theta',
        choices=['optimal', 'meta', 'gd', 'null'],
        default='optimal',
        help='Type of control. Default: optimal',
    )
    parser.add_argument(
        '--theta-init',
        dest='theta_init',
        choices=['null', 'meta', 'optimal'],
        default='null',
        help='Type of initial control. Default: null',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser