from mds.potentials_and_gradients_2d import POTENTIAL_NAMES

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
        '--potential',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='2d_4well',
        help='Set the potential for the 2D MD SDE. Default: quadruple well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1, 1, 1, 1],
        help='Set the parameters for the 2D MD SDE potential. Default: [1, 1, 1, 1]',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 2D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        nargs=2,
        type=float,
        default=[-1, -1],
        help='Set the initial value of the process. Default: [-1, -1]',
    )
    parser.add_argument(
        '--domain',
        dest='domain',
        nargs=4,
        type=float,
        default=[-3, 3, -3, 3],
        help='Set the domain set. Default: [[-3, 3],[-3, 3]]',
    )
    parser.add_argument(
        '--target-set',
        dest='target_set',
        nargs=4,
        type=float,
        default=[0.9, 1.1, 0.9, 1.1],
        help='Set the target set interval. Default: [[0.9, 1.1], [0.9, 1.1]]',
    )
    parser.add_argument(
        '--h',
        dest='h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--M',
        dest='M',
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
        '--N-lim',
        dest='N_lim',
        type=int,
        default=10**8,
        help='Set maximal number of time steps. Default: 100.000.000',
    )
    parser.add_argument(
        '--m',
        dest='m',
        nargs=2,
        type=int,
        default=[10, 10],
        help='Set the number of uniformly distributed ansatz functions per axis \
              that you want to use. Default: [50, 50]',
    )
    parser.add_argument(
        '--sigma',
        dest='sigma',
        nargs=2,
        type=float,
        default=[1, 1],
        help='Set the standard deviation of the ansatz functions. Default: [1, 1]',
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
        default=1,
        help='Set learning rate. Default: 1',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser
