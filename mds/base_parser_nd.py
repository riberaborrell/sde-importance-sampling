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
        default=1,
        help='Set the dimension n. Default: 1',
    )
    parser.add_argument(
        '--potential-name',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='nd_2well',
        help='Set type of potential. Default: double well',
    )
    parser.add_argument(
        '--alpha-i',
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
        '--xzero-i',
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
        '--h-hjb',
        dest='h_hjb',
        type=float,
        default=0.1,
        help='Set the discretization step size for the hjb sol. Default: 0.1',
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
        '--distributed',
        dest='distributed',
        choices=['uniform', 'meta'],
        default='uniform',
        help='Type of ansatz distribution. Default: uniform',
    )
    parser.add_argument(
        '--m-i',
        dest='m_i',
        type=int,
        default=30,
        help='Set number of uniformly distributed ansatz functions per axis \
              that you want to use. Default: 30',
    )
    parser.add_argument(
        '--sigma-i',
        dest='sigma_i',
        type=float,
        default=0.5,
        help='Set the diagonal of the covariance matrix of the ansatz functions. Default: 0.5',
    )
    parser.add_argument(
        '--dt-meta',
        dest='dt_meta',
        type=float,
        default=0.001,
        help='Set dt. Default: 0.001',
    )
    parser.add_argument(
        '--N-meta',
        dest='N_meta',
        type=int,
        default=1,
        help='Set number of trajectories to sample. Default: 1',
    )
    parser.add_argument(
        '--sigma-i-meta',
        dest='sigma_i_meta',
        type=float,
        default=0.5,
        help='Set the diagonal of the covariance matrix of the ansatz functions. Default: 0.5',
    )
    parser.add_argument(
        '--k',
        dest='k',
        type=int,
        default=100,
        help='Steps before adding a bias function. Default: 100',
    )
    parser.add_argument(
        '--theta',
        dest='theta',
        choices=['random', 'null', 'meta', 'flat', 'semi-flat', 'hjb'],
        default='null',
        help='Type of control. Default: null',
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
        '--N-gd',
        dest='N_gd',
        type=int,
        default=1000,
        help='Set number of trajectories to sample. Default: 1000',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--do-report',
        dest='do_report',
        action='store_true',
        help='Write / Print report. Default: False',
    )
    parser.add_argument(
        '--load',
        dest='load',
        action='store_true',
        help='Load already computed hjb results. Default: False',
    )
    parser.add_argument(
        '--save-trajectory',
        dest='save_trajectory',
        action='store_true',
        help='Save the first trajectory sampled. Default: False',
    )
    return parser
