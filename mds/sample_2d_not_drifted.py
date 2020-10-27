from mds.langevin_2d_importance_sampling import Sampling
from mds.potentials_and_gradients_2d import POTENTIAL_NAMES

import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='sample not drifted 2D overdamped Langevin SDE')
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
        help='Set the value of the process at time t=0. Default: [-1, -1]',
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
        '--h',
        dest='h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser


def main():
    args = get_parser().parse_args()

    # initialize langevin_1d object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain).reshape((2, 2)),
        target_set=np.array(args.target_set).reshape((2, 2)),
        h=args.h,
        is_drifted=False,
    )

    # set path
    sample.set_not_drifted_dir_path()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # plot potential and gradient
    #if args.do_plots:
    #    sample.plot_tilted_potential(file_name='tilted_potential')
    #    sample.plot_tilted_drift(file_name='tilted_drift')

    # sample
    sample.sample_not_drifted()

    # compute and print statistics
    sample.compute_statistics()
    sample.write_report()


if __name__ == "__main__":
    main()
