import sampling

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='not drifted 1D overdamped Langevin')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=['sym_1well', 'sym_2well', 'asym_2well'],
        default='sym_2well',
        help='Set the potential for the 1D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=1,
        help='Set the parameter alpha for the chosen potential. Default: 1',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 1D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1.,
        help='Set the value of the process at time t=0. Default: -1',
    )
    parser.add_argument(
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
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
        default=10**6,
        help='Set maximal number of time steps. Default: 1.100.000',
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
    sample = sampling.langevin_1d(
        potential_name=args.potential_name,
        alpha=args.alpha,
        beta=args.beta,
        target_set=args.target_set,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )
    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential(file_name='sampling_drifted_potential')
        sample.plot_tilted_drift(file_name='sampling_drifted_drift')

    # sample
    sample.sample_not_drifted()

    # compute and print statistics
    sample.compute_statistics()
    sample.save_statistics()


if __name__ == "__main__":
    main()
