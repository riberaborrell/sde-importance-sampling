import sampling

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='drifted 1D overdamped Langevin')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
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
        '--N',
        dest='N',
        type=int,
        default=10**5,
        help='Set number of time steps. Default: 100.000',
    )
    parser.add_argument(
        '--m',
        dest='m',
        type=int,
        default=10,
        help='Set the number of uniformly distributed ansatz functions \
              that you want to use. Default: 10',
    )
    parser.add_argument(
        '--sigma',
        dest='sigma',
        type=float,
        default=0.2,
        help='Set the standard deviation of the ansatz functions. Default: 0.2',
    )
    parser.add_argument(
        '--a-type',
        dest='a_type',
        choices=['optimal', 'meta'],
        default='optimal',
        help='Type of drift. Default: optimal',
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
    sample = sampling.langevin_1d(beta=args.beta)

    # set ansatz functions and a optimal
    sample.set_unif_dist_ansatz_functions(m=args.m, sigma=args.sigma)
    sample.set_a_optimal()
    
    # set a 
    if args.a_type == 'optimal':
        sample.is_drifted = True
        sample.a = sample.a_opt
    else:
        sample.set_a_from_metadynamics()

    # plot potential and gradient
    if args.do_plots:
        sample.plot_potential_and_gradient(file_name='potential_and_gradient_drifted')

    # set sampling and Euler-Majurama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M, 
        target_set=args.target_set,
        dt=args.dt,
        N=args.N,
    )

    # sample
    sample.sample_drifted()

    # compute and print statistics
    sample.compute_statistics()
    sample.save_statistics()
    

if __name__ == "__main__":
    main()
