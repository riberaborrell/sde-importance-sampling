from potentials_and_gradients import POTENTIAL_NAMES
import sampling

import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='sample drifted 1D overdamped Langevin SDE')
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
        default='1d_sym_2well',
        help='Set the potential for the 1D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        nargs='+',
        type=float,
        default=[1],
        help='Set the parameter alpha for the chosen potential. Default: [1]',
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
        '--domain',
        nargs=2,
        dest='domain',
        type=float,
        default=[-3, 3],
        help='Set the interval domain. Default: [-3, 3]',
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
        '--m',
        dest='m',
        type=int,
        default=30,
        help='Set the number of uniformly distributed ansatz functions \
              that you want to use. Default: 30',
    )
    parser.add_argument(
        '--sigma',
        dest='sigma',
        type=float,
        default=1,
        help='Set the standard deviation of the ansatz functions. Default: 1',
    )
    parser.add_argument(
        '--theta',
        dest='theta',
        choices=['optimal', 'meta', 'gd', 'null'],
        default='optimal',
        help='Type of control. Default: optimal',
    )
    parser.add_argument(
        '--gd-theta-init',
        dest='gd_theta_init',
        choices=['null', 'meta', 'optimal'],
        default='null',
        help='Type of initial control in the gd. Default: null',
    )
    parser.add_argument(
        '--gd-lr',
        dest='gd_lr',
        type=float,
        default=1,
        help='Set learning rate used in the gd. Default: 1',
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
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain),
        target_set=np.array(args.target_set),
        is_drifted=True,
    )

    # set gaussian ansatz functions
    sample.set_gaussian_ansatz_functions(args.m, args.sigma)

    # set load optimal coefficients and chosen coefficients
    sample.compute_theta_optimal()
    if args.theta == 'optimal':
        sample.set_theta_optimal()
    elif args.theta == 'meta':
        sample.set_theta_from_metadynamics()
    elif args.theta == 'gd':
        sample.set_theta_from_gd(
            gd_type='ipa-value-f',
            gd_theta_init=args.gd_theta_init,
            gd_lr=args.gd_lr,
        )
    else:
        sample.set_theta_null()

    # plot potential and gradient
    if args.do_plots:
        theta_stamp = 'theta-{}'.format(args.theta)
        sample.ansatz.plot_gaussian_ansatz_functions()
        sample.plot_appr_mgf(file_name=theta_stamp+'_appr_mgf')
        sample.plot_appr_free_energy(file_name=theta_stamp+'_appr_free_energy')
        sample.plot_control(file_name=theta_stamp+'_control')
        sample.plot_tilted_potential(file_name=theta_stamp+'_tilted_potential')
        sample.plot_tilted_drift(file_name=theta_stamp+'_tilted_drift')

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # sample
    sample.sample_drifted()

    # compute and print statistics
    sample.compute_statistics()
    sample.write_report()


if __name__ == "__main__":
    main()
