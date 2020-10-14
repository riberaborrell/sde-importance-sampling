import gradient_descent
from plotting import Plot
from potentials_and_gradients import POTENTIAL_NAMES
import sampling

import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='gd ipa')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=None,
        help='Set seed. Default: None',
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
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1,
        help='Set the initial position. Default: -1',
    )
    parser.add_argument(
        '--M',
        dest='M',
        type=int,
        default=500,
        help='Set number of trajectories to sample. Default: 500',
    )
    parser.add_argument(
        '--epochs-lim',
        dest='epochs_lim',
        type=int,
        default=100,
        help='Set maximal number of epochs. Default: 100',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.1,
        help='Set learning rate. Default: 0.1',
    )
    parser.add_argument(
        '--atol',
        dest='atol',
        type=float,
        default=0.01,
        help='Set absolute tolerance between value funtion and loss at xinit. Default: 0.01',
    )
    parser.add_argument(
        '--m',
        dest='m',
        type=int,
        default=50,
        help='Set the number of uniformly distributed ansatz functions \
              that you want to use. Default: 50',
    )
    parser.add_argument(
        '--sigma',
        dest='sigma',
        type=float,
        default=1,
        help='Set the standard deviation of the gaussian ansatz functions \
              that you want to use. Default: 1',
    )
    parser.add_argument(
        '--theta-init',
        dest='theta_init',
        choices=['null', 'meta', 'optimal'],
        default='null',
        help='Type of initial control. Default: null',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--do-epoch-plots',
        dest='do_epoch_plots',
        action='store_true',
        help='Do plots for each epoch. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize langevin_1d object
    sample = sampling.langevin_1d(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set),
        is_drifted=True,
    )

    # set gaussian ansatz functions
    sample.set_gaussian_ansatz_functions(args.m, args.sigma)

    # set chosen coefficients
    if args.theta_init == 'optimal':
        sample.set_theta_optimal()
    elif args.theta_init == 'meta':
        sample.set_theta_from_metadynamics()
    else:
        sample.set_theta_null()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M,
        dt=0.001,
        N_lim=100000,
    )

    # initialize gradient descent object
    gd = gradient_descent.gradient_descent(
        sample=sample,
        grad_type='ipa-value-f',
        theta_init=args.theta_init,
        lr=args.lr,
        epochs_lim=args.epochs_lim,
        do_epoch_plots=args.do_epoch_plots,
    )

    gd.get_value_f_at_xzero()
    gd.gd_ipa()
    gd.save_gd()
    gd.write_report()

    if args.do_plots:
        gd.plot_gd_controls()
        gd.plot_gd_appr_free_energies()
        gd.plot_gd_tilted_potentials()
        gd.plot_gd_losses()

if __name__ == "__main__":
    main()
