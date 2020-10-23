from mds.langevin_1d_gradient_descent import GradientDescent
from mds.langevin_1d_importance_sampling import Sampling
from mds.potentials_and_gradients import POTENTIAL_NAMES
from mds.utils import make_dir_path

import argparse
import numpy as np

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Plot IPA')
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
        '--m',
        dest='m',
        type=int,
        default=50,
        help='Set number of ansatz functions. Default: 50',
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
        '--lr',
        dest='lr',
        type=float,
        default=0.1,
        help='Set learning rate. Default: 0.1',
    )
    parser.add_argument(
        '--theta-init',
        dest='theta_init',
        choices=['null', 'meta', 'optimal'],
        default='null',
        help='Type of initial control. Default: null',
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

    # initialize Sampling object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set),
        is_drifted=True,
    )

    # set gaussian ansatz functions
    sample.set_gaussian_ansatz_functions(args.m, args.sigma)

    # initialize gradient descent object
    gd = GradientDescent(
        sample=sample,
        grad_type='ipa-value-f',
        theta_init=args.theta_init,
        lr=args.lr,
        epochs_lim=100,
        do_epoch_plots=args.do_epoch_plots,
    )

    # load already run gd
    gd.load_gd()

    # plot each epoch
    if args.do_epoch_plots:
        for epoch in range(gd.epochs):
            sample.theta = gd.thetas[epoch]
            epoch_stamp = '_epoch{}'.format(epoch)
            sample.plot_appr_free_energy('appr_free_energy' + epoch_stamp, gd.epochs_dir_path)
            sample.plot_control('control' + epoch_stamp, gd.epochs_dir_path)
            sample.plot_tilted_potential('tilted_potential' + epoch_stamp, gd.epochs_dir_path)

    # plot all epochs
    gd.plot_gd_controls()
    gd.plot_gd_appr_free_energies()
    gd.plot_gd_tilted_potentials()
    gd.plot_gd_losses()

if __name__ == "__main__":
    main()
