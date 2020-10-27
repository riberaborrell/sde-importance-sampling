from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_gradient_descent import GradientDescent
from mds.langevin_1d_importance_sampling import Sampling
from mds.utils import make_dir_path

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Plot ipa'
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
