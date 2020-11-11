from mds.base_parser_2d import get_base_parser
from mds.langevin_nd_gradient_descent import GradientDescent
from mds.langevin_2d_importance_sampling import Sampling

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
        domain=np.array(args.domain).reshape(2, 2),
        target_set=np.array(args.target_set).reshape(2, 2),
        h=args.h,
        is_drifted=True,
    )

    # set gaussian ansatz functions
    m_x, m_y = args.m
    sigma_x, sigma_y = args.sigma
    sample.set_gaussian_ansatz_functions(m_x, m_y, sigma_x, sigma_y)

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
            sample.plot_appr_free_energy_contour('appr_free_energy_contour' + epoch_stamp, gd.epochs_dir_path)
            sample.plot_control('control' + epoch_stamp, gd.epochs_dir_path)
            sample.plot_tilted_potential_contour('tilted_potential_contour' + epoch_stamp, gd.epochs_dir_path)

    # plot last epochs
    last_epoch = gd.epochs -1
    sample.theta = gd.thetas[last_epoch]
    epoch_stamp = '_epoch{}'.format(last_epoch)
    sample.plot_appr_free_energy_surface('appr_free_energy_surface' + epoch_stamp, gd.dir_path)
    sample.plot_appr_free_energy_contour('appr_free_energy_contour' + epoch_stamp, gd.dir_path)
    sample.plot_control('control' + epoch_stamp, gd.dir_path)
    sample.plot_tilted_potential_surface('tilted_potential_surface' + epoch_stamp, gd.dir_path)
    sample.plot_tilted_potential_contour('tilted_potential_contour' + epoch_stamp, gd.dir_path)

    # plot losses and time steps for all epochs
    gd.plot_gd_losses()
    gd.plot_gd_time_steps()

if __name__ == "__main__":
    main()
