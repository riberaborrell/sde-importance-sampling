from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_gradient_descent import GradientDescent
from mds.langevin_nd_importance_sampling import Sampling

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
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_drifted=True,
    )

    # distribute Gaussians like meta and start with the coefficients from meta
    assert args.distributed == 'meta', ''
    assert args.theta_init == 'meta', ''
    sample.set_gaussian_ansatz_from_meta()

    # set xzero
    sample.xzero = np.full(args.n, args.xzero_i)

    # get value f at xzero from the reference solution
    sample.get_value_f_at_xzero(h=args.h)

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

    # plot losses and time steps for all epochs
    gd.plot_gd_losses()
    gd.plot_gd_time_steps()

if __name__ == "__main__":
    main()
