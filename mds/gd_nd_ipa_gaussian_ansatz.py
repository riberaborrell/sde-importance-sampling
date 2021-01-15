from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_gradient_descent import GradientDescent
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Performs GD method using ipa estimator for the gradient of the loss' \
                         'function. The Gaussian ansatz functions are distributed according to' \
                         'the metadynamics algorithm.'
    parser.add_argument(
        '--epochs-lim',
        dest='epochs_lim',
        type=int,
        default=100,
        help='Set maximal number of epochs. Default: 100',
    )
    parser.add_argument(
        '--atol',
        dest='atol',
        type=float,
        default=0.01,
        help='Set absolute tolerance between value funtion and loss at xinit. Default: 0.01',
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
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_drifted=True,
    )

    # distribute Gaussian like meta and start with the coefficients from meta
    assert args.distributed == 'meta', ''
    assert args.theta_init == 'meta', ''
    sample.set_gaussian_ansatz_from_meta()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=100000,
    )

    # get value f at xzero from the reference solution
    sample.get_value_f_at_xzero(h=args.h)

    # initialize gradient descent object
    gd = GradientDescent(
        sample=sample,
        grad_type='ipa-value-f',
        theta_init=args.theta_init,
        lr=args.lr,
        epochs_lim=args.epochs_lim,
        do_epoch_plots=args.do_epoch_plots,
    )

    breakpoint()
    gd.gd_ipa()
    gd.save_gd()
    gd.write_report()

    if args.do_plots:
        gd.plot_gd_losses()
        gd.plot_gd_time_steps()

if __name__ == "__main__":
    main()
