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

    # distribute Gaussian ansatz
    if args.distributed == 'uniform':
        sample.set_gaussian_ansatz_uniformly()
    elif args.distributed == 'meta':
        sample.set_gaussian_ansatz_from_meta()
    else:
        return

    # set initial coefficients
    if args.theta_init == 'null':
        sample.set_theta_null()
    elif args.theta_init == 'meta':
        sample.h = args.h
        sample.set_theta_from_metadynamics()
    elif args.theta_init == 'optimal':
        #sample.set_theta_optimal()
        return

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=100000,
    )

    # initialize gradient descent object
    gd = GradientDescent(
        sample=sample,
        grad_type='ipa-value-f',
        theta_init=args.theta_init,
        lr=args.lr,
        epochs_lim=args.epochs_lim,
        do_epoch_plots=args.do_epoch_plots,
    )

    # start gd with ipa estimator for the gradient
    try:
        gd.gd_ipa()
    # save statistics if job is manually interrupted
    except KeyboardInterrupt:
        gd.stop_timer()
        gd.save_gd()

    gd.write_report()

    if args.do_plots:
        gd.plot_gd_losses()
        gd.plot_gd_time_steps()

if __name__ == "__main__":
    main()
