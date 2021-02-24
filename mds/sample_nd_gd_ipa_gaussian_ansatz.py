from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_gradient_descent import GradientDescent
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np

import os

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
    parser.add_argument(
        '--do-importance-sampling',
        dest='do_importance_sampling',
        action='store_true',
        help='Sample controlled dynamics',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_controlled=True,
    )

    # initialize Gaussian ansatz
    sample.ansatz = GaussianAnsatz(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
    )

    # distribute Gaussian ansatz
    if args.distributed == 'uniform':
        sample.ansatz.set_unif_dist_ansatz_functions(args.m_i, args.sigma_i)
    elif args.distributed == 'meta':
        sample.ansatz.set_meta_dist_ansatz_functions(args.sigma_i_meta, args.k, args.N_meta)

    # set initial coefficients
    if args.theta_init == 'null':
        sample.ansatz.set_theta_null()
    elif args.theta_init == 'meta':
        sample.ansatz.h = args.h
        sample.ansatz.set_theta_from_metadynamics(args.sigma_i_meta, args.k, args.N_meta)
    elif args.theta_init == 'optimal':
        #sample.set_theta_optimal()
        return

    # set dir path for gaussian ansatz
    sample.ansatz.set_dir_path()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N_gd,
        dt=args.dt,
        k_lim=100000,
    )

    # initialize gradient descent object
    gd = GradientDescent(
        sample=sample,
        grad_type='ipa-value-f',
        lr=args.lr,
        epochs_lim=args.epochs_lim,
        do_epoch_plots=args.do_epoch_plots,
    )

    # start gd with ipa estimator for the gradient
    if not args.load:
        try:
            gd.gd_ipa()

        # save if job is manually interrupted
        except KeyboardInterrupt:
            gd.stop_timer()
            gd.save_gd()

    # load already run gd
    else:
        if not gd.load_gd():
            return

    # report gd
    if args.do_report:
        gd.write_report()

    # do plots 
    if args.do_plots:
        gd.plot_losses(args.h_hjb, args.N)
        gd.plot_time_steps()
        #gd.plot_1d_epoch(epoch=5)
        gd.plot_1d_epochs()
        #gd.plot_2d_epoch(epoch=0)

    if args.do_importance_sampling:
        # set sampling and Euler-Marujama parameters
        sample.set_sampling_parameters(
            seed=args.seed,
            xzero=np.full(args.n, args.xzero_i),
            N=args.N,
            dt=args.dt,
            k_lim=args.k_lim,
        )

        # set controlled sampling dir path
        dir_path = os.path.join(
            gd.dir_path,
            'is',
            'N_{:.0e}'.format(sample.N),
        )
        sample.set_dir_path(dir_path)

        # sample and compute statistics
        sample.ansatz.theta = gd.thetas[-1]
        sample.sample_controlled()

        # print statistics
        sample.write_report()


if __name__ == "__main__":
    main()
