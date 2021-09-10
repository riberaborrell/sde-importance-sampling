from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_sde import LangevinSDE
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_soc_optimization_method import StochasticOptimizationMethod
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Performs SGD method using the inexact ipa gradient estimator.' \
                         'The space of possible controls is given by the linear combination' \
                         'of vector fields from Gaussian ansatz functions.'
    parser.add_argument(
        '--do-iteration-plots',
        dest='do_iteration_plots',
        action='store_true',
        help='Do plots for each iteration. Default: False',
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

    # initialize sde object
    sde = LangevinSDE.new_from(sample)

    # initialize Gaussian ansatz
    sample.ansatz = GaussianAnsatz(n=args.n)

    # distribute Gaussian ansatz
    if args.distributed == 'uniform':
        sample.ansatz.set_unif_dist_ansatz_functions(sde, args.m_i, args.sigma_i)
    elif args.distributed == 'meta':
        sample.ansatz.set_meta_dist_ansatz_functions(sde, args.dt_meta, args.sigma_i_meta,
                                                     args.k, args.N_meta)

    # set initial coefficients
    if args.theta == 'null':
        sample.ansatz.set_theta_null()
    elif args.theta == 'meta':
        sde.h = args.h
        sample.ansatz.set_theta_from_metadynamics(sde, args.dt_meta, args.sigma_i_meta,
                                                  args.k, args.N_meta)
    elif args.theta == 'optimal':
        #sample.set_theta_optimal()
        return

    # set dir path for gaussian ansatz
    sample.ansatz.set_dir_path(sde)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=100000,
    )

    # set l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize gradient descent object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        loss_type='ipa',
        optimizer='sgd',
        lr=args.lr,
        iterations_lim=args.iterations_lim,
        do_iteration_plots=args.do_iteration_plots,
    )

    # start gd with ipa estimator for the gradient
    if not args.load:
        try:
            sgd.sgd_ipa_gaussian_ansatz()

        # save if job is manually interrupted
        except KeyboardInterrupt:
            sgd.cut_arrays()
            sgd.stop_timer()
            sgd.save()

    # load already run gd
    else:
        if not sgd.load_som():
            return

    # report gd
    if args.do_report:
        sgd.write_report()

    # do plots 
    if args.do_plots:

        # plot loss function, relative error and time steps
        #sgd.plot_losses()
        #sgd.plot_I_u()
        #sgd.plot_time_steps()

        if args.n == 1:
            sgd.plot_1d_iteration()
            sgd.plot_1d_iterations()

        elif args.n == 2:
            sgd.plot_2d_iteration()

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
            sgd.dir_path,
            'is',
            'N_{:.0e}'.format(sample.N),
        )
        sample.set_dir_path(dir_path)

        # sample and compute statistics
        sample.ansatz.theta = sgd.thetas[-1]
        sample.sample_controlled()

        # print statistics
        sample.write_report()


if __name__ == "__main__":
    main()
