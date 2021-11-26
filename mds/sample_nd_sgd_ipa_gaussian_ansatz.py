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
    return parser

def main():
    args = get_parser().parse_args()

    # set alpha array
    if args.potential_name == 'nd_2well':
        alpha = np.full(args.n, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.n)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # initialize sampling object
    sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
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
    elif args.theta == 'random':
        sample.ansatz.set_theta_random()
    elif args.theta == 'meta':
        sde.h = args.h
        sample.ansatz.set_theta_metadynamics(sde, args.dt_meta, args.sigma_i_meta,
                                                  args.k, args.N_meta)
    elif args.theta == 'hjb':
        sample.ansatz.set_theta_hjb(sde, args.h_hjb)
    else:
        return

    # set dir path for gaussian ansatz
    sample.ansatz.set_dir_path(sde)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
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
        n_iterations_lim=args.n_iterations_lim,
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
        if not sgd.load():
            return

    # report gd
    if args.do_report:
        sgd.write_report()

    # do plots 
    if args.do_plots:

        if args.n == 1:
            sgd.plot_1d_iteration()
            sgd.plot_1d_iterations()

        elif args.n == 2:
            sgd.plot_2d_iteration()

        # load mc sampling and hjb solution and prepare labels
        sgd.load_mc_sampling(dt_mc=0.01, N_mc=10**3)
        sgd.load_hjb_solution_and_sampling(h_hjb=0.001, dt_hjb=0.01, N_hjb=10**3)
        sgd.load_plot_labels_colors_and_linestyles()

        # loss
        sgd.plot_loss()

        # mean and relative error of the reweighted quantity of interest
        sgd.plot_mean_I_u()
        sgd.plot_re_I_u()

        # time steps and computational time
        sgd.plot_time_steps()
        sgd.plot_cts()

        # u l2 error and its change
        if sgd.u_l2_errors is not None:
            sgd.plot_u_l2_error()
            sgd.plot_u_l2_error_change()


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
