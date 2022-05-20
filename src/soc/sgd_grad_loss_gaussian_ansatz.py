import os

import numpy as np

from function_approximation.gaussian_ansatz import GaussianAnsatz
from sampling.importance_sampling import Sampling
from sde.langevin_sde import LangevinSDE
from soc.soc_optimization_method import StochasticOptimizationMethod
from utils.base_parser import get_base_parser


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
        alpha = np.full(args.d, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.d)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # set target set array
    if args.potential_name == 'nd_2well':
        target_set = np.full((args.d, 2), [1, 3])
    elif args.potential_name == 'nd_2well_asym':
        target_set = np.empty((args.d, 2))
        target_set[0] = [1, 3]
        target_set[1:] = [-3, 3]

    # initialize sde object
    sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name=args.potential_name,
        d=args.d,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
    )

    # initialize sampling object
    sample = Sampling(sde, is_controlled=True)

    # initialize Gaussian ansatz
    sample.ansatz = GaussianAnsatz(sde, normalized=True)

    # distribute Gaussian ansatz
    if args.distributed == 'uniform':
        sample.ansatz.set_unif_dist_ansatz_functions(args.m_i, args.sigma_i)
    elif args.distributed == 'meta':
        #sample.ansatz.set_meta_dist_ansatz_functions(args.dt_meta, args.sigma_i_meta,
        #                                             args.delta, args.K_meta)
        pass

    # set initial coefficients
    if args.theta == 'null':
        sample.ansatz.set_theta_null()
    elif args.theta == 'random':
        sample.ansatz.set_theta_random()
    elif args.theta == 'meta':
        sde.h = args.h
        meta = sde.get_metadynamics_sampling(args.meta_type, args.weights_type,
                                             args.omega_0_meta, args.sigma_i, args.dt_meta,
                                             args.delta_meta, args.K_meta, args.seed)

        sample.ansatz.set_theta_metadynamics(meta, args.h)
    elif args.theta == 'hjb':
        sample.ansatz.set_theta_hjb(sde, args.h_hjb)
    else:
        return

    # set dir path for gaussian ansatz
    sample.ansatz.set_dir_path()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.d, args.xzero_i),
        K=args.K,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize gradient descent object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        grad_estimator='ipa',
        optimizer='sgd',
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
        n_iterations_backup=args.n_iterations_backup,
    )

    # start gd with ipa estimator for the gradient
    if not args.load:
        try:
            sgd.sgd_gaussian_ansatz()

        # save if job is manually interrupted
        except KeyboardInterrupt:
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

        if args.d == 1:
            sgd.plot_1d_iteration()
            sgd.plot_1d_iterations()

        elif args.d == 2:
            sgd.plot_2d_iteration()

        # load mc sampling and hjb solution and prepare labels
        sgd.n_iter_run_avg = 10
        sgd.load_mc_sampling(dt_mc=0.01, K_mc=10**3, seed=args.seed)
        sgd.load_hjb_solution_and_sampling(h_hjb=0.001, dt_hjb=0.01, K_hjb=10**3, seed=args.seed)
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
        if hasattr(sgd, 'u_l2_errors'):
            sgd.plot_u_l2_error()
            sgd.plot_u_l2_error_change()


    if args.do_importance_sampling:

        # set sampling and Euler-Marujama parameters
        sample.set_sampling_parameters(
            seed=args.seed,
            xzero=np.full(args.d, args.xzero_i),
            K=args.K,
            dt=args.dt,
            k_lim=args.k_lim,
        )

        # set controlled sampling dir path
        dir_path = os.path.join(
            sgd.dir_path,
            'is',
            'K_{:.0e}'.format(sample.K),
        )
        sample.set_dir_path(dir_path)

        # sample and compute statistics
        sample.ansatz.theta = sgd.thetas[-1]
        sample.sample_controlled()

        # print statistics
        sample.write_report()


if __name__ == "__main__":
    main()
