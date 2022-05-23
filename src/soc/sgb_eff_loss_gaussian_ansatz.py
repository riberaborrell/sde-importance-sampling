import numpy as np
import torch

from function_approximation.function_approximation import FunctionApproximation
from function_approximation.models import GaussianAnsatzModel
from sde.langevin_sde import LangevinSDE
from sampling.importance_sampling import Sampling
from soc.soc_optimization_method import StochasticOptimizationMethod
from utils.base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = ''
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

    # initialize gaussian ansatz nn 
    model = GaussianAnsatzModel(sde, args.m_i, args.sigma_i)

    # initialize function approximation
    func = FunctionApproximation(
        target_function='control',
        model=model,
        initialization=args.theta,
    )

    # get dir path for nn
    if args.theta in ['random', 'null']:
        dir_path = sde.settings_dir_path

    if args.theta == 'not-controlled':

        # set training algorithm
        func.training_algorithm = args.train_alg

        dir_path = sde.settings_dir_path

    elif args.theta == 'meta':

        # set training algorithm
        func.training_algorithm = args.train_alg

        # get metadynamics
        meta = sde.get_metadynamics_sampling(args.meta_type, args.weights_type,
                                             args.omega_0_meta, args.sigma_i, args.dt_meta,
                                             args.delta_meta, args.K_meta, args.seed)
        dir_path = meta.dir_path

    # set dir path for nn
    func.set_dir_path(dir_path)

    # set initial parameters
    if args.theta == 'random':

        # the nn parameters are randomly initialized 
        pass

    elif args.theta == 'null':

        # set nn parameters to be zero
        func.zero_parameters()

    elif args.theta == 'not-controlled':

        # train nn parameters such that control is zero
        func.train_parameters(sde=sde)

    elif args.theta == 'meta':

        # train parameters if not trained yet
        func.train_parameters(meta=meta)
    else:
        return

    # add nn function approximation
    sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.d, args.xzero_i),
        K=args.K,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize SOM object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        grad_estimator=args.grad_estimator,
        optimizer=args.optimizer,
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
        n_iterations_backup=args.n_iterations_backup,
    )

    # start sgd
    if not args.load:
        try:
            sgd.som_nn()

        # save if job is manually interrupted
        except KeyboardInterrupt:
            sgd.stop_timer()
            sgd.save_som()

    # load already run gd
    else:
        if not sgd.load():
            return

    # report adam
    if args.do_report:
        sgd.write_report()

    # do plots 
    if args.do_plots:

        # load mc sampling and hjb solution and prepare labels
        som.compute_arrays_running_averages(n_iter_run_window=1)
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

        if args.n == 1:
            #sgd.plot_1d_iteration()
            sgd.plot_1d_iterations()

        elif args.n == 2:
            sgd.plot_2d_iteration()


if __name__ == "__main__":
    main()
