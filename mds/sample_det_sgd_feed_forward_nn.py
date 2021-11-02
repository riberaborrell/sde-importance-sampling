from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_sde import LangevinSDE
from mds.langevin_nd_soc_optimization_method import StochasticOptimizationMethod
from mds.langevin_nd_function_approximation import FunctionApproximation
from mds.neural_networks import SequentialNN

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = ''
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
        problem_name='langevin_det-t',
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        is_controlled=True,
        T=args.T,
    )

    # set sampling and Euler-Marujama parameters
    k_lim = int(np.ceil(args.T / args.dt))
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=k_lim,
    )

    # get dimensions of each layer
    if args.d_layers is not None:
        d_layers = [args.n] + args.d_layers + [args.n]
    else:
        d_layers = [args.n, args.n]

    # initialize nn
    model = SequentialNN(k_lim + 1, d_layers, args.activation_type, args.dense)

    # initialize function approximation
    func = FunctionApproximation(
        target_function='control',
        model=model,
        initialization=args.theta,
    )

    # set initial choice of parametrization
    if args.theta == 'random':

        # set dir path for nn
        func.set_dir_path(sample.settings_dir_path)

    elif args.theta == 'null':

        # set parameters to zero
        func.zero_parameters()

        # set dir path for nn
        func.set_dir_path(sample.settings_dir_path)

    elif args.theta == 'meta' and not args.load:

        # get metadynamics
        sde = LangevinSDE.new_from(sample)
        meta = sde.get_metadynamics_sampling(args.meta_type, args.weights_type,
                                             args.omega_0_meta, args.k_meta, args.N_meta)

        # set dir path for nn
        func.set_dir_path(meta.dir_path)

        # train parameters if not trained yet
        func.train_parameters_with_metadynamics(meta)
    else:
        return


    # add nn function approximation
    sample.nn_func_appr = func


    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize SOM object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        loss_type=args.loss_type,
        optimizer='adam',
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
    )

    # start sgd
    if not args.load:
        try:
            sgd.som_nn()

        # save if job is manually interrupted
        except KeyboardInterrupt:
            sgd.cut_arrays()
            sgd.stop_timer()
            sgd.save()

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
        sgd.load_mc_sampling(dt_mc=0.005, N_mc=10**3)
        sgd.load_hjb_solution_and_sampling(h_hjb=0.01, dt_hjb=0.005, N_hjb=10**3)
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
        #if sgd.u_l2_errors is not None:
        #    sgd.plot_u_l2_error()
        #    sgd.plot_u_l2_error_change()



if __name__ == "__main__":
    main()
