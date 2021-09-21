from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_sde import LangevinSDE
from mds.langevin_nd_soc_optimization_method import StochasticOptimizationMethod
from mds.langevin_nd_function_approximation import FunctionApproximation
from mds.neural_networks import FeedForwardNN, DenseNN

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    parser.add_argument(
        '--d-layers',
        nargs='+',
        dest='d_layers',
        type=int,
        help='Set dimensions of the NN inner layers',
    )
    parser.add_argument(
        '--dense',
        dest='dense',
        action='store_true',
        help='Chooses a dense feed forward NN. Default: False',
    )
    parser.add_argument(
        '--activation',
        dest='activation_type',
        choices=['relu', 'tanh'],
        default='relu',
        help='Type of activation function. Default: relu',
    )
    parser.add_argument(
        '--do-iteration-plots',
        dest='do_iteration_plots',
        action='store_true',
        help='Do plots for each iteration. Default: False',
    )
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
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        is_controlled=True,
    )

    # get dimensions of each layer
    if args.d_layers is not None:
        d_layers = [args.n] + args.d_layers + [args.n]
    else:
        d_layers = [args.n, args.n]

    # initialize nn
    if not args.dense:
        model = FeedForwardNN(d_layers, args.activation_type)
    else:
        model = DenseNN(d_layers, args.activation_type)

    # initialize function approximation
    func = FunctionApproximation(
        target_function='control',
        model=model,
    )

    # set initial choice of parametrization
    if args.theta == 'random':
        pass
    elif args.theta == 'null':
        func.zero_parameters()
    elif args.theta == 'meta':
        if not args.load:
            sde = LangevinSDE.new_from(sample)
            func.fit_parameters_from_metadynamics(sde, dt_meta=args.dt_meta,
                                                  sigma_i_meta=args.sigma_i_meta,
                                                  k=args.k_meta, N_meta=args.N_meta)
        else:
            func.initialization = 'meta'
    elif args.theta == 'flat':
        if not args.load:
            sde = LangevinSDE.new_from(sample)
            func.fit_parameters_flat_controlled_potential(sde, N=1000)
        else:
            func.initialization = 'flat'
    elif args.theta == 'semi-flat':
        if not args.load:
            sde = LangevinSDE.new_from(sample)
            func.fit_parameters_semiflat_controlled_potential(sde)
        else:
            func.initialization = 'semi-flat'
    else:
        return

    # set dir path for nn
    func.set_dir_path(sample.settings_dir_path)

    # add nn function approximation
    sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=100000,
    )

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
        do_iteration_plots=args.do_iteration_plots,
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

        # plot loss function, relative error and time steps
        sgd.plot_losses(args.h_hjb)
        sgd.plot_I_u()
        if sgd.u_l2_errors is not None:
            sgd.plot_u_l2_error()
        sgd.plot_time_steps()
        sgd.plot_cts()

        if args.n == 1:
            #sgd.plot_1d_iteration()
            sgd.plot_1d_iterations()

        elif args.n == 2:
            sgd.plot_2d_iteration(i=0)


if __name__ == "__main__":
    main()
