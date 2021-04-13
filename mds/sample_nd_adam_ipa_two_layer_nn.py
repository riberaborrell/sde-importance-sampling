from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_sde import LangevinSDE
from mds.langevin_nd_soc_optimization_method import StochasticOptimizationMethod
from mds.langevin_nd_function_approximation import FunctionApproximation
from mds.neural_networks import TwoLayerNet

import numpy as np

import torch
import torch.optim as optim

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    parser.add_argument(
        '--hidden-layer-dim',
        dest='hidden_layer_dim',
        type=int,
        default=10,
        help='Set dimension of the hidden layer. Default: 10',
    )
    parser.add_argument(
        '--iterations-lim',
        dest='iterations_lim',
        type=int,
        default=100,
        help='Set maximal number of adam iterations. Default: 100',
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

    # initialize sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_controlled=True,
    )

    # initialize two layer nn 
    d_in, d_1, d_out = args.n, args.hidden_layer_dim, args.n
    model = TwoLayerNet(d_in, d_1, d_out)

    # initialize function approximation
    func = FunctionApproximation(
        target_function='control',
        model=model,
    )

    # set initial choice of parametrization
    if args.theta == 'random':
        pass
        func.reset_parameters()
    elif args.theta == 'null':
        func.zero_parameters()
    elif args.theta == 'meta' and not args.load:
        sde = LangevinSDE.new_from(sample)
        func.fit_parameters_from_metadynamics(sde)
    elif args.theta == 'meta' and args.load:
        func.initialization='meta'
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
        N=args.N_gd,
        dt=args.dt,
        k_lim=100000,
    )

    # initialize SOM
    adam = StochasticOptimizationMethod(
        sample=sample,
        grad_estimator='ipa',
        optimizer='adam',
        lr=args.lr,
        iterations_lim=args.iterations_lim,
        do_iteration_plots=args.do_iteration_plots,
    )

    # start gd with ipa estimator for the gradient
    if not args.load:
        try:
            adam.som_ipa_nn()

        # save if job is manually interrupted
        except KeyboardInterrupt:
            adam.stop_timer()
            adam.save_som()

    # load already run gd
    else:
        if not adam.load_som():
            return

    # report adam
    if args.do_report:
        #adam.write_report()
        pass

    # do plots 
    if args.do_plots:

        # plot loss function, relative error and time steps
        adam.plot_losses(args.h_hjb, dt_mc=0.001, N_mc=100000)
        adam.plot_time_steps()

        if args.n == 1:
            adam.plot_1d_iteration(i=4)
            adam.plot_1d_iterations()

        elif args.n == 2:
            adam.plot_2d_iteration(i=0)


if __name__ == "__main__":
    main()
