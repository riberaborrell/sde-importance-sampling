from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.importance_sampling import Sampling
from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.soc_optimization_method import StochasticOptimizationMethod
from sde_importance_sampling.function_approximation import FunctionApproximation
from sde_importance_sampling.neural_networks import FeedForwardNN, DenseNN

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

    # set target set array
    if args.potential_name == 'nd_2well':
        target_set = np.full((args.n, 2), [1, 3])
    elif args.potential_name == 'nd_2well_asym':
        target_set = np.empty((args.n, 2))
        target_set[0] = [1, 3]
        target_set[1:] = [-3, 3]

    # initialize sampling object
    sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
        is_controlled=True,
    )

    # get dimensions of each layer
    if args.d_layers is not None:
        d_layers = [args.n] + args.d_layers + [args.n]
    else:
        d_layers = [args.n, args.n]

    # initialize feed-forward or dense nn
    if not args.dense:
        model = FeedForwardNN(d_layers, args.activation_type)
    else:
        model = DenseNN(d_layers, args.activation_type)

    # initialize function approximation
    func = FunctionApproximation(
        target_function='control',
        model=model,
        initialization=args.theta,
    )

    # get dir path for nn
    if args.theta in ['random', 'null']:
        dir_path = sample.settings_dir_path

    if args.theta == 'not-controlled':

        # set training algorithm
        func.training_algorithm = args.train_alg

        dir_path = sample.settings_dir_path

    elif args.theta == 'meta':

        # set training algorithm
        func.training_algorithm = args.train_alg

        # get metadynamics
        sde = LangevinSDE.new_from(sample)
        meta = sde.get_metadynamics_sampling(args.meta_type, args.weights_type,
                                             args.omega_0_meta, args.sigma_i, args.dt_meta,
                                             args.k_meta, args.N_meta, args.seed)
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
        sde = LangevinSDE.new_from(sample)
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
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize SOM object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        loss_type=args.loss_type,
        optimizer=args.optimizer,
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
        n_iterations_backup=args.n_iterations_backup,
    )

    # start sgd
    if not args.load:
        try:
            sgd.som_nn_variance_gradient(args.N_grad)

        # save if job is manually interrupted
        except KeyboardInterrupt:
            sgd.stop_timer()
            sgd.save()

    # load already run gd
    else:
        if not sgd.load():
            return

    # do plots 
    if args.do_plots:
        pass



if __name__ == "__main__":
    main()