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
    sgd_sample = Sampling(
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
        dir_path = sgd_sample.settings_dir_path

    if args.theta == 'not-controlled':

        # set training algorithm
        func.training_algorithm = args.train_alg

        dir_path = sgd_sample.settings_dir_path

    elif args.theta == 'meta':

        # set training algorithm
        func.training_algorithm = args.train_alg

        # get metadynamics
        sde = LangevinSDE.new_from(sgd_sample)
        meta = sde.get_metadynamics_sampling(args.meta_type, args.weights_type,
                                             args.omega_0_meta, args.sigma_i, args.dt_meta,
                                             args.k_meta, args.N_meta, args.seed)
        dir_path = meta.dir_path

    # set dir path for nn
    func.set_dir_path(dir_path)

    # add nn function approximation
    sgd_sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sgd_sample.seed = args.seed
    sgd_sample.xzero = np.full(args.n, args.xzero_i)
    sgd_sample.N = args.N_sgd
    sgd_sample.dt = args.dt

    # initialize SOM object
    sgd = StochasticOptimizationMethod(
        sample=sgd_sample,
        loss_type=args.loss_type,
        optimizer=args.optimizer,
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
        n_iterations_backup=args.n_iterations_backup,
    )

    # load stochastic optimization method
    if not sgd.load():
        return

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

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set controlled sampling dir path
    sample.set_controlled_dir_path(sgd.dir_path)

    if not args.load:

        # load model parameters from last iteration
        sgd_sample.nn_func_appr.model.load_parameters(sgd.thetas[-1])
        sample.nn_func_appr = sgd_sample.nn_func_appr

        # sample and compute statistics
        sample.sample_controlled()

        # save statistics
        sample.save()

    # load already sampled bias potential
    else:
        if not sample.load():
            return

    # print statistics
    if args.do_report:
        sample.write_report()


if __name__ == "__main__":
    main()
