import numpy as np

from function_approximation.function_approximation import FunctionApproximation
from function_approximation.models import FeedForwardNN, DenseNN
from sampling.importance_sampling import Sampling
from sde.langevin_sde import LangevinSDE
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
    sgd_sample = Sampling(sde, is_controlled=True)

    # get dimensions of each layer
    if args.d_layers is not None:
        d_layers = [args.d] + args.d_layers + [args.d]
    else:
        d_layers = [args.d, args.d]

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
        dir_path = sde.settings_dir_path

    if args.theta == 'not-controlled':

        # set training algorithm
        func.training_algorithm = args.train_alg

        dir_path = sde.settings_dir_path

    elif args.theta == 'meta':

        # set training algorithm
        func.training_algorithm = args.train_alg

        # get metadynamics
        meta = sde.get_metadynamics_sampling(
            meta_type=args.meta_type,
            weights_type=args.weights_type,
            omega_0=args.omega_0_meta,
            sigma_i=args.sigma_i,
            dt=args.dt_meta,
            delta=args.delta_meta,
            K=args.K_meta,
            seed=args.seed_sgd,
        )
        dir_path = meta.dir_path

    # set dir path for nn
    func.set_dir_path(dir_path)

    # add nn function approximation
    sgd_sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sgd_sample.seed = args.seed_sgd
    sgd_sample.xzero = np.full(args.d, args.xzero_i)
    sgd_sample.K = args.K_sgd
    sgd_sample.dt = args.dt

    # initialize SOM object
    som = StochasticOptimizationMethod(
        sample=sgd_sample,
        grad_estimator=args.grad_estimator,
        optimizer=args.optimizer,
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
        n_iterations_backup=args.n_iterations_backup,
    )

    # load stochastic optimization method
    if not som.load():
        return

    # initialize sampling object
    sample = Sampling(sde, is_controlled=True)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.d, args.xzero_i),
        K=args.K,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set controlled sampling dir path
    sample.set_controlled_dir_path(som.dir_path)

    if not args.load:

        # load model parameters from last iteration
        sgd_sample.nn_func_appr.model.load_parameters(som.thetas[-1])
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
