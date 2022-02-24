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

    # check number of batch trajectories
    assert args.N % args.N_batch == 0, ''

    # number of batch samples
    n_batch_samples = args.N // args.N_batch

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
                                             args.k_meta, args.N_meta, args.seed_sgd)
        dir_path = meta.dir_path

    # set dir path for nn
    func.set_dir_path(dir_path)

    # add nn function approximation
    sgd_sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sgd_sample.seed = args.seed_sgd
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
    batch_sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
        is_controlled=True,
    )

    # set sampling and Euler-Marujama parameters
    assert args.seed is None, ''
    sample.set_sampling_parameters(
        xzero=np.full(args.n, args.xzero_i),
        dt=args.dt,
        k_lim=args.k_lim,
        N=args.N,
        seed=args.seed,
    )

    # set number of batch samples used
    sample.n_batch_samples = n_batch_samples

    # set controlled sampling dir path
    sample.set_controlled_dir_path(sgd.dir_path)

    # preallocate first hitting times array and been in target set array flag
    sample.preallocate_fht()
    sample.preallocate_integrals()

    # initialize total number of time steps and delta time
    sample.k = 0
    sample.ct = 0.

    for i in np.arange(n_batch_samples):

        # set same dir path
        batch_sample.dt = args.dt
        batch_sample.N = args.N_batch
        batch_sample.seed = i + 1

        # load files
        batch_sample.set_controlled_dir_path(sgd.dir_path)
        batch_sample.load()

        # add fht
        idx_i_batch = slice(batch_sample.N * i, batch_sample.N * (i + 1))
        sample.been_in_target_set[(idx_i_batch, 0)] = batch_sample.been_in_target_set[:, 0]
        sample.fht[idx_i_batch] = batch_sample.fht

        # add stochastic and deterministic integrals
        sample.stoch_int_fht[idx_i_batch] = batch_sample.stoch_int_fht
        sample.det_int_fht[idx_i_batch] = batch_sample.det_int_fht

        # add time steps and computational time
        sample.k += batch_sample.k
        sample.ct += batch_sample.ct

    # compute statistics
    sample.compute_fht_statistics()
    sample.compute_I_u_statistics()

    # save files
    sample.save()

    # report statistics
    if args.do_report:
        sample.write_report()


if __name__ == "__main__":
    main()
