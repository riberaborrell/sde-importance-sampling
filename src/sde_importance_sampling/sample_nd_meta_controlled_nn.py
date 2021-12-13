from sde_importance_sampling.base_parser_nd import get_base_parser
from sde_importance_sampling.langevin_nd_function_approximation import FunctionApproximation
from sde_importance_sampling.langevin_nd_importance_sampling import Sampling
from sde_importance_sampling.langevin_nd_metadynamics import Metadynamics
from sde_importance_sampling.neural_networks import FeedForwardNN, DenseNN

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample controlled nd overdamped Langevin SDE. The control ' \
                         'is parametrized with neural network. ' \
                         'The weights are fitted from the metadynamics bias potential.'
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

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
    )

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        k=args.k_meta,
        N=args.N_meta,
        seed=args.seed,
        meta_type=args.meta_type,
        weights_type=args.weights_type,
        omega_0=args.omega_0_meta,
    )

    # set path
    meta.set_dir_path()

    # load arrays
    meta.load()

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

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
        initialization='meta',
    )

    # set dir path for nn
    func.set_dir_path(meta.dir_path)

    # train nn network
    if not args.load:
        func.train_parameters_with_metadynamics(meta)

    # add nn function approximation
    sample.nn_func_appr = func

    # set controlled sampling dir path
    sample.set_controlled_dir_path(func.dir_path)

    if not args.load:

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
