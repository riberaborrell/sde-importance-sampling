from sde_importance_sampling.base_parser_nd import get_base_parser
from sde_importance_sampling.gaussian_nd_ansatz_functions import GaussianAnsatz
from sde_importance_sampling.langevin_nd_importance_sampling import Sampling
from sde_importance_sampling.langevin_nd_metadynamics import Metadynamics
from sde_importance_sampling.langevin_nd_sde import LangevinSDE
from sde_importance_sampling.langevin_nd_function_approximation import FunctionApproximation
from sde_importance_sampling.neural_networks import FeedForwardNN, DenseNN

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for the nd overdamped Langevin SDE'
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

    # train control to be zero
    if not args.load:
        sde = LangevinSDE.new_from(sample)
        func.initialization = 'null'
        func.set_dir_path(sde.settings_dir_path)
        func.train_parameters_with_not_controlled_potential(sde)

    # add nn function approximation
    sample.nn_func_appr = func

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        k=args.k_meta,
        N=args.N_meta,
        seed=args.seed,
        meta_type='cum-nn',
        weights_type=args.weights_type,
        omega_0=args.omega_0_meta,
    )

    # set sampling parameters
    meta.set_sampling_parameters(
        k_lim=args.k_lim,
        dt=args.dt_meta,
        xzero=np.full(args.n, args.xzero_i),
    )

    # set path
    meta.set_dir_path()

    if not args.load:

        # start timer
        meta.start_timer()

        # sample metadynamics trjectories
        meta.preallocate_metadynamics_coefficients()

        # set the weights of the bias functions for each trajectory
        meta.set_weights()

        # metadynamics algorythm for different samples
        for i in np.arange(meta.N):
            if meta.meta_type == 'cum-nn':
                meta.cumulative_nn_metadynamics_algorithm(i)
            else:
                pass

        # stop timer
        meta.stop_timer()

        # save bias potential
        meta.save()

    # load already sampled bias potential
    else:
        if not meta.load():
            return

    if args.do_report:
        meta.write_report()

if __name__ == "__main__":
    main()
