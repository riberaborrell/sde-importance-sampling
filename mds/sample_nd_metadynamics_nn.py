from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_metadynamics_nn import MetadynamicsNN
from mds.langevin_nd_sde import LangevinSDE
from mds.langevin_nd_function_approximation import FunctionApproximation
from mds.neural_networks import FeedForwardNN, DenseNN

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for the nd overdamped Langevin SDE'
    parser.add_argument(
        '--meta-type',
        dest='meta_type',
        choices=['cum', 'ind'],
        default='cum',
        help='Type of metadynamics algorithm. Default: cum',
    )
    parser.add_argument(
        '--do-updates-plots',
        dest='do_updates_plots',
        action='store_true',
        help='Do plots after adding a gaussian. Default: False',
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
        h=args.h,
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
        func.train_parameters_with_not_controlled_potential(sde)

    # add nn function approximation
    sample.nn_func_appr = func

    # initialize meta nd object
    meta = MetadynamicsNN(
        sample=sample,
        k=args.k_meta,
        N=args.N_meta,
        sigma_i=args.sigma_i_meta,
        seed=args.seed,
        meta_type=args.meta_type,
        do_updates_plots=args.do_updates_plots,
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

        # metadynamics algorythm for different samples
        for i in np.arange(meta.N):
            if meta.meta_type == 'cum':
                meta.cumulative_metadynamics_algorithm(i)
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
