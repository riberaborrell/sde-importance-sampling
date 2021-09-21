from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample controlled nd overdamped Langevin SDE. The bias potential ' \
                         'is parametrized with linear combination of Gaussian functions. ' \
                         'The weights are chosen from the metadynamics sampling.'
    parser.add_argument(
        '--is-cumulative',
        dest='is_cumulative',
        action='store_true',
        help='Cumulative metadynamics algorithm. Default: False',
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

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
    )

    # get meta sampling
    meta = sample.get_metadynamics_sampling(args.dt_meta, args.sigma_i_meta,
                                            args.is_cumulative, args.k_meta, args.N_meta)

    # get the corresponding Gaussian ansatz
    meta.sample.ansatz = GaussianAnsatz(n=args.n, normalized=False)
    if meta.is_cumulative:
        meta.set_ansatz_cumulative()
    else:
        meta.set_ansatz_averaged()
    sample.ansatz = meta.sample.ansatz

    # set controlled sampling dir path
    sample.set_controlled_dir_path(meta.dir_path)

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
