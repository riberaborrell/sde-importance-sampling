from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample controlled nd overdamped Langevin SDE. The bias potential is' \
                         ' is parametrized with linear combination of Gaussian functions'
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

    # initialize Gaussian ansatz
    sample.ansatz = GaussianAnsatz(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
    )

    # distribute Gaussian ansatz
    if args.distributed == 'uniform':
        sample.ansatz.set_unif_dist_ansatz_functions(args.m_i, args.sigma_i)
    elif args.distributed == 'meta':
        sample.ansatz.set_meta_dist_ansatz_functions(args.sigma_i_meta, args.k, args.N_meta)
    else:
        return

    # set chosen coefficients
    if args.theta == 'null':
        sample.ansatz.set_theta_null()
    elif args.theta == 'meta':
        sample.ansatz.h = args.h
        sample.ansatz.set_theta_from_metadynamics(args.sigma_i_meta, args.k, args.N_meta)
    elif args.theta == 'optimal':
        sample.ansatz.set_theta_from_hjb_solution(args.h_hjb)

    # set dir path for gaussian ansatz
    sample.ansatz.set_dir_path()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
    )

    # set controlled sampling dir path
    sample.set_controlled_dir_path(sample.ansatz.dir_path)

    # plot potential and gradient
    if args.do_plots:
        return

    # sample and compute statistics
    sample.sample_controlled()

    # save statistics
    sample.save_controlled_statistics()

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()