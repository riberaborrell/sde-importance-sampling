from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample drifted nd overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize Sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_drifted=True,
        h=args.h,
    )

    # distribute Gaussian ansatz
    if args.distributed == 'uniform':
        sample.set_gaussian_ansatz_uniformly()
    elif args.distributed == 'meta':
        sample.set_gaussian_ansatz_from_meta()

    # set chosen coefficients
    if args.theta == 'optimal':
        sample.set_theta_optimal()
    elif args.theta == 'meta':
        sample.set_theta_from_metadynamics()
    elif args.theta == 'gd':
        #sample.set_theta_from_gd(
        #    gd_type='ipa-value-f',
        #    gd_theta_init=args.theta_init,
        #    gd_lr=args.lr,
        #)
        return
    else:
        sample.set_theta_null()

    # plot potential and gradient
    if args.do_plots:
        return

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # sample and compute statistics
    sample.sample_drifted()

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()
