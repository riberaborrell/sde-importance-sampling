from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_controlled=False,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set path
    sample.set_not_controlled_dir_path()

    # sample not controlled trajectories
    if not args.load:

        # save trajectory flag
        if args.save_trajectory:
            sample.save_trajectory = True

        # sample and compute statistics
        sample.sample_not_controlled()

        # save statistics
        sample.save_not_controlled_statistics()

    # load already sampled statistics
    else:
        if not sample.load_not_controlled_statistics():
            return

    # report statistics
    if args.do_report:
        sample.write_report()

    # plot trajectory
    if args.do_plots and args.save_trajectory:
        sample.plot_trajectory()


if __name__ == "__main__":
    main()
