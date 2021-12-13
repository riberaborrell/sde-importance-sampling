from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE'
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
        is_controlled=False,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        seed=args.seed,
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
        sample.compute_fht_statistics()
        sample.compute_I_statistics()

        # save files
        sample.save()

    # load already computed statistics
    else:
        if not sample.load():
            return

    # report statistics
    if args.do_report:
        sample.write_report()

    # plot trajectory
    if args.do_plots and args.save_trajectory:
        sample.plot_trajectory()


if __name__ == "__main__":
    main()
