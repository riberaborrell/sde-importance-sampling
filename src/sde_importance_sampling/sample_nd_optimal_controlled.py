from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.utils_path import get_hjb_solution_dir_path


import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample optimal controlled nd overdamped Langevin SDE'
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
        is_optimal = True,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
    )

    # set path
    hjb_dir_path = get_hjb_solution_dir_path(sample.settings_dir_path, args.h_hjb)
    sample.set_controlled_dir_path(hjb_dir_path)

    # sample trajectories with optimal control
    if not args.load:

        # sample and compute statistics
        sample.sample_optimal_controlled(h=args.h_hjb)

        # save statistics
        sample.save()

    # load already computed statistics
    else:
        if not sample.load():
            return

    # report statistics
    if args.do_report:
        sample.write_report()


if __name__ == "__main__":
    main()
