from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.utils import get_hjb_solution_dir_path


import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample optimal controlled nd overdamped Langevin SDE'
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
        is_optimal = True,
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
    hjb_dir_path = get_hjb_solution_dir_path(sample.settings_dir_path, args.h)
    dir_path = os.path.join(
        hjb_dir_path,
        'optimal-is',
        'N_{:.0e}'.format(sample.N),
    )
    sample.set_dir_path(dir_path)

    # sample and compute statistics
    sample.sample_optimal_controlled(h=args.h)

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()
