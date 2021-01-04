from mds.base_parser_2d import get_base_parser
from mds.langevin_2d_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'sample optimal drifted 2D overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize langevin_1d object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain).reshape(2, 2),
        target_set=np.array(args.target_set).reshape(2, 2),
        h=args.h,
        is_drifted=True,
    )

    # set path
    dir_path = os.path.join(
        sample.example_dir_path,
        'reference_solution',
        'optimal-importance-sampling',
    )
    sample.set_dir_path(dir_path)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.array(args.xzero),
        N=args.N,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # set is optimal flag
    sample.is_optimal = True

    # sample and compute statistics
    sample.sample_optimal_drifted()

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()

