from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling
from mds.utils import get_hjb_solution_dir_path

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample optimal drifted nd overdamped Langevin SDE'
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

    # set path
    hjb_dir_path = get_hjb_solution_dir_path(sample.example_dir_path, sample.h)
    dir_path = os.path.join(
        hjb_dir_path,
        'optimal-importance-sampling',
    )
    sample.set_dir_path(dir_path)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set is optimal flag
    sample.is_optimal = True

    # sample and compute statistics
    sample.sample_optimal_drifted()

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()
