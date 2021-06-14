from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE. Save fht array'
    parser.add_argument(
        '--batch-id',
        dest='batch_id',
        type=int,
        default=1,
        help='Set batch id. Default: 1',
    )
    parser.add_argument(
        '--N-batch',
        dest='N_batch',
        type=int,
        default=1000,
        help='Set number of trajectories for the batch sampling. Default: 1000',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # check number of batch trajectories
    assert args.N % args.N_batch == 0, ''

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
        is_controlled=False,
        is_batch=True,
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

    # set batch number of trajectories and batch id
    sample.N = args.N_batch
    sample.batch_id = args.batch_id

    # sample not controlled trajectories
    sample.sample_not_controlled()

    # save files
    sample.save_not_controlled_statistics()


if __name__ == "__main__":
    main()
