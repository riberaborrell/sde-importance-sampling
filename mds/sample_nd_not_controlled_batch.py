from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE split in ' \
                         'multiple batches'
    return parser

def main():
    args = get_parser().parse_args()

    # check number of batch trajectories
    assert args.N > args.N_batch, ''
    assert args.N % args.N_batch == 0, ''
    n_batch_samples = int(args.N / args.N_batch)

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
        N=args.N_batch,
    )

    for i in np.arange(n_batch_samples):

        # set path
        sample.seed = i
        sample.set_not_controlled_dir_path()

        # sample and compute statistics
        sample.sample_not_controlled()
        sample.compute_fht_statistics()
        sample.compute_I_statistics()

        # save files
        sample.save()

        msg = 'mc sampling with seed {:d} done'.format(i)
        print(msg)

if __name__ == "__main__":
    main()
