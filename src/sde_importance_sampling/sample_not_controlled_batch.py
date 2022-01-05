from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.importance_sampling import Sampling

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

    # set target set array
    if args.potential_name == 'nd_2well':
        target_set = np.full((args.n, 2), [1, 3])
    elif args.potential_name == 'nd_2well_asym':
        target_set = np.empty((args.n, 2))
        target_set[0] = [1, 3]
        target_set[1:] = [-3, 3]

    # initialize sampling object
    sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
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
        sample.seed = i + 1
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