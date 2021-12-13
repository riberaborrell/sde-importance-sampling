from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.importance_sampling import Sampling

import numpy as np
import os
import re

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE. Multiple batches'
    return parser

def main():
    args = get_parser().parse_args()

    # check number of batch trajectories
    assert args.N % args.N_batch == 0, ''

    # number of batch samples
    n_batch_samples = args.N // args.N_batch

    # set alpha array
    if args.potential_name == 'nd_2well':
        alpha = np.full(args.n, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.n)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # initialize sampling and batch sampling object
    sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        is_controlled=False,
    )
    batch_sample = Sampling.new_from(sample)

    # set sampling and Euler-Marujama parameters
    assert args.seed is None, ''
    sample.set_sampling_parameters(
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        seed=args.seed,
    )

    # set number of batch samples used
    sample.n_batch_samples = n_batch_samples

    # set path
    sample.set_not_controlled_dir_path()

    # preallocate first hitting times array and been in target set array flag
    sample.preallocate_fht()

    # initialize total number of time steps and delta time
    sample.k = 0
    sample.ct = 0.

    for i in np.arange(n_batch_samples):

        # set same dir path
        batch_sample.dt = args.dt
        batch_sample.N = args.N_batch
        batch_sample.seed = i + 1

        # load files
        batch_sample.set_not_controlled_dir_path()
        batch_sample.load()

        # add fht
        idx_i_batch = slice(batch_sample.N * i, batch_sample.N * (i + 1))
        sample.been_in_target_set[(idx_i_batch, 0)] = batch_sample.been_in_target_set[:, 0]
        sample.fht[idx_i_batch] = batch_sample.fht

        # add time steps and computational time
        sample.k += batch_sample.k
        sample.ct += batch_sample.ct

    # compute statistics
    sample.compute_fht_statistics()
    sample.compute_I_statistics()

    # save files
    sample.save()

    # report statistics
    if args.do_report:
        sample.write_report()


if __name__ == "__main__":
        main()
