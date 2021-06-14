from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os
import re

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE. Multiple batches'
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

    # number of batch samples
    n_batch_samples = args.N // args.N_batch

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
        is_batch=False,
    )

    # set sampling and Euler-Marujama parameters
    assert args.seed is None, ''
    sample.set_sampling_parameters(
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        seed=args.seed,
    )

    # set path
    sample.set_not_controlled_dir_path()

    # check batch samples
    batch_files = [
        name for name in os.listdir(sample.dir_path)
        if re.match(r'mc-sampling_batch-\d+.npz', name) is not None
    ]
    assert len(batch_files) == n_batch_samples

    # preallocate first hitting times array and been in target set array flag
    sample.preallocate_fht()

    # initialize total number of time steps and delta time
    sample.k = 0
    sample.ct_delta = 0

    for i in np.arange(n_batch_samples):

        # initialize batch sampling object
        batch_sample = Sampling(
            potential_name=args.potential_name,
            n=args.n,
            alpha=alpha,
            beta=args.beta,
            is_controlled=False,
            is_batch=True,
        )

        # set same dir path
        batch_sample.dir_path = sample.dir_path
        batch_sample.batch_id = i

        # load files
        batch_sample.load_not_controlled_statistics()

        # add fht
        #idx_i_batch = slice(batch_sample.N * i, batch_sample.N * (i + 1))
        idx_i_batch = slice(args.N_batch * i, args.N_batch * (i + 1))
        sample.been_in_target_set[idx_i_batch] = batch_sample.been_in_target_set
        sample.fht[idx_i_batch] = batch_sample.fht

        # add time steps and computational time
        sample.k += batch_sample.k
        sample.ct_delta += batch_sample.ct_delta

    # compute statistics
    sample.compute_fht_statistics()
    sample.compute_I_statistics()

    # save files
    sample.save_not_controlled_statistics()

    # report statistics
    if args.do_report:
        sample.write_report()


if __name__ == "__main__":
        main()
