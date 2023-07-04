import numpy as np

from sde.langevin_sde import LangevinSDE
from sampling.importance_sampling import Sampling
from utils.base_parser import get_base_parser


def get_parser():
    parser = get_base_parser()
    parser.description = 'Load not controlled nd overdamped Langevin SDE from Multiple batches'
    return parser

def main():
    args = get_parser().parse_args()

    # check number of batch trajectories
    assert args.K >= args.K_batch, ''
    assert args.K % args.K_batch == 0, ''

    # number of batch samples
    n_batch_samples = args.K // args.K_batch

    # set alpha array
    if args.potential_name == 'nd_2well':
        alpha = np.full(args.d, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.d)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # set target set array
    if args.potential_name == 'nd_2well':
        target_set = np.full((args.d, 2), [1, 3])
    elif args.potential_name == 'nd_2well_asym':
        target_set = np.empty((args.d, 2))
        target_set[0] = [1, 3]
        target_set[1:] = [-3, 3]

    # initialize sde object
    sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name=args.potential_name,
        d=args.d,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
    )

    # initialize sampling object
    sample = Sampling(sde, is_controlled=False)

    # initialize batch sampling object
    batch_sample = Sampling(sde, is_controlled=False)

    # set sampling and Euler-Marujama parameters
    assert args.seed is None, ''
    sample.set_sampling_parameters(
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.d, args.xzero_i),
        K=args.K,
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

        # set sample parameters
        batch_sample.dt = args.dt
        batch_sample.K = args.K_batch

        batch_sample.seed = i + 1

        # load files
        batch_sample.set_not_controlled_dir_path()
        batch_sample.load()

        # add fht
        idx_i_batch = slice(batch_sample.K * i, batch_sample.K * (i + 1))
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
