import numpy as np

from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.importance_sampling import Sampling

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE split in ' \
                         'multiple batches'
    return parser

def main():
    args = get_parser().parse_args()

    # check number of batch trajectories
    assert args.K > args.K_batch, ''
    assert args.K % args.K_batch == 0, ''
    n_batch_samples = int(args.K / args.K_batch)

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

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.d, args.xzero_i),
        K=args.K_batch,
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
