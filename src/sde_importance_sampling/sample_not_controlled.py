import numpy as np

from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.importance_sampling import Sampling
from sde_importance_sampling.langevin_sde import LangevinSDE

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

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
        K=args.K,
        seed=args.seed,
    )

    # set path
    sample.set_not_controlled_dir_path()

    # sample not controlled trajectories
    if not args.load:

        # save trajectory flag
        if args.save_trajectory:
            sample.save_trajectory = True

        # sample and compute statistics
        sample.sample_not_controlled()
        sample.compute_fht_statistics()
        sample.compute_I_statistics()

        # save files
        sample.save()

    # load already computed statistics
    else:
        if not sample.load():
            return

    # report statistics
    if args.do_report:
        sample.write_report()

    # plot trajectory
    if args.do_plots and args.save_trajectory:
        sample.plot_trajectory()


if __name__ == "__main__":
    main()
