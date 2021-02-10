from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_sde import LangevinSDE

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample not controlled nd overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_controlled=False,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set path
    dir_path = os.path.join(
        sample.example_dir_path,
        'mc-sampling',
        'N_{:.0e}'.format(sample.N),
    )
    sample.set_dir_path(dir_path)

    if args.save_trajectory:
        sample.save_trajectory = True

    # plot potential and gradient
    if args.do_plots:
        pass
        return

    # sample and compute statistics
    sample.sample_not_controlled()

    # print statistics and save data
    sample.save_not_controlled()
    sample.write_report()

    # plot trajectory
    if args.save_trajectory:
        sample.plot_trajectory()


if __name__ == "__main__":
    main()
