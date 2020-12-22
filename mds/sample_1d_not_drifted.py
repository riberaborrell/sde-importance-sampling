from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Samples not drifted 1D overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize Sampling object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain),
        target_set=np.array(args.target_set),
        is_drifted=False,
    )

    # set path
    dir_path = os.path.join(sample.example_dir_path, 'not-drifted-sampling')
    sample.set_dir_path(dir_path)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        N=args.N,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential()
        sample.plot_tilted_drift()

    # sample and compute statistics
    sample.sample_not_drifted()

    # print statistics and save data
    sample.save_not_drifted()
    sample.write_report()


if __name__ == "__main__":
    main()
