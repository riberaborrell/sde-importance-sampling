from mds.base_parser_2d import get_base_parser
from mds.langevin_2d_importance_sampling import Sampling

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'sample not drifted 2D overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize langevin_1d object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain).reshape(2, 2),
        target_set=np.array(args.target_set).reshape(2, 2),
        h=args.h,
        is_drifted=False,
    )

    # set path
    sample.set_not_drifted_dir_path()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.array(args.xzero),
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential_surface()
        sample.plot_tilted_potential_contour()
        #sample.plot_tilted_drift()

    # sample and compute statistics
    sample.sample_not_drifted()

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()
