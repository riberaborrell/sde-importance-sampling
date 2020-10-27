from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_importance_sampling import Sampling

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'sample not drifted 1D overdamped Langevin SDE'
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
    sample.set_not_drifted_dir_path()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # plot potential and gradient
    if args.do_plots:
        sample.plot_tilted_potential(file_name='tilted_potential')
        sample.plot_tilted_drift(file_name='tilted_drift')

    # sample
    sample.sample_not_drifted()

    # compute and print statistics
    sample.compute_statistics()
    sample.write_report()


if __name__ == "__main__":
    main()
