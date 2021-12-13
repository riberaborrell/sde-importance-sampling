from sde_importance_sampling.base_parser_nd import get_base_parser
from sde_importance_sampling.langevin_nd_importance_sampling import Sampling
from sde_importance_sampling.langevin_nd_flat_bias_potential import GetFlatBiasPotential

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'get training data for the flat bias potential'
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
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
    )

    # initialize object
    flatbias = GetFlatBiasPotential(sample)

    # set path
    flatbias.set_dir_path()

    # sample not controlled trajectories
    flatbias.get_training_data()

    flatbias.save()

    # report statistics
    if args.do_report:
        pass
        #flatbias.write_report()

    if args.do_plots:
        pass

        if args.n == 2:
            flatbias.plot_2d_training_data()


if __name__ == "__main__":
    main()
