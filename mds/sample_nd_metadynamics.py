from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_metadynamics import Metadynamics
from mds.langevin_nd_sde import LangevinSDE

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for the nd overdamped Langevin SDE'
    parser.add_argument(
        '--do-updates-plots',
        dest='do_updates_plots',
        action='store_true',
        help='Do plots after adding a gaussian. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h,
        is_controlled=True,
    )

    # initialize Gaussian Ansatz
    sample.ansatz = GaussianAnsatz(n=args.n)

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        k=args.k,
        N=args.N_meta,
        sigma_i=args.sigma_i_meta,
        seed=args.seed,
        do_updates_plots=args.do_updates_plots,
    )

    # set sampling parameters
    meta.set_sampling_parameters(
        k_lim=args.k_lim,
        dt=args.dt,
        xzero=np.full(args.n, args.xzero_i),
    )

    # set path
    meta.set_dir_path()

    if not args.load:
        # sample metadynamics trjectories
        meta.metadynamics_algorithm()

        # save bias potential
        meta.save_bias_potential()

    # load already sampled bias potential
    else:
        if not meta.load_bias_potential():
            return

    if args.do_report:
        meta.write_report()

    if args.do_plots:
        if sample.n == 1:
            meta.plot_1d_updates()
            meta.plot_1d_update()
        elif sample.n == 2:
            meta.plot_2d_update()

if __name__ == "__main__":
    main()
