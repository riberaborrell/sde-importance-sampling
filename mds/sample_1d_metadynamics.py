from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_importance_sampling import Sampling
from mds.langevin_1d_metadynamics import Metadynamics
from mds.gaussian_1d_ansatz_functions import GaussianAnsatz
from mds.plots_1d import Plot1d

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for the 1D overdamped Langevin SDE'
    parser.add_argument(
        '--num-samples',
        dest='num_samples',
        type=int,
        default=5,
        help='Number of samples. Default: 5',
    )
    parser.add_argument(
        '--k',
        dest='k',
        type=int,
        default=100,
        help='Steps before adding a bias function. Default: 100',
    )
    parser.add_argument(
        '--do-updates-plots',
        dest='do_updates_plots',
        action='store_true',
        help='Do plots after adding a gaussian. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize 1d sampling object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain),
        target_set=np.array(args.target_set),
        h=args.h,
        is_drifted=False,
    )

    # initialize gaussian ansatz
    sample.ansatz = GaussianAnsatz(sample.domain)

    # set k-steps sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.k,
    )

    # initialize meta 1d object
    meta = Metadynamics(
        sample=sample,
        num_samples=args.num_samples,
        xzero=np.array(args.xzero),
        N_lim=args.N_lim,
        k=args.k,
        seed=args.seed,
        do_updates_plots=args.do_updates_plots,
    )
    meta.metadynamics_algorithm()
    meta.save_bias_potential()
    meta.write_report()

    # plot potential and gradient
    if args.do_plots:
        # set bias potential
        sample.set_bias_potential(meta.theta, meta.mus, meta.sigmas)
        sample.plot_appr_psi(dir_path=meta.dir_path)
        sample.plot_appr_free_energy(dir_path=meta.dir_path)
        sample.plot_control(dir_path=meta.dir_path)
        sample.plot_tilted_potential(dir_path=meta.dir_path)
        sample.plot_tilted_drift(dir_path=meta.dir_path)

if __name__ == "__main__":
    main()
