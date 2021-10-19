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

    # set alpha array
    if args.potential_name == 'nd_2well':
        alpha = np.full(args.n, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.n)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # initialize sampling object
    sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        is_controlled=True,
    )

    # initialize Gaussian Ansatz
    sample.ansatz = GaussianAnsatz(n=args.n, normalized=False)

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        k=args.k_meta,
        N=args.N_meta,
        seed=args.seed,
        meta_type=args.meta_type,
        weights_type=args.weights_type,
        omega_0=args.omega_0_meta,
        do_updates_plots=args.do_updates_plots,
    )

    # set sampling parameters
    meta.set_sampling_parameters(
        k_lim=args.k_lim,
        dt=args.dt_meta,
        xzero=np.full(args.n, args.xzero_i),
    )

    # set path
    meta.set_dir_path()

    if not args.load:

        # start timer
        meta.start_timer()

        # sample metadynamics trjectories
        meta.preallocate_metadynamics_coefficients()

        # set the weights of the bias functions for each trajectory
        meta.set_weights()

        # metadynamics algorythm for different samples
        for i in np.arange(meta.N):
            if args.meta_type == 'ind':
                meta.independent_metadynamics_algorithm(i)
            elif args.meta_type == 'cum':
                meta.cumulative_metadynamics_algorithm(i)
            elif args.meta_type == 'cum-nn':
                pass
                #meta.cumulative_metadynamics_algorithm(i)

        # stop timer
        meta.stop_timer()

        # save bias potential
        meta.save()

    # load already sampled bias potential
    else:
        if not meta.load():
            return


    if args.do_report:
        meta.write_report()

    if args.do_plots:

        # n gaussians added for each trajectory
        #meta.plot_n_gaussians()

        # 1d plots
        if sample.n == 1:
            idx_traj_with_updates = [i for i in range(args.N_meta) if meta.ms[i] != 0]
            for i in idx_traj_with_updates:
                meta.plot_1d_updates(i=i)
                meta.plot_1d_update()

        # 2d plots
        elif sample.n == 2:
            meta.plot_2d_update()
            meta.plot_2d_means()

        # 3d plots
        elif sample.n == 3:
            meta.plot_3d_means()

        # nd plots
        else:
            meta.plot_nd_means()


if __name__ == "__main__":
    main()
