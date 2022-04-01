import numpy as np

from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz
from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.importance_sampling import Sampling
from sde_importance_sampling.metadynamics import Metadynamics


def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for the nd overdamped Langevin SDE'
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

    # initialize Gaussian Ansatz
    sample.ansatz = GaussianAnsatz(sde, normalized=False)

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        meta_type=args.meta_type,
        weights_type=args.weights_type,
        omega_0=args.omega_0_meta,
        sigma_i=args.sigma_i,
        delta=args.delta_meta,
        K=args.K_meta,
        seed=args.seed,
    )

    # set sampling parameters
    meta.set_sampling_parameters(
        k_lim=args.k_lim,
        dt=args.dt_meta,
        xzero=np.full(args.d, args.xzero_i),
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
        for i in np.arange(meta.K):
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


    #meta.compute_means_distance()
    if args.do_report:
        meta.write_report()

    if args.do_plots:

        # n gaussians added for each trajectory
        meta.plot_n_gaussians()

        # time steps used for each trajectory
        meta.plot_time_steps()

        # 1d plots
        if sde.d == 1:
            idx_traj_with_updates = [i for i in range(args.K_meta) if meta.ms[i] != 0]
            for i in idx_traj_with_updates:
                meta.plot_1d_updates(i=i)
                meta.plot_1d_update()

        # 2d plots
        elif sde.d == 2:
            meta.plot_2d_means()
            meta.plot_2d_update()
            meta.plot_nd_ith_coordinate_update(i=0)
            meta.plot_nd_ith_coordinate_update(i=1)

        # 3d plots
        elif sde.d == 3:
            meta.plot_3d_means()
            meta.plot_nd_ith_coordinate_update(i=0)
            meta.plot_nd_ith_coordinate_update(i=1)

        # nd plots
        else:
            #meta.plot_nd_means()
            meta.plot_nd_ith_coordinate_update(i=0)
            meta.plot_nd_ith_coordinate_update(i=1)

        #meta.compute_probability_sampling_in_support()


if __name__ == "__main__":
    main()
