import os

import numpy as np

from function_approximation.gaussian_ansatz import GaussianAnsatz
from meta.metadynamics import Metadynamics
from sampling.importance_sampling import Sampling
from sde.langevin_sde import LangevinSDE
from utils.base_parser import get_base_parser


def get_parser():
    parser = get_base_parser()
    parser.description = 'Metadynamics for collective variables. Metadynamics for the ' \
                         'effective dynamics cooresponding to the overdamped Langevin SDE.'
    return parser

def main():
    args = get_parser().parse_args()

    # set alpha array
    alpha = np.empty(args.d)
    alpha[0] = args.alpha_i
    alpha[1:] = args.alpha_j

    # set target set array
    target_set = np.empty((args.d, 2))
    target_set[0] = [1, 3]
    target_set[1:] = [-3, 3]

    # initialize sde object
    sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name='nd_2well_asym',
        d=args.d,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
    )

    # initialize efficient sde object
    eff_sde = LangevinSDE(
        problem_name='langevin_stop-t',
        potential_name='nd_2well',
        d=1,
        alpha=alpha[0:1],
        beta=args.beta,
        target_set=target_set[0:1],
    )

    # initialize sampling object
    sample = Sampling(eff_sde, is_controlled=False)

    # initialize Gaussian Ansatz
    sample.ansatz = GaussianAnsatz(eff_sde, normalized=False)

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        cv_type='projection',
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
        xzero=np.full(eff_sde.d, args.xzero_i),
    )

    # set path
    meta.set_dir_path()
    meta.dir_path = os.path.join(
        sde.settings_dir_path,
        meta.meta_rel_path,
    )

    if not args.load:

        # start timer
        meta.start_timer()

        # sample metadynamics trjectories
        meta.preallocate_metadynamics_coefficients()

        # set the weights of the bias functions for each trajectory
        meta.set_weights()

        # metadynamics algorythm for different samples
        for i in np.arange(meta.K):
            if args.meta_type == 'independent':
                meta.independent_metadynamics_algorithm(i)
            elif args.meta_type == 'cumulative':
                meta.cumulative_metadynamics_algorithm(i)

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


if __name__ == "__main__":
    main()
