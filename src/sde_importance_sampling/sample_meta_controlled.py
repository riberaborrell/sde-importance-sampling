import os

import numpy as np

from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz
from sde_importance_sampling.importance_sampling import Sampling
from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.metadynamics import Metadynamics

def get_parser():
    parser = get_base_parser()
    parser.description = 'Sample controlled nd overdamped Langevin SDE. The bias potential ' \
                         'is parametrized with linear combination of Gaussian functions. ' \
                         'The weights are chosen from the metadynamics sampling.'
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

    # get meta object
    meta = sde.get_metadynamics_sampling(
        meta_type=args.meta_type,
        weights_type=args.weights_type,
        omega_0=args.omega_0_meta,
        sigma_i=args.sigma_i,
        dt=args.dt_meta,
        delta=args.delta_meta,
        K=args.K_meta,
        seed=args.seed,
    )

    # initialize sampling object
    sample = Sampling(sde, is_controlled=True)

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.d, args.xzero_i),
        K=args.K,
    )

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # get the corresponding Gaussian ansatz
    meta.set_ansatz()
    sample.ansatz = meta.sample.ansatz
    sample.ansatz.distributed = 'meta'
    sample.ansatz.sigma_i_meta = meta.sigma_i
    sample.ansatz.delta_meta = meta.delta
    sample.ansatz.K_meta = meta.K

    # set controlled sampling dir path
    sample.set_controlled_dir_path(meta.dir_path)

    if not args.load:

        # sample and compute statistics
        sample.sample_controlled()

        # save statistics
        sample.save()

    # load already sampled bias potential
    else:
        if not sample.load():
            return

    # print statistics
    if args.do_report:
        sample.write_report()


if __name__ == "__main__":
    main()
