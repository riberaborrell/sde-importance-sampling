from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.gaussian_ansatz_functions import GaussianAnsatz
from sde_importance_sampling.importance_sampling import Sampling
from sde_importance_sampling.metadynamics import Metadynamics

import numpy as np
import os

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
        alpha = np.full(args.n, args.alpha_i)
    elif args.potential_name == 'nd_2well_asym':
        alpha = np.empty(args.n)
        alpha[0] = args.alpha_i
        alpha[1:] = args.alpha_j

    # set target set array
    if args.potential_name == 'nd_2well':
        target_set = np.full((args.n, 2), [1, 3])
    elif args.potential_name == 'nd_2well_asym':
        target_set = np.empty((args.n, 2))
        target_set[0] = [1, 3]
        target_set[1:] = [-3, 3]

    # initialize sampling object
    sample = Sampling(
        problem_name=args.problem_name,
        potential_name=args.potential_name,
        n=args.n,
        alpha=alpha,
        beta=args.beta,
        target_set=target_set,
        is_controlled=True,
    )

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        dt=args.dt,
        k_lim=args.k_lim,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
    )

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        k=args.k_meta,
        N=args.N_meta,
        seed=args.seed,
        meta_type=args.meta_type,
        weights_type=args.weights_type,
        omega_0=args.omega_0_meta,
    )

    # set path
    meta.set_dir_path()

    # load arrays
    meta.load()

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # get the corresponding Gaussian ansatz
    meta.sample.ansatz = GaussianAnsatz(n=args.n, beta=args.beta, normalized=False)
    meta.set_ansatz()
    sample.ansatz = meta.sample.ansatz

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