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

    # initialize langevin sde object
    sde = LangevinSDE(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h,
    )

    # initialize sampling object
    sample = Sampling(
        sde,
        is_controlled=True,
    )

    # initialize Gaussian Ansatz
    sample.ansatz = GaussianAnsatz(sde)

    # set k-steps sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        xzero=np.full(args.n, args.xzero_i),
        N=1,
        dt=args.dt,
        k_lim=args.k,
    )

    # initialize meta nd object
    meta = Metadynamics(
        sample=sample,
        N=args.N_meta,
        xzero=np.full(args.n, args.xzero_i),
        N_lim=args.k_lim,
        k=args.k,
        sigma_i=args.sigma_i,
        seed=args.seed,
        do_updates_plots=args.do_updates_plots,
    )

    meta.metadynamics_algorithm()
    meta.save_bias_potential()
    meta.write_report()

if __name__ == "__main__":
    main()
