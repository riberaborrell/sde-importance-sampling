from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_importance_sampling import Sampling

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'sample drifted 1D overdamped Langevin SDE'
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
        is_drifted=True,
    )

    # set gaussian ansatz functions
    sample.set_gaussian_ansatz_functions(args.m, args.sigma)

    # set path
    sample.set_drifted_dir_path()

    # set chosen coefficients
    if args.theta == 'optimal':
        sample.set_theta_optimal()
    elif args.theta == 'meta':
        sample.set_theta_from_metadynamics()
    elif args.theta == 'gd':
        sample.set_theta_from_gd(
            gd_type='ipa-value-f',
            gd_theta_init=args.theta_init,
            gd_lr=args.lr,
        )
    else:
        sample.set_theta_null()

    # plot potential and gradient
    if args.do_plots:
        theta_stamp = 'theta-{}'.format(args.theta)
        sample.ansatz.plot_gaussian_ansatz_functions()
        sample.plot_appr_mgf(file_name=theta_stamp+'_appr_mgf')
        sample.plot_appr_free_energy(file_name=theta_stamp+'_appr_free_energy')
        sample.plot_control(file_name=theta_stamp+'_control')
        sample.plot_tilted_potential(file_name=theta_stamp+'_tilted_potential')
        sample.plot_tilted_drift(file_name=theta_stamp+'_tilted_drift')

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        M=args.M,
        dt=args.dt,
        N_lim=args.N_lim,
    )

    # sample and compute statistics
    sample.sample_drifted()

    # print statistics
    sample.write_report()


if __name__ == "__main__":
    main()
