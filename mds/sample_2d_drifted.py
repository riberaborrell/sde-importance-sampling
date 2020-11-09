from mds.base_parser_2d import get_base_parser
from mds.langevin_2d_importance_sampling import Sampling

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'sample drifted 2D overdamped Langevin SDE'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize Sampling object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        domain=np.array(args.domain).reshape(2, 2),
        target_set=np.array(args.target_set).reshape(2, 2),
        h=args.h,
        is_drifted=True,
    )

    # set gaussian ansatz functions
    m_x, m_y = args.m
    sigma_x, sigma_y = args.sigma
    sample.set_gaussian_ansatz_functions(m_x, m_y, sigma_x, sigma_y)

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
        sample.ansatz.plot_multivariate_normal_pdf(j=0)
        #sample.ansatz.plot_gaussian_ansatz_functions(omega=None)
        sample.plot_appr_psi_surface()
        sample.plot_appr_psi_contour()
        sample.plot_appr_free_energy_surface()
        sample.plot_appr_free_energy_contour()
        sample.plot_control()
        sample.plot_tilted_potential_surface()
        sample.plot_tilted_potential_contour()
        sample.plot_tilted_drift()
    return

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.array(args.xzero),
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
