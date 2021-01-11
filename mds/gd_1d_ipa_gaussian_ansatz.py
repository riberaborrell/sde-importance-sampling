from mds.base_parser_1d import get_base_parser
from mds.langevin_nd_gradient_descent import GradientDescent
from mds.langevin_1d_importance_sampling import Sampling

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Performs GD method using ipa estimator for the gradient of the loss' \
                         'function'
    parser.add_argument(
        '--epochs-lim',
        dest='epochs_lim',
        type=int,
        default=100,
        help='Set maximal number of epochs. Default: 100',
    )
    parser.add_argument(
        '--atol',
        dest='atol',
        type=float,
        default=0.01,
        help='Set absolute tolerance between value funtion and loss at xzero. Default: 0.01',
    )
    parser.add_argument(
        '--rtol',
        dest='rtol',
        type=float,
        default=0.01,
        help='Set relative tolerance between value funtion and loss at xzero. Default: 0.01',
    )
    parser.add_argument(
        '--do-epoch-plots',
        dest='do_epoch_plots',
        action='store_true',
        help='Do plots for each epoch. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize Sampling object
    sample = Sampling(
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set),
        is_drifted=True,
    )

    # set gaussian ansatz functions
    sample.set_gaussian_ansatz_functions(args.m, args.sigma)

    # set chosen coefficients
    if args.theta_init == 'optimal':
        sample.set_theta_optimal()
    elif args.theta_init == 'meta':
        sample.set_theta_from_metadynamics()
    elif args.theta_init == 'flat':
        sample.set_theta_flat()
    else:
        sample.set_theta_null()

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=args.xzero,
        N=args.N,
        dt=0.001,
        N_lim=100000,
    )

    # get value f at xzero from the reference solution
    sample.get_value_f_at_xzero()

    # initialize gradient descent object
    gd = GradientDescent(
        sample=sample,
        grad_type='ipa-value-f',
        theta_init=args.theta_init,
        lr=args.lr,
        epochs_lim=args.epochs_lim,
        do_epoch_plots=args.do_epoch_plots,
        atol=args.atol,
        rtol=args.rtol,
    )

    gd.gd_ipa()
    gd.save_gd()
    gd.write_report()

    if args.do_plots:
        gd.plot_gd_controls()
        gd.plot_gd_free_energies()
        gd.plot_gd_tilted_potentials()
        gd.plot_gd_losses()
        gd.plot_gd_time_steps()

if __name__ == "__main__":
    main()
