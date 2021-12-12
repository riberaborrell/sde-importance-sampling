from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_importance_sampling import Sampling
from mds.langevin_nd_sde import LangevinSDE
from mds.langevin_nd_soc_optimization_method import StochasticOptimizationMethod
from mds.langevin_nd_function_approximation import FunctionApproximation
from mds.neural_networks import GaussianAnsatzNN

import numpy as np

import torch

def get_parser():
    parser = get_base_parser()
    parser.description = ''
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

    # define uniformed distributed means
    m = args.m_i ** args.n

    mgrid_input = []
    for i in range(args.n):
        slice_i = slice(sample.domain[i, 0], sample.domain[i, 1], complex(0, args.m_i))
        mgrid_input.append(slice_i)
    means = np.mgrid[mgrid_input]
    means = np.moveaxis(means, 0, -1).reshape(m, args.n)
    means = torch.tensor(means, dtype=torch.float32)

    # define cov matrix
    cov = torch.eye(args.n)
    cov *= args.sigma_i

    # initialize gaussian ansatz nn 
    model = GaussianAnsatzNN(
        n=args.n,
        m=m,
        means=means,
        cov=cov,
    )

    # initialize function approximation
    func = FunctionApproximation(
        target_function='control',
        model=model,
    )

    # set initial choice of parametrization
    if args.theta == 'random':
        pass
    elif args.theta == 'null':
        func.zero_parameters()
    elif args.theta == 'meta':
        if not args.load:
            sde = LangevinSDE.new_from(sample)
            func.fit_parameters_from_metadynamics(sde, dt_meta=args.dt_meta,
                                                  sigma_i_meta=args.sigma_i_meta,
                                                  k=args.k, N_meta=args.N_meta)
        else:
            func.initialization = 'meta'
    elif args.theta == 'flat':
        if not args.load:
            sde = LangevinSDE.new_from(sample)
            func.fit_parameters_flat_controlled_potential(sde, N=1000)
        else:
            func.initialization = 'flat'
    elif args.theta == 'semi-flat':
        if not args.load:
            sde = LangevinSDE.new_from(sample)
            func.fit_parameters_semiflat_controlled_potential(sde)
        else:
            func.initialization = 'semi-flat'
    else:
        return

    # set dir path for nn
    func.set_dir_path(sample.settings_dir_path)

    # add nn function approximation
    sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=100000,
    )

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize SOM object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        loss_type=args.loss_type,
        optimizer='adam',
        lr=args.lr,
        n_iterations_lim=args.n_iterations_lim,
    )

    # start sgd
    if not args.load:
        try:
            sgd.som_nn()

        # save if job is manually interrupted
        except KeyboardInterrupt:
            sgd.cut_arrays()
            sgd.stop_timer()
            sgd.save_som()

    # load already run gd
    else:
        if not sgd.load_som():
            return

    # report adam
    if args.do_report:
        sgd.write_report()

    # do plots 
    if args.do_plots:

        # plot loss function, relative error and time steps
        sgd.plot_losses(args.h_hjb)
        sgd.plot_I_u()
        sgd.plot_time_steps()
        sgd.plot_cts()

        if args.n == 1:
            #adam.plot_1d_iteration()
            adam.plot_1d_iterations()

        elif args.n == 2:
            adam.plot_2d_iteration()


if __name__ == "__main__":
    main()
