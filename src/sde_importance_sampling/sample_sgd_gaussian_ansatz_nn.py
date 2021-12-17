from sde_importance_sampling.base_parser import get_base_parser
from sde_importance_sampling.importance_sampling import Sampling
from sde_importance_sampling.langevin_sde import LangevinSDE
from sde_importance_sampling.soc_optimization_method import StochasticOptimizationMethod
from sde_importance_sampling.function_approximation import FunctionApproximation
from sde_importance_sampling.neural_networks import GaussianAnsatzNN

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
        initialization=args.theta,
    )

    # get dir path for nn
    if args.theta in ['random', 'null', 'not-controlled']
        dir_path = sample.settings_dir_path

    elif args.theta == 'meta':

        # get metadynamics
        sde = LangevinSDE.new_from(sample)
        meta = sde.get_metadynamics_sampling(args.meta_type, args.weights_type,
                                             args.omega_0_meta, args.k_meta,
                                             args.N_meta, args.seed)
        dir_path = meta.dir_path

    # set dir path for nn
    func.set_dir_path(dir_path)

    # set initial parameters
    if args.theta == 'random':

        # the nn parameters are randomly initialized 
        pass

    elif args.theta == 'null':

        # set nn parameters to be zero
        func.zero_parameters()

    elif args.theta == 'not-controlled':

        # train nn parameters such that control is zero
        sde = LangevinSDE.new_from(sample)
        func.train_parameters_classic(sde=sde)
        #func.train_parameters_alternative(sde=sde)

    elif args.theta == 'meta':

        # train parameters if not trained yet
        func.train_parameters_classic(meta=meta)
        #func.train_parameters_alternative(meta=meta)
    else:
        return

    # add nn function approximation
    sample.nn_func_appr = func

    # set sampling and Euler-Marujama parameters
    sample.set_sampling_parameters(
        seed=args.seed,
        xzero=np.full(args.n, args.xzero_i),
        N=args.N,
        dt=args.dt,
        k_lim=args.k_lim,
    )

    # set u l2 error flag
    if args.do_u_l2_error:
        sample.do_u_l2_error = True

    # initialize SOM object
    sgd = StochasticOptimizationMethod(
        sample=sample,
        loss_type=args.loss_type,
        optimizer=args.optimizer,
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
