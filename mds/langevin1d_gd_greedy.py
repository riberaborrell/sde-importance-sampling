import gradient_descent
from plotting import Plot

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Gradient Descent')
    parser.add_argument(
        '--learning-rate',
        dest='lr',
        type=float,
        default=1.,
        help='Set learning rate. Default:',
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=15,
        help='Set number of epochs. Default: 15',
    )
    parser.add_argument(
        '--M',
        dest='M',
        type=int,
        default=500,
        help='Set number of trajectories to sample. Default: 500',
    )
    parser.add_argument(
        '--m',
        dest='m',
        type=int,
        default=10,
        help='Set the number of uniformly distributed ansatz functions \
              that you want to use. Default: 10',
    )
    parser.add_argument(
        '--sigma',
        dest='sigma',
        type=float,
        default=0.2,
        help='Set the standard deviation of the ansatz functions. Default: 0.2',
    )
    parser.add_argument(
        '--grad-type',
        dest='grad_type',
        choices=['ipa', 'fd'],
        default='ipa',
        help='Type of gradient computation (ipa or fd). Default: ipa',
    )
    parser.add_argument(
        '--delta',
        dest='delta',
        type=float,
        default=None,
        help='Set step size for the finite differences. Default: None',
    )
    parser.add_argument(
        '--a-type',
        dest='a_type',
        choices=['optimal', 'meta', 'null'],
        default='optimal',
        help='Type of drift. Default: optimal',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # initialize gradient descent object
    soc = gradient_descent.gradient_descent(
        lr=args.lr,
        epochs=args.epochs,
        M=args.M,
        grad_type=args.grad_type,
        delta=args.delta,
        do_plots=args.do_plots,
    )

    # set the sde to sample and the sampling parameters
    soc.set_sample()

    # set ansatz functions and a optimal
    soc.set_ansatz_functions_greedy(m=args.m, sigma=args.sigma)
    print(soc.sample.a_opt)

    # set initial a
    if args.a_type == 'optimal':
        soc.set_a_optimal_greedy()
    elif args.a_type == 'meta':
        soc.set_a_from_metadynamics_greedy()
    else:
        soc.set_a_null_greedy()

    soc.gradient_descent_greedy()
    soc.save_statistics()
    soc.write_statistics()

    # plot tilted potential and gradient
    if args.do_plots:
        soc.plot_tilted_potentials()

if __name__ == "__main__":
    main()
