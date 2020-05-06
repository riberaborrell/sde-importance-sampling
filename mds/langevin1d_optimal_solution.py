from plotting import Plot
from potentials_and_gradients import double_well_1d_potential, \
                                     double_well_1d_gradient
from reference_solution import langevin_1d_reference_solution
import sampling

import argparse
import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

def get_parser():
    parser = argparse.ArgumentParser(description='drifted 1D overdamped Langevin')
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 1D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
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
    return parser

def main():
    args = get_parser().parse_args()
    
    # compute reference solution
    sol = langevin_1d_reference_solution(
        beta=args.beta,
        target_set_min=args.target_set[0],
        target_set_max=args.target_set[1],
    )

    sol.compute_reference_solution()

    samp = sampling.langevin_1d(beta=args.beta)
    samp.set_uniformly_dist_ansatz_functions(
        m=args.m,
        sigma=args.sigma,
    )

    X = sol.omega_h
    
    # compute the optimal a given a basis of ansatz functions
    a = samp.ansatz_functions(X).T
    b = sol.F
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)

    # save optimal bias potential
    a_opt = x
    np.savez(
        os.path.join(DATA_PATH, 'langevin1d_bias_potential_optimal.npz'),
        a_opt=a_opt,
        mus=samp.mus,
        sigmas=samp.sigmas,
    )

    # tilted optimal potential and gradient from reference solution
    pl = Plot()
    V = double_well_1d_potential(X)
    DV = double_well_1d_gradient(X)
    F = sol.F
    Vbias = 2 * F
    U = sol.u_opt
    DVbias = - np.sqrt(2) * U
    pl.file_name='potential_and_gradient_reference_solution'
    pl.tilted_potential_and_gradient(X, V, DV, Vbias, DVbias)
   
    # tilted optimal potential and gradient on a gaussian basis 
    Vbias = samp.bias_potential(X, a_opt)
    U = samp.control(X, a_opt)
    DVbias = - np.sqrt(2) * U
    pl.file_name='potential_and_gradient_optimal'
    pl.tilted_potential_and_gradient(X, V, DV, Vbias, DVbias)

if __name__ == "__main__":
    main()
