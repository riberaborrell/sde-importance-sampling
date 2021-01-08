from mds.base_parser_nd import get_base_parser
from mds.gaussian_nd_ansatz_functions import GaussianAnsatz
from mds.langevin_nd_importance_sampling import Sampling

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Test nd Gaussian ansatz module'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize Sampling object
    sample = Sampling(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        is_drifted=False,
    )

    # initialize Gaussians
    ansatz = GaussianAnsatz(args.n)

    # distribute them uniformly along each axis
    ansatz.set_unif_dist_ansatz_functions(m_x=10, sigma_x=0.5)
    ansatz.set_dir_path(sample.example_dir_path)

    # plot 1d and 2d cases
    #ansatz.plot_1d_multivariate_normal_pdf(j=4)
    ansatz.plot_2d_multivariate_normal_pdf(j=45)

if __name__ == "__main__":
    main()
