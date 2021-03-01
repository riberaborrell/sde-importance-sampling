from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_sde import LangevinSDE
from mds.utils import get_metadynamics_dir_path

import numpy as np
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'print not controlled sampling report'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize sampling object
    sde = LangevinSDE(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
    )

    # set path
    dir_path = get_metadynamics_dir_path(
        sde.example_dir_path,
        args.sigma_i_meta,
        args.k,
        args.N_meta,
    )

    # print report
    sde.print_report(dir_path)

if __name__ == "__main__":
    main()
