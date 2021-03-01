from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_sde import LangevinSDE

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
    dir_path = os.path.join(
        sde.example_dir_path,
        'mc-sampling',
        'N_{:.0e}'.format(args.N),
    )

    # print report
    sde.print_report(dir_path)

if __name__ == "__main__":
    main()
