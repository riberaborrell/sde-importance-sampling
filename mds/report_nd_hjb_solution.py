from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_hjb_solver import Solver

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = 'Reports the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE for different xzero'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize hjb solver
    sol = Solver(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h,
    )

    # load already computed solution
    sol.load_hjb_solution()

    # write report
    sol.write_report(x=np.full(args.n, args.xzero_i))

if __name__ == "__main__":
    main()
