from mds.base_parser_1d import get_base_parser
from mds.langevin_1d_hjb_solver import Solver

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Report the numerical solution of the HJB equation associated to the' \
                         'overdamped Langevin SDE evaluated at xzero'
    return parser

def main():
    args = get_parser().parse_args()

    def f(x):
        return 1

    def g(x):
        return 0

    # get hjb solver
    sol = Solver(
        f=f,
        g=g,
        potential_name=args.potential_name,
        alpha=np.array(args.alpha),
        beta=args.beta,
        target_set=np.array(args.target_set),
    )

    # load already computed solution
    sol.load_reference_solution()

    # write report for a given xzero
    sol.write_report(x=args.xzero)

if __name__ == "__main__":
    main()
