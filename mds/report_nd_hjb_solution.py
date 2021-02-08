from mds.base_parser_nd import get_base_parser
from mds.langevin_nd_hjb_solver import Solver
from mds.langevin_nd_sde import LangevinSDE

import numpy as np

import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Reports the numerical solution of the HJB equation associated to' \
                         'the overdamped Langevin SDE for different xzero'
    return parser

def main():
    args = get_parser().parse_args()

    # initialize langevin sde object
    lang_sde = LangevinSDE(
        n=args.n,
        potential_name=args.potential_name,
        alpha=np.full(args.n, args.alpha_i),
        beta=args.beta,
        h=args.h,
    )

    def f(x):
        return 1

    def g(x):
        return 0

    # initialize solver
    sol = Solver(
        sde=lang_sde,
        f=f,
        g=g,
    )

    # load already computed solution
    sol.load_hjb_solution()

    # write report
    sol.write_report(x=np.full(args.n, args.xzero_i))

if __name__ == "__main__":
    main()
