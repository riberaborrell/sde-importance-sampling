from mds.base_parser_1d import get_base_parser
from mds.potentials_and_gradients_1d import get_potential_and_gradient

import numpy as np
import multiprocessing as mp
import random as random
import os

def get_parser():
    parser = get_base_parser()
    parser.description = 'Samples not drifted 1D overdamped Langevin SDE (Parallel)'
    return parser

def sample_not_drifted(bla, output):
    alpha = np.array([1])
    potential_name = '1d_sym_2well'
    potential, gradient, _, _, _ = get_potential_and_gradient(potential_name, alpha)

    beta = 1

    N_lim = 10**6
    dt = 0.001

    xtemp = -1
    target_set_min = 1
    target_set_max = 3

    for n in np.arange(1, N_lim +1):
        # Brownian increment
        #dB = np.sqrt(dt) * np.random.normal(0, 1)
        dB = np.sqrt(dt) * random.normalvariate(0, 1)

        # sde update
        drift = - gradient(xtemp) * dt
        diffusion = np.sqrt(2 / beta) * dB
        xtemp += drift + diffusion

        if ((xtemp >= target_set_min) & (xtemp <= target_set_max)):
            break

    output.put(n * dt)

def main():
    args = get_parser().parse_args()

    #np.random.seed(0)
    random.seed(0)

    output = mp.Queue()
    processes = [mp.Process(target=sample_not_drifted, args=(None, output)) for x in range(args.M)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    fhts = [output.get() for p in processes]
    fht = sum(fhts) / args.M
    print('{:d}, {:2.2f}'.format(args.M, fht))

    #pool = Pool(processes=args.M)
    #result = pool.map(sample_not_drifted, [])
    #print(result)


if __name__ == "__main__":
    main()
