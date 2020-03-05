#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from plotting import Plot
from decorators import timer
from tools import gradient_double_well_1d_potential, \
                  gradient_bias_potential
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

def get_parser():
    parser = argparse.ArgumentParser(description='Brownian path simulation')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=10,
        help='Set the parameter beta for the 1D MD SDE. Default: 1',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1.,
        help='Set the value of the process at time t=0. Default: 1',
    )
    parser.add_argument(
        '--target-set',
        nargs=2,
        dest='target_set',
        type=float,
        default=[0.9, 1.1],
        help='Set the target set interval. Default: [0.9, 1.1]',
    )
    return parser

@timer
def main():
    args = get_parser().parse_args()
    
    # set random seed
    if args.seed:
        np.random.seed(args.seed)
    
    # check target set
    target_set_min = args.target_set[0]
    target_set_max = args.target_set[1]
    if target_set_min >= target_set_max:
        #TODO raise error
        print("The target set interval is not valid")
        exit() 
    # 
    beta = args.beta
    
    # load bias potential parameters
    bias_potential = np.load(os.path.join(DATA_PATH, 'langevin1d_metadynamic.npz'))
    omegas = bias_potential['omegas']
    mus = bias_potential['mus']
    sigmas = bias_potential['sigmas']
    
    # number of sampled trajectories
    N_traj = 10
    
    # time interval, time steps and number of time steps
    tzero = 0
    T = 6*10**2 # 60 seconds
    N = 6*10**4 # 6000 thousand time steps
    dt = (T - tzero) / N # 0,01
    t = np.linspace(tzero, T, N+1)

    # preallocate first hitting step/time array
    FHs = np.zeros(N_traj)
    FHt = np.zeros(N_traj)

    # preallocate Girsanov Martingale
    M_FHt = np.zeros(N_traj)
    M_T = np.zeros(N_traj)

    # preallocate observable of interest
    # for f=1, g=0, and therefore W = tau t
    I = np.ones(N_traj)

    # sample of N_traj trajectories 
    for i in np.arange(N_traj):
        # Brownian increments
        dB = np.sqrt(dt)*np.random.normal(0, 1, N)
        
        # preallocate EM solution
        Xem = np.zeros(N+1) 
        Xtemp = args.xzero
        Xem[0] = Xtemp

        # preallocate martingale terms, M_t = e^(M1_t + M2_t)
        M1em = np.zeros(N+1)
        M2em = np.zeros(N+1)

        # not been in target set yet
        has_been_in_target_set = False

        for n in np.arange(1, N+1):
            # compute gradient of the bias potential evaluated at x
            dVbias = gradient_bias_potential(
                x=Xtemp,
                omegas=omegas,
                mus=mus,
                sigmas=sigmas,
            )

            # compute Xem
            drift = (- gradient_double_well_1d_potential(Xtemp) - dVbias)*dt
            diffusion = np.sqrt(2 / beta) * dB[n-1]
            Xtemp = Xtemp + drift + diffusion
            Xem[n] = Xtemp

            # evaluate the control drift
            Ut = -1 * dVbias / np.sqrt(2 / beta)

            # compute martingale terms
            M1em[n] = M1em[n-1] - Ut * dB[n-1]
            M2em[n] = M2em[n-1] - (1/2)* (Ut**2) * dt 

            # check if we have arrived to the target set
            if (has_been_in_target_set == False and
                Xtemp >= target_set_min and 
                Xtemp <= target_set_max):
                
                # change flag
                has_been_in_target_set = True

                # save first hitting time/step
                FHs[i] = n 
                FHt[i] = n * dt

                # compute martingale at the fht
                M_FHt[i] = np.exp(M1em[n] + M2em[n])
                
                # compute quantity of interest at the fht
                I[i] = np.exp(-beta * FHt[i] * dt) * M_FHt[i] 
                
                Xem[n:N+1] = np.nan
                break


        # recall that 
        # if not has_been_in_target_set:
        #    FHs[i] = 0 
        #    FHt[i] = 0
        #    M_FHt[i] = 0
        #    I[i] = 0
       
        #TODO: do I need this?
        # compute Martingale at time T
        #M_T[i] = np.exp(M1em[N] + M2em[N])
    
    # plot last trajectory
    pl = Plot(file_name='trajectory_tilted_dynamics', file_type='png')
    pl.trajectory(t, Xem)

    # compute mean and variance of I
    mean_I = np.mean(np.array([x for x in I if x > 0 ]))
    var_I = np.var(np.array([x for x in I if x > 0 ]))
    re_I = np.sqrt(var_I) / mean_I
    print('Expectation of exp(-beta*tau*t)Mt: {:2.8f}'.format(mean_I))
    print('Variance of exp(-beta*tau*t)Mt: {:2.8f}'.format(var_I))
    print('Relative error of exp(-beta*tau*t)Mt: {:2.8f}'.format(re_I))
    

if __name__ == "__main__":
    main()
