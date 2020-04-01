import argparse
import numpy as np

from decorators import timer
from plotting import Plot
from tools import gradient_double_well_1d_potential, \
                  gradient_bias_potential

from datetime import datetime
import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
FIGURES_PATH = os.path.join(MDS_PATH, 'figures')

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
        default=5,
        help='Set the parameter beta for the 1D MD SDE. Default: 5',
    )
    parser.add_argument(
        '--xzero',
        dest='xzero',
        type=float,
        default=-1.,
        help='Set the value of the process at time t=0. Default: -1',
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
        '--num-trajectories',
        dest='M',
        type=int,
        default=10**2,
        help='Set number of trajectories to sample. Default: 100',
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
    #bias_potential = np.load(os.path.join(DATA_PATH, 'langevin1d_metadynamic.npz'))
    bias_potential = np.load(os.path.join(DATA_PATH, 'langevin1d_tilted_potential.npz'))
    omegas = bias_potential['omegas']
    mus = bias_potential['mus']
    sigmas = bias_potential['sigmas']
    
    # number of trajectories to sample
    M = args.M
    
    # time interval, time steps and number of time steps
    tzero = 0
    T = 10**3
    N = 10**6
    dt = (T - tzero) / N
    t = np.linspace(tzero, T, N+1)
    
    # 1D MD SDE is dX_t = -grad V(X_t)dt + sqrt(2 beta**-1)dB_t, X_0 = x,
    # preallocate EM solution
    Xem = np.zeros((M, N+1))
    Xem[:, 0] = args.xzero

    # preallocate first hitting step/time array
    FHs = np.zeros(M)
    FHt = np.zeros(M)

    # preallocate Girsanov Martingale
    M_FHt = np.zeros(M)

    # preallocate observable of interest I
    # for f=1, g=0, and therefore W = tau t
    I = np.ones(M)

    # sample of M trajectories 
    for i in np.arange(M):
        # Brownian increments
        dB = np.sqrt(dt)*np.random.normal(0, 1, N)
        
        # initialize Xtemp
        Xtemp = Xem[i, 0]

        # preallocate martingale terms, M_t = e^(M1_t + M2_t)
        M1em = np.zeros(N+1)
        M2em = np.zeros(N+1)

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
            Xem[i, n] = Xtemp
            
            # evaluate the control drift
            #Ut = - dVbias / np.sqrt(2 / beta)

            # compute martingale terms
            #M1em[n] = M1em[n-1] - Ut * dB[n-1]
            M1em[n] = M1em[n-1] + np.sqrt(beta / 2) * dVbias * dB[n-1]             

            #M2em[n] = M2em[n-1] - (1/2)* (Ut**2) * dt 
            M2em[n] = M2em[n-1] - (beta / 4) * (dVbias**2) * dt 

            # check if we have arrived to the target set
            if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                
                # save first hitting time/step
                FHs[i] = n 
                FHt[i] = n * dt

                # compute martingale at the fht
                #M_FHt[i] = np.exp(M1em[n] + M2em[n])
                
                # compute quantity of interest at the fht
                #I[i] = np.exp(FHt[i]) * M_FHt[i] 
                I[i] = np.exp(-beta * FHt[i] + M1em[n] + M2em[n]) 
                
                Xem[i, n:N+1] = np.nan
                break

    # plot last trajectory
    pl = Plot(file_name='last_tilted_trajectory', file_type='png')
    pl.trajectory(t, Xem[0])

    # sort out trajectories which have not arrived
    FHt = np.array([t for t in FHt if t > 0])
    I = np.array([x for x in I if x != 1])

    # compute mean and variance of tau
    mean_tau = np.mean(FHt)
    var_tau = np.var(FHt)
    if mean_tau != 0:
        re_tau = np.sqrt(var_tau) / mean_tau
    else:
        re_tau = np.nan

    # compute mean and variance of I
    mean_I = np.mean(I)
    var_I = np.var(I)
    if mean_I != 0:
        re_I = np.sqrt(var_I) / mean_I
    else:
        re_I = np.nan

    # save output in a file
    time_stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(DATA_PATH, 'langevin_1d_2well_tilted_'+ time_stamp + '.txt')
    f = open(file_path, "w")
    f.write('beta: {:2.1f}\n'.format(beta))
    f.write('dt: {:2.4f}\n'.format(dt))
    f.write('Y_0: {:2.1f}\n'.format(args.xzero))
    f.write('target set: [{:2.1f}, {:2.1f}]\n\n'.format(target_set_min, target_set_max))

    f.write('sampled trajectories: {:d}\n'.format(M))
    f.write('% trajectories which have arrived: {:2.2f}\n\n'.format(len(FHt) / M))

    f.write('Expectation of tau: {:2.4f}\n'.format(mean_tau))
    f.write('Variance of tau: {:2.4f}\n'.format(var_tau))
    f.write('Relative error of tau: {:2.4f}\n\n'.format(re_tau))
    
    f.write('Expectation of exp(-beta * tau): {:2.4e}\n'.format(mean_I))
    f.write('Variance of exp(-beta * tau): {:2.4e}\n'.format(var_I))
    f.write('Relative error of exp(-beta * tau): {:2.4e}\n\n'.format(re_I))
    
    f.close()
    

if __name__ == "__main__":
    main()
