import gradient_descent
import sampling

import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

def main():
    # load optimal bias potential from the greedy gradient descent
    gd_greedy = np.load(
        os.path.join(DATA_PATH, 'langevin1d_gd_greedy.npz')
    )
    
    # plot last tilted potential and gradient
    sample = sampling.langevin_1d(beta=1)

    sample.set_bias_potential(
        a=gd_greedy['a_s'][-1],
        mus=gd_greedy['mus'],
        sigmas=gd_greedy['sigmas'],
    )
    sample.plot_potential_and_gradient(
        file_name='potential_and_gradient_gd_greedy',
    )
    
    # plot tilted potential and gradient
    gd = gradient_descent.gradient_descent(
        lr=gd_greedy['lr'],
        epochs=gd_greedy['epochs'],
        M=gd_greedy['M'],
    )
    gd.sample = sample
    gd.sample.a_opt = gd_greedy['a_opt']
    #gd.mus = gd_greedy['mus']
    #gd.sigmas = gd_greedy['sigmas']
    gd.a_s = gd_greedy['a_s']
    gd.losses = gd_greedy['losses']
    gd.plot_tilted_potentials()

if __name__ == "__main__":
    main()
