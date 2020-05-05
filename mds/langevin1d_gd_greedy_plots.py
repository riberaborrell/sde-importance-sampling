import sampling

import numpy as np

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')

def main():
    # load optimal bias potential from the greedy gradient descent
    optimal_bias_pot_coeff = np.load(
        os.path.join(
            DATA_PATH, 'langevin1d_bias_potential_gd_greedy.npz'
        )
    )
    samp = sampling.langevin_1d(beta=1)
    samp.set_bias_potential(
        a=optimal_bias_pot_coeff['a'],
        mus=optimal_bias_pot_coeff['mus'],
        sigmas=optimal_bias_pot_coeff['sigmas'],
    )
    samp.plot_potential_and_gradient(
        file_name='potential_and_gradient_gd_greedy',
    )

if __name__ == "__main__":
    main()
