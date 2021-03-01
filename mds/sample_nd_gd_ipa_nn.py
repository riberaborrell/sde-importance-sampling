from mds.base_parser_nd import get_base_parser

from mds.neural_networks import TwoLayerNet

import numpy as np

import torch
import torch.optim as optim

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def sample_loss(model):
    N = 1000
    k_max = 100000

    dt = 0.001

    J = np.zeros(N)
    phi = np.zeros(N)
    grad_phi = np.zeros(N)
    grad_S = np.zeros(N)

    for i in np.arange(N):

        # initialize trajectory
        xtemp = -1
        for k in np.arange(1, k_max + 1):

            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)

            # control
            inputs = torch.tensor([[xtemp]], dtype=torch.float)
            utemp = model.forward(inputs)
            utemp_det = model.forward(inputs).detach()
            breakpoint()

            # ipa statistics 
            phi[i] += 0.5 * (utemp ** 2) * self.dt
            grad_phi_temp += np.sum(utemp[:, np.newaxis, :] * btemp, axis=2) * self.dt
            grad_S_temp -= np.sqrt(self.beta) * np.sum(dB[:, np.newaxis, :] * btemp, axis=2)




def main():
    args = get_parser().parse_args()

    # initialize control parametrization by a nn 
    dim_in, dim_1, dim_out = 1, 3, 1
    model = TwoLayerNet(dim_in, dim_1, dim_out)

    # zeros parameters
    model.zero_parameters()

    N = 1
    inputs = torch.rand(N, dim_in)
    control = model.forward(inputs)

    # gd
    updates_max = 100

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    for update in np.arange(updates_max):
        # reset gradients
        optimizer.zero_grad()

        # sample loss
        loss = sample_loss(model)

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()



if __name__ == "__main__":
    main()
