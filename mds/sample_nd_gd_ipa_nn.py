from mds.base_parser_nd import get_base_parser

from mds.neural_networks import TwoLayerNet

import numpy as np

import torch
import torch.optim as optim

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # initialize control parametrization by a nn 
    dim_in, dim_1, dim_out = 1, 3, 1
    model = TwoLayerNet(dim_in, dim_1, dim_out)

    # zeros parameters
    model.zero_parameters()

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

        # compute loss
        loss, tilted_loss = sample_loss(model)
        print('{:d}, {:2.3f}'.format(update, loss))

        # compute gradients
        tilted_loss.backward()

        # update parameters
        optimizer.step()

def save_nn_coefficients():
    pass

def double_well_1d_gradient(x):
    alpha = 1
    return 4 * alpha * x * (x**2 - 1)

def sample_loss(model):
    beta = 1

    N = 100

    dt = 0.001
    k_max = 100000

    #states = torch.empty(0, dtype=np.float)
    #fhts = np.empty(N, dtype=np.int)

    loss = np.zeros(N)
    tilted_loss = torch.empty(0)
    #a = torch.empty(0)
    #b = torch.empty(0)
    #c = torch.empty(0)

    #tilted_loss = torch.zeros(N)
    #a = torch.zeros(N)
    #b = torch.zeros(N)
    #c = torch.zeros(N)

    for i in np.arange(N):

        # initialize trajectory
        xtemp = -1
        #tilted_loss_temp = torch.zeros(1)
        a_temp = torch.zeros(1)
        b_temp = torch.zeros(1)
        c_temp = torch.zeros(1)
        #states = np.append(states, xtemp)
        #print(id(a), id(b), id(c))

        for k in np.arange(1, k_max + 1):

            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)

            # control
            xtemp_tensor = torch.tensor([xtemp], dtype=torch.float)
            utemp_tensor = model.forward(xtemp_tensor)
            utemp_tensor_det = model.forward(xtemp_tensor).detach()
            utemp = model.forward(xtemp_tensor).detach().numpy()[0]

            # sde update
            drift = (- double_well_1d_gradient(xtemp) + np.sqrt(2) * utemp) * dt
            diffusion = np.sqrt(2 / beta) * dB
            xtemp += drift + diffusion
            #states = np.append(states, xtemp)

            # update statistics
            loss[i] += 0.5 * (utemp ** 2) * dt
            #tilted_loss_temp = tilted_loss_temp + utemp_tensor_det * utemp_tensor * dt
            a_temp = a_temp + utemp_tensor_det * utemp_tensor * dt
            b_temp = b_temp + 0.5 * (utemp_tensor_det ** 2) * dt
            c_temp = c_temp - np.sqrt(beta) * dB * utemp_tensor
            #a = torch.cat((a, utemp_tensor_det * utemp_tensor * dt))
            #b = torch.cat((b, 0.5 * (utemp_tensor_det ** 2) * dt))
            #c = torch.cat((c, - np.sqrt(beta) * dB * utemp_tensor))

            #a[i] = a[i] + utemp_tensor_det * utemp_tensor * dt
            #b[i] = b[i] + 0.5 * (utemp_tensor_det ** 2) * dt
            #c[i] = c[i] - np.sqrt(beta) * dB * utemp_tensor

            # stop if xtemp in target set
            if xtemp >= 1:
                loss[i] += k * dt
                b_temp = b_temp + k * dt
                tilted_loss_temp = a_temp - b_temp * c_temp
                tilted_loss = torch.cat((tilted_loss, tilted_loss_temp))
                #b = torch.cat((b,  torch.tensor([k * dt])))
                #tilted_loss = torch.cat((tilted_loss, a[i] - b[i] * c[i]))

                #b[i] = b[i] + k * dt
                #tilted_loss[i] = a[i] - b[i] * c[i]
                break

    return np.mean(loss), torch.mean(tilted_loss)

def compute_loss(model, states, fhts):

    N = fhts.shape[0]

    loss = torch.zeros(N)
    a = torch.zeros(N)
    b = torch.zeros(N)
    c = torch.zeros(N)

    # control
    inputs = torch.tensor([[xtemp]], dtype=torch.float)
    utemp = model.forward(inputs)[0, 0]
    utemp_det = model.forward(inputs)[0, 0].detach()

    # ipa statistics 
    a[i] += utemp_det * utemp * dt
    b[i] += 0.5 * (utemp_det ** 2) * dt
    c[i] -= np.sqrt(beta) * dB * utemp

    b[i] += k * dt
    loss[i] = a[i] - b[i] * c[i]



if __name__ == "__main__":
    main()
