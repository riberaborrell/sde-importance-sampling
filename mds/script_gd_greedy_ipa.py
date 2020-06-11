from potentials_and_gradients import get_potential_and_gradient

import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import stats
from scipy.optimize import lsq_linear

import argparse
import os
import pdb

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
GD_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/gradient_descent_greedy')


def get_parser():
    parser = argparse.ArgumentParser(description='IPA')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=None,
        help='Set seed. Default: None',
    )
    parser.add_argument(
        '--potential',
        dest='potential_name',
        choices=['sym_1well', 'sym_2well', 'asym_2well'],
        default='sym_2well',
        help='Set the potential for the 1D MD SDE. Default: symmetric double well',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1,
        help='Set the parameter beta for the 1D MD SDE. Default: 1',
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
        '--xzero',
        dest='xzero',
        type=float,
        default=-1,
        help='Set the initial position. Default: -1',
    )
    return parser

def main():
    args = get_parser().parse_args()

    # set seed
    if args.seed:
        np.random.seed(args.seed)

    # set potential
    potential, gradient = get_potential_and_gradient(args.potential_name)

    # sampling parameters
    beta = args.beta
    xzero = args.xzero
    target_set = args.target_set
    M = 100

    # get reference solution
    ref_sol = np.load(
        os.path.join(DATA_PATH, 'langevin1d_reference_solution.npz')
    )
    omega_h = ref_sol['omega_h']
    Psi_opt = ref_sol['Psi']
    F_opt = ref_sol['F']
    u_opt = ref_sol['u_opt']

    # ansatz functions basis
    m = 50
    omega = np.array([-3, 3])
    mus, sigmas = set_unif_dist_ansatz_functions(omega, target_set, m)
    sigma = sigmas[0]

    # plot ansatz functions
    #plot_ansatz_functions(omega_h, mus, sigmas)

    # get optimal coefficients
    a_opt = get_optimal_coefficients(omega_h, target_set, u_opt, mus, sigmas)

    # set gd parameters
    lr = 0.08
    epochs = 10

    # coefficients, performance function and free energy
    a_s = np.zeros((epochs + 1, m))
    loss = np.zeros(epochs + 1)
    F = np.zeros((epochs + 1, len(omega_h)))

    # set initial coefficients
    a_init_type = 'null'
    a_s[0, :] = set_initial_coefficients(m, sigma, a_opt, a_init_type)

    # gradient descent
    for epoch in range(epochs):
        F[epoch, :] = compute_free_energy(omega_h, target_set, a_s[epoch, :], mus, sigmas)
        print(epoch)
        #plot_control(epoch, omega_h, u_opt, a_s[epoch, :], mus, sigmas)
        #plot_free_energy(epoch, omega_h, potential, F_opt, F[epoch])
        plot_tilted_potential(epoch, omega_h, potential, F_opt, F[epoch])

        mean_fht, mean_cost, mean_gradJ, mean_gradSh \
            = sample_loss(gradient, beta, xzero, target_set,
                          M, m, a_s[epoch], mus, sigmas)

        # save performance function
        loss[epoch] = mean_fht + mean_cost

        # gradient of the loss function (gradient of an expectation!)
        grad_loss = mean_gradJ + loss[epoch] * mean_gradSh

        # Update parameters
        a_s[epoch + 1, :] = a_s[epoch, :] - lr * grad_loss

    epoch += 1
    print(epoch)
    #plot_control(epoch, omega_h, u_opt, a_s[epoch, :], mus, sigmas)
    #plot_tilted_potential_and_free_energy(epoch, omega_h, potential, F_opt,
    #                                      a_s[epoch, :], mus, sigmas)

def get_ansatz_functions(x, mus, sigmas):
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)
    return stats.norm.pdf(x, mus, sigmas)


def set_unif_dist_ansatz_functions(omega, target_set, m):
    # assume target_set is connected and contained in omega
    omega_min, omega_max = omega
    target_set_min, target_set_max = target_set

    # set grid 
    h = 0.001
    N = int((omega_max - omega_min) / h)
    omega = np.around(np.linspace(omega_min, omega_max, N + 1), decimals=3)

    # get indexes for nodes in/out the target set
    idx_ts = np.where((omega >= target_set_min) & (omega <= target_set_max))[0]
    idx_nts = np.where((omega < target_set_min) | (omega > target_set_max))[0]
    idx_l = np.where(omega < target_set_min)[0]
    idx_r = np.where(omega > target_set_max)[0]

    # compute ratio of nodes in the left/right side of the target set
    ratio_left = idx_l.shape[0] / idx_nts.shape[0]
    ratio_right = idx_r.shape[0] / idx_nts.shape[0]

    # assigm number of ansatz functions in each side
    m_left = int(np.round(m * ratio_left))
    m_right = int(np.round(m * ratio_right))
    assert m == m_left + m_right

    # distribute ansatz functions unif (in each side)
    mus_left = np.around(
        #np.linspace(omega[idx_l][0], omega[idx_l][-1], m_left),
        np.linspace(omega[idx_l][0], omega[idx_l][-1], m_left + 2)[1:-1],
        decimals=3,
    )
    mus_right = np.around(
        #np.linspace(omega[idx_r][0], omega[idx_r][-1], m_right),
        np.linspace(omega[idx_r][0], omega[idx_r][-1], m_right + 2)[1:-1],
        decimals=3,
    )
    mus = np.concatenate((mus_left, mus_right), axis=0)

    # compute sigma
    sigma_left = mus_left[1] - mus_left[0]
    sigma_right = mus_right[1] - mus_right[0]
    sigma = (sigma_left + sigma_right) / 2
    sigmas = sigma * np.ones(m)

    mu_min = -2.2
    mu_max = 0.8
    #mu_min = -1.1
    #mu_max = 1.9
    #m = 51
    #mus = np.linspace(mu_min, mu_max, m)
    #sigma = (mu_max - mu_min) / (m - 1)
    #sigmas = sigma * np.ones(m)
    #m = 11
    #mus = np.linspace(-2, 2, m)
    #sigma = 1
    #sigmas = sigma * np.ones(m)

    return mus, sigmas


def get_optimal_coefficients(omega, target_set, u_opt, mus, sigmas):
    # ansatz functions at the grid
    ansatz_functions = get_ansatz_functions(omega, mus, sigmas)
    #ansatz_functions = get_ansatz_functions(omega[200: 900], mus, sigmas)
    #ansatz_functions = get_ansatz_functions(omega[500:1250], mus, sigmas)
    a = ansatz_functions.T

    # optimal control (from ref solution) at the grid
    b = u_opt
    #b = u_opt[200: 900]
    #b = u_opt[500:1250]

    # solve lin system of equations by using least squares
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
    #res = lsq_linear(a, b, bounds=(0, 'inf'), lsmr_tol='auto', verbose=1)
    #a_opt = res.x

    return x


def set_initial_coefficients(m, sigma, a_opt, a_init_type):
    pert_factor = sigma

    # set a_init
    if a_init_type == 'null':
        a_init = np.zeros(m)
    elif a_init_type == 'null-pert':
        a_init = pert_factor * np.ones(m)
        #a_init = pert_factor * np.random.choice([-1, 1], m)
    elif a_init_type == 'opt':
        a_init = a_opt
    elif a_init_type == 'opt-pert':
        a_init = a_opt + pert_factor * np.ones(m)
        #a_init = a_opt * np.random.choice([-1, 1], m)

    # impose positive/negative value
    #for j in range(m):
    #    a_s[0, j] *= (- np.sign(a_s[0, j]))
    return a_init


def sample_loss(gradient, beta, xzero, target_set, M, m, a, mus, sigmas):

    # sampling parameters
    target_set_min, target_set_max = target_set

    # Euler-Majurama
    dt = 0.001
    N = 100000

    # initialize statistics 
    fht = np.zeros(M)
    cost = np.zeros(M)
    gradJ = np.zeros((m, M))
    gradSh = np.zeros((m, M))

    # initialize temp variables
    Xtemp = xzero * np.ones(M)
    cost_temp = np.zeros(M)
    sum_grad_gh_temp = np.zeros((m, M))
    gradSh_temp = np.zeros((m, M))

    # has arrived in target set
    been_in_target_set = np.repeat([False], M)

    for n in np.arange(1, N+1):
        #breakpoint()
        normal_dist_samples = np.random.normal(0, 1, M)

        # Brownian increment
        dB = np.sqrt(dt) * normal_dist_samples

        # control
        btemp = get_ansatz_functions(Xtemp, mus, sigmas)
        utemp = np.dot(a, btemp)

        # compute gradient
        tilted_gradient = gradient(Xtemp) - np.sqrt(2) * utemp

        # SDE iteration
        drift = - tilted_gradient * dt
        diffusion = np.sqrt(2 / beta) * dB
        Xtemp += drift + diffusion

        # compute cost, ...
        cost_temp += 0.5 * (utemp ** 2) * dt
        sum_grad_gh_temp += dt * utemp * btemp
        gradSh_temp += normal_dist_samples * btemp

        # trajectories in the target set
        is_in_target_set = ((Xtemp >= target_set_min) & (Xtemp <= target_set_max))

        # indices of trajectories new in the target set
        new_idx = np.where(
            (is_in_target_set == True) & (been_in_target_set == False)
        )[0]
        if len(new_idx) == 0:
            continue

        # update list of indices whose trajectories have been in the target set
        been_in_target_set[new_idx] = True

        # save first hitting time
        fht[new_idx] = n * dt
        cost[new_idx] = cost_temp[new_idx]
        #gradSh[:, new_idx] = - gradSh_temp[:, new_idx] * beta * np.sqrt(dt / 2)
        gradSh[:, new_idx] = - gradSh_temp[:, new_idx] * np.sqrt(dt * beta)
        gradJ[:, new_idx] = sum_grad_gh_temp[:, new_idx] \
                          - (n * dt + cost_temp[new_idx]) \
                          * gradSh[:, new_idx]

        # check if all trajectories have arrived to the target set
        if been_in_target_set.all() == True:
            break

    # compute averages
    mean_fht = np.mean(fht)
    mean_cost = np.mean(cost)
    mean_gradJ = np.mean(gradJ, axis=1)
    mean_gradSh = np.mean(gradSh, axis=1)

    return mean_fht, mean_cost, mean_gradJ, mean_gradSh


def compute_free_energy(omega, target_set, a, mus, sigmas):
    X = omega
    dx = X[1] - X[0]
    N = len(omega)

    # compute Free energy from the control
    F = np.zeros(N)
    for k in np.flip(np.arange(0, N-1)):
    #for k in np.arange(1, N):
        v = get_ansatz_functions(X[k], mus, sigmas)
        u_at_x = np.dot(a, v)
        F[k - 1] = F[k] + (1 / np.sqrt(2.0)) * u_at_x * dx
        #F[k] = F[k - 1] - (1 / np.sqrt(2.0)) * u_at_x * dx

    return F


def plot_control(epoch, X, u_opt, a, mus, sigmas):
    v = get_ansatz_functions(X, mus, sigmas)
    u = np.dot(a, v)

    plt.plot(X, u, label='control')
    plt.plot(X, u_opt, label='optimal control')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=-5, top=5)
    plt.legend()

    file_name = 'control_epoch{}.png'.format(epoch)
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_free_energy(epoch, X, potential, F_opt, F):
    plt.plot(X, F, label='approx free energy')
    plt.plot(X, F_opt, label='ref solution free energy')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=0, top=5)
    plt.legend()

    file_name = 'free_energy_epoch{}.png'.format(epoch)
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_tilted_potential(epoch, X, potential, F_opt, F):
    V = potential(X)
    plt.plot(X, V + 2 * F, label='approx potential')
    plt.plot(X, V + 2 * F_opt, label='optimal tilted potential')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=0, top=15)
    plt.legend()

    file_name = 'approx_tilted_potential_epoch{}.png'.format(epoch)
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_tilted_potentials(epoch, Omega_h, u_opt, mus, sigmas, a):
    potential, gradient = get_potential_and_gradient('asym_2well')

    N = 1000
    X = np.linspace(-2, 2, N)
    dx = X[1] - X[0]

    # compute Free energy from the control
    F = np.zeros(N)
    for k in range(1, N):
        v = get_ansatz_functions(X[k], mus, sigmas)
        u_at_x = np.dot(a, v)
        F[k] = F[k - 1] - (1 / np.sqrt(2.0)) * u_at_x * dx

    # plot potential
    V = potential(X)
    plt.plot(X, V + 2 * F, label='approx potential')
    V = potential(Omega_h)
    plt.plot(Omega_h, V + 2 * F_opt, label='optimal tilted potential')
    plt.xlim(left=-2, right=2)
    plt.ylim(bottom=0, top=15)
    plt.legend()
    file_name = 'approx_tilted_potential_epoch{}.png'.format(epoch)
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_ansatz_functions(omega_h, mus, sigmas):
    X = omega_h
    v = get_ansatz_functions(X, mus, sigmas)
    m = v.shape[0]
    for j in range(m):
        mu = mus[j]
        sigma = sigmas[j]
        label = r'$v_{' + str(j+1) + '}(x;' + str(mu) + ',' + str(sigma) + ')$'
        plt.plot(X, v[j], label=label)
    plt.title(r'$v_{j}(x; \mu, \sigma)$')
    plt.xlabel('x', fontsize=16)
    plt.legend(loc='upper left', fontsize=8)

    file_name = 'ansatz_functions.png'
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

# not used
def sample_loss_not_vectorized(m, mus, sigmas, a):
    # set potential
    potential, gradient = get_potential_and_gradient('asym_2well')

    # sampling parameters
    beta = 4
    xzero = 1
    M = 100
    target_set_min = -1.1
    target_set_max = -1.0

    # Euler-Majurama
    dt = 0.001
    N = 100000

    # initialize statistics 
    fht = np.zeros(M)
    cost = np.zeros(M)
    gradJ = np.zeros((m, M))
    gradSh = np.zeros((m, M))

    for i in np.arange(M):
        Xtemp = xzero
        cost_temp = 0
        sum_grad_gh_temp = np.zeros(m)
        gradSh_temp = np.zeros(m)

        for n in np.arange(1, N+1):
            normal_dist_sample = np.random.normal(0, 1)

            # Brownian increment
            dB = np.sqrt(dt) * normal_dist_sample

            # control
            btemp = get_ansatz_functions(Xtemp, mus, sigmas)
            utemp = np.dot(a, btemp)

            # compute gradient
            tilted_gradient = gradient(Xtemp) - np.sqrt(2) * utemp

            # SDE iteration
            drift = - tilted_gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp += drift + diffusion

            # compute cost, ...
            cost_temp += 0.5 * (utemp ** 2) * dt
            sum_grad_gh_temp += dt * utemp * btemp
            gradSh_temp += normal_dist_sample * btemp

            # trajectories in the target set
            if (Xtemp > target_set_min and Xtemp < target_set_max):
                break

        # save trajectory observables
        fht[i] = n * dt
        cost[i] = cost_temp
        gradSh[:, i] = gradSh_temp * (- np.sqrt(dt * beta))
        gradJ[:, i] = sum_grad_gh_temp - (fht[i] + cost[i]) * gradSh[:, i]

    # compute averages
    mean_fht = np.mean(fht)
    mean_cost = np.mean(cost)
    mean_gradJ = np.mean(gradJ, axis=1)
    mean_gradSh = np.mean(gradSh, axis=1)

    return mean_fht, mean_cost, mean_gradJ, mean_gradSh

if __name__ == "__main__":
    main()
