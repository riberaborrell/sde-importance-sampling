from potentials_and_gradients import derivative_normal_pdf

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
GD_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/gradient_descent_greedy')

def get_reference_solution():
    ref_sol = np.load(
        os.path.join(DATA_PATH, 'langevin1d_reference_solution.npz')
    )
    omega_h = ref_sol['omega_h']
    F_opt = ref_sol['F']
    u_opt = ref_sol['u_opt']

    return omega_h, F_opt, u_opt


def get_ansatz_functions(x, mus, sigmas):
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)
    return stats.norm.pdf(x, mus, sigmas)


def get_derivative_ansatz_functions(x, mus, sigmas):
    if type(x) == np.ndarray:
        mus = mus.reshape(mus.shape[0], 1)
        sigmas = sigmas.reshape(sigmas.shape[0], 1)
    return derivative_normal_pdf(x, mus, sigmas)


def set_unif_dist_ansatz_functions(mus_min, mus_max, target_set, m):
    # assume target_set is connected and contained in [mus_min, mus_max]
    target_set_min, target_set_max = target_set

    # set grid 
    h = 0.001
    N = int((mus_max - mus_min) / h) + 1
    X = np.around(np.linspace(mus_min, mus_max, N), decimals=3)

    # get indexes for nodes in/out the target set
    idx_ts = np.where((X >= target_set_min) & (X <= target_set_max))[0]
    idx_nts = np.where((X < target_set_min) | (X > target_set_max))[0]
    idx_l = np.where(X < target_set_min)[0]
    idx_r = np.where(X > target_set_max)[0]

    # compute ratio of nodes in the left/right side of the target set
    ratio_left = idx_l.shape[0] / idx_nts.shape[0]
    ratio_right = idx_r.shape[0] / idx_nts.shape[0]

    # assigm number of ansatz functions in each side
    m_left = int(np.round(m * ratio_left))
    m_right = int(np.round(m * ratio_right))
    assert m == m_left + m_right

    # distribute ansatz functions unif (in each side)
    mus_left = np.around(
        np.linspace(X[idx_l][0], X[idx_l][-1], m_left + 2)[:-2],
        decimals=3,
    )
    mus_right = np.around(
        np.linspace(X[idx_r][0], X[idx_r][-1], m_right + 2)[2:],
        decimals=3,
    )
    mus = np.concatenate((mus_left, mus_right), axis=0)

    # compute sigmas
    factor = 2
    sigma_left = factor * np.around(mus_left[1] - mus_left[0], decimals=3)
    sigma_right = factor * np.around(mus_right[1] - mus_right[0], decimals=3)
    sigmas_left = sigma_left * np.ones(m_left)
    sigmas_right = sigma_right * np.ones(m_right)
    sigmas = np.concatenate((sigmas_left, sigmas_right), axis=0)
    sigma_avg = np.around(np.mean(sigmas), decimals=3)

    print(m_left, m_right, m)
    print(mus_left[0], mus_left[-1])
    print(mus_right[0], mus_right[-1])
    print(sigma_left, sigma_right, sigma_avg)

    return mus, sigmas

def get_optimal_coefficients(X, target_set, u_opt, mus, sigmas):
    # ansatz functions at the grid
    v = get_ansatz_functions(X, mus, sigmas)
    a = v.T

    # optimal control (from ref solution) at the grid
    b = u_opt

    # solve lin system of equations by using least squares
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
    #res = lsq_linear(a, b, bounds=(0, 'inf'), lsmr_tol='auto', verbose=1)
    #return res.x
    print(x)
    return x

def get_optimal_coefficients2(X, target_set, F_opt, mus, sigmas):
    v = get_ansatz_functions(X, mus, sigmas)
    a = v.T
    b = F_opt
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
    print(x)
    return x

def set_initial_coefficients(m, sigma_avg, a_opt, a_init_type):
    pert_factor = sigma_avg

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

def free_energy_on_grid(X, target_set, a, mus, sigmas):
    # discretization step and number of grid points
    dx = X[1] - X[0]
    N = len(X)

    # initialize free energy array
    F = np.zeros(N)

    # get indices where grid in the left / right of the target set
    target_set_min, target_set_max = target_set
    idx_l = np.where(X < target_set_min)[0]
    idx_r = np.where(X > target_set_max)[0]

    # compute free energy in the left of the target set 
    for k in np.flip(idx_l):
        v = get_ansatz_functions(X[k], mus, sigmas)
        u = np.dot(a, v)
        F[k - 1] = F[k] + (1 / np.sqrt(2.0)) * u * dx

    # compute free energy in the right of the target set
    for k in idx_r:
        v = get_ansatz_functions(X[k], mus, sigmas)
        u = np.dot(a, v)
        F[k] = F[k - 1] - (1 / np.sqrt(2.0)) * u * dx

    return F

def free_energy_on_grid2(X, a, mus, sigmas):
    v = get_ansatz_functions(X, mus, sigmas)
    F = np.dot(a, v)
    return F

def control_on_grid(X, a, mus, sigmas):
    v = get_ansatz_functions(X, mus, sigmas)
    u = np.dot(a, v)
    return u

def control_on_grid2(X, a, mus, sigmas):
    b = - np.sqrt(2) * get_derivative_ansatz_functions(X, mus, sigmas)
    u = np.dot(a, b)
    return u

def plot_control(epoch, X, u_opt, u):
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

def plot_gd_tilted_potentials(epochs, X, potential, F_opt, F):
    V = potential(X)
    plt.plot(X, V + 2 * F_opt, linestyle='dashed', label='optimal')
    for epoch in range(epochs + 1):
        label = r'epoch = {:d}'.format(epoch)
        plt.plot(X, V + 2 * F[epoch], linestyle='-', label=label)
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=0, top=15)
    plt.legend()

    file_name = 'approx_tilted_potentials_gd.png'
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_ansatz_functions(X, mus, sigmas):
    v = get_ansatz_functions(X, mus, sigmas)
    m = v.shape[0]
    for j in range(m):
        plt.plot(X, v[j])
    plt.title(r'$v_{j}(x; \mu, \sigma)$')
    plt.xlabel('x', fontsize=16)

    file_name = 'ansatz_functions.png'
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_basis_functions(X, mus, sigmas):
    b = - np.sqrt(2) * get_derivative_ansatz_functions(X, mus, sigmas)
    m = b.shape[0]
    for j in range(m):
        plt.plot(X, b[j])
    plt.title(r'$b_{j}(x; \mu, \sigma)$')
    plt.xlabel('x', fontsize=16)

    file_name = 'basis_functions.png'
    file_path = os.path.join(GD_FIGURES_PATH, file_name)
    plt.savefig(file_path)
    plt.close()
