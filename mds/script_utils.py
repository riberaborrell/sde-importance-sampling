from potentials_and_gradients import derivative_normal_pdf
from utils import get_data_path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import os

MDS_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MDS_PATH, 'data')
GD_FIGURES_PATH = os.path.join(MDS_PATH, 'figures/gradient_descent_greedy')

def get_reference_solution(data_path):
    ref_sol = np.load(os.path.join(data_path, 'reference_solution.npz'))
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

def set_unif_dist_ansatz_functions2(mus_min, mus_max, target_set, m):
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

    factor = 2
    sigma_left = (target_set_min - mus_min) / (factor + m_left -1)
    sigma_right = (mus_max - target_set_max) / (factor + m_right -1)

    # distribute ansatz functions unif (in each side)
    mus_left = np.around(
        np.linspace(mus_min, mus_min + (m_left -1) * sigma_left, m_left),
        decimals=3,
    )
    mus_right = np.around(
        np.linspace(mus_max - (m_right -1) * sigma_right, mus_max, m_right),
        decimals=3,
    )
    mus = np.concatenate((mus_left, mus_right), axis=0)

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

def get_meta_coefficients(potential_name, beta, target_set, X, mus, sigmas):
    # load metadynamics parameters
    dir_path = get_data_path(potential_name, beta, target_set)
    bias_pot = np.load(
        os.path.join(dir_path, 'metadynamics_bias_potential.npz')
    )
    omegas = bias_pot['omegas']
    meta_mus = bias_pot['mus']
    meta_sigmas = bias_pot['sigmas']

    # define a coefficients 
    m = mus.shape[0]
    a = np.zeros(m)

    # ansatz functions evaluated at the grid
    v = get_ansatz_functions(X, mus, sigmas)

    # control evaluated at the grid u(x) = -1 / sqrt(2) dVb(x)
    basis = get_derivative_ansatz_functions(X, meta_mus, meta_sigmas)
    dVb = np.dot(omegas, basis)
    u = - dVb / np.sqrt(2)

    # solve v a = u
    #a = np.linalg.solve(v, u)
    x, residuals, rank, s = np.linalg.lstsq(a=v.T, b=u, rcond=None)
    print(x)
    return x


def set_initial_coefficients(m, sigma_avg, a_opt, a_meta, a_init):
    pert_factor = sigma_avg

    if a_init == 'null':
        a = np.zeros(m)
    elif a_init == 'null-pert':
        a = pert_factor * np.ones(m)
        #a = pert_factor * np.random.choice([-1, 1], m)
    elif a_init == 'optimal':
        a = a_opt
    elif a_init == 'opt-pert':
        a = a_opt + pert_factor * np.ones(m)
        #a = a_opt * np.random.choice([-1, 1], m)
    elif a_init == 'meta':
        a = a_meta

    return a

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

def plot_control(dir_path, epoch, X, u_opt, u):
    plt.plot(X, u, label='control')
    plt.plot(X, u_opt, label='optimal control')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=-5, top=5)
    plt.legend()

    file_name = 'control_epoch{}.png'.format(epoch)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_free_energy(dir_path, epoch, X, potential, F_opt, F):
    plt.plot(X, F, label='approx free energy')
    plt.plot(X, F_opt, label='ref solution free energy')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=0, top=5)
    plt.legend()

    file_name = 'free_energy_epoch{}.png'.format(epoch)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_tilted_potential(dir_path, epoch, X, potential, F_opt, F):
    V = potential(X)
    plt.plot(X, V + 2 * F, label='approx potential')
    plt.plot(X, V + 2 * F_opt, label='optimal tilted potential')
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=0, top=15)
    plt.legend()

    file_name = 'approx_tilted_potential_epoch{}.png'.format(epoch)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_gd_tilted_potentials(dir_path, epochs, X, potential, F_opt, F):
    V = potential(X)
    plt.plot(X, V + 2 * F_opt, linestyle='dashed', label='optimal')
    for epoch in range(epochs + 1):
        label = r'epoch = {:d}'.format(epoch)
        plt.plot(X, V + 2 * F[epoch], linestyle='-', label=label)
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=0, top=15)
    plt.legend()

    file_name = 'approx_tilted_potentials_gd.png'
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_ansatz_functions(dir_path, X, mus, sigmas):
    v = get_ansatz_functions(X, mus, sigmas)
    m = v.shape[0]
    for j in range(m):
        plt.plot(X, v[j])
    plt.title(r'$v_{j}(x; \mu, \sigma)$')
    plt.xlabel('x', fontsize=16)

    file_name = 'ansatz_functions.png'
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

def plot_basis_functions(dir_path, X, mus, sigmas):
    b = - np.sqrt(2) * get_derivative_ansatz_functions(X, mus, sigmas)
    m = b.shape[0]
    for j in range(m):
        plt.plot(X, b[j])
    plt.title(r'$b_{j}(x; \mu, \sigma)$')
    plt.xlabel('x', fontsize=16)

    file_name = 'basis_functions.png'
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()
