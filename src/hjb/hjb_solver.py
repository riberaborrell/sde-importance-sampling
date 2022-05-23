import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from sde.langevin_sde import LangevinSDE
from utils.paths import get_hjb_solution_dir_path, get_time_in_hms
from utils.numeric import arange_generator, from_1dndarray_to_string
from utils.figures import TITLES_FIG, COLORS_FIG


class SolverHJB(object):
    ''' This class provides a solver of the following BVP by using a
    finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
    where f = 1, g = 0 and L is the infinitessimal generator
    of the not controlled d-dimensional overdamped langevin process:
        L = - ∇V·∇ + 1/2 Δ

    Attributes
    ----------
    sde: langevinSDE object
        overdamped langevin sde object
    dir_path: str
        directory path for the hjb solver
    psi: array
        solution of the BVP problem
    solved: bool
        flag telling us if the problem is solved
    value_function: array
        value function of the HJB equation
    u_opt: array
        optimal control of the HJB equation
    bias_potential: array
        bias potential
    perturbed_potential: array
        perturbed potential
    perturbed_drift: array
        perturbed drift
    ct_initial: float
        initial computational time
    ct_time: float
        final computational time
    ct: float
        computational time


    Methods
    -------
    __init__(h)

    start_timer()

    stop_timer()

    get_flatten_index(idx)

    get_bumpy_index(k)

    is_on_domain_boundary(idx)

    is_on_domain_boundary_i_axis(idx, i)

    is_on_domain_corner(idx)

    is_on_ts(idx)

    get_flatten_idx_from_axis_neighbours(idx, i)

    get_flatten_idx_from_corner_neighbour(k)

    solve_bvp()

    compute_value_function()

    get_idx_value_f_type(i)

    get_idx_u_type(i)

    compute_optimal_control()

    save()

    load()

    get_psi_at_x(x)

    get_value_function_at_x(x)

    get_u_opt_at_x(x)

    get_perturbed_potential_and_drift()

    write_report(x)

    plot_1d(array_name, ylim=None, dir_path=None, file_name=None)

    plot_2d_psi(dir_path=None, file_name='psi')

    plot_2d_value_function(dir_path=None, file_name='value-function')

    plot_2d_perturbed_potential(dir_path=None, file_name='perturbed-potential')

    plot_2d_control(scale=None, width=0.005, dir_path=None, file_name='control')

    plot_2d_perturbed_drift(scale=None, width=0.005, file_name='perturbed-drift')
   '''

    def __init__(self, sde, h):
        ''' init method

        Parameters
        ----------
        sde: langevinSDE object
            overdamped langevin sde object
        h: float
            step size
        '''

        # sde object
        self.sde = sde

        # discretization step
        self.sde.h = h

        # dir_path
        self.dir_path = get_hjb_solution_dir_path(sde.settings_dir_path, h)

    def start_timer(self):
        ''' start timer
        '''
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        ''' stop timer
        '''
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def get_flatten_index(self, idx):
        ''' maps the bumpy index of the node (index of each axis) to
            the flatten index of the node, i.e. the node number.

        Parameters
        ----------
        idx: tuple
            bumpy index of the node

        Returns
        -------
        int
            flatten index of the node
        '''
        assert type(idx) == tuple, ''
        assert len(idx) == self.sde.d, ''

        k = 0
        for i in range(self.sde.d):
            assert 0 <= idx[i] <= self.sde.Nx[i] - 1, ''
            Nx_prod = 1
            for j in range(i+1, self.sde.d):
                Nx_prod *= self.sde.Nx[j]
            k += idx[i] * Nx_prod

        return k

    def get_bumpy_index(self, k):
        ''' maps the flatten index of the node (node number) to
            the bumpy index of the node.

        Parameters
        ----------
        k: int
            flatten index of the node

        Returns
        -------
        tuple
            bumpy index of the node
        '''
        assert type(k) == int, ''
        assert 0 <= k <= self.sde.Nh -1, ''

        idx = [None for i in range(self.sde.d)]
        for i in range(self.sde.d):
            Nx_prod = 1
            for j in range(i+1, self.sde.d):
                Nx_prod *= self.sde.Nx[j]
            idx[i] = k // Nx_prod
            k -= idx[i] * Nx_prod
        return tuple(idx)

    def is_on_domain_boundary(self, idx):
        ''' returns True if the idx is on the
            boundary of the domain

        Parameters
        ----------
        idx: tuple
            bumpy index of the node

        Returns
        -------
        bool
            True if the node is on the boundary of the domain
        '''
        for i in range(self.sde.d):
            if (idx[i] == 0 or
                idx[i] == self.sde.Nx[i] - 1):
                return True
        return False

    def is_on_domain_boundary_i_axis(self, idx, i):
        ''' returns True if the idx is on the
            i axis boundary of the domain

        Parameters
        ----------
        idx: tuple
            bumpy index of the node
        i: int
            index of the ith coordinate

        Returns
        -------
        bool
            True if the node is on the boundary of the i-th coordinate of the domain
        '''
        if (idx[i] == 0 or
            idx[i] == self.sde.Nx[i] - 1):
            return True
        else:
            return False

    def is_on_domain_corner(self, idx):
        ''' returns True if the idx is on the corner of the rectangular boundary

        Parameters
        ----------
        idx: tuple
            bumpy index of the node

        Returns
        -------
        bool
            True if the node is on the domain corner
        '''
        for i in range(self.sde.d):
            if (idx[i] != 0 and
                idx[i] != self.sde.Nx[i] - 1):
                return False
        return True

    def is_on_ts(self, idx):
        ''' returns True if the idx is on the target set

        Parameters
        ----------
        idx: tuple
            bumpy index of the node

        Returns
        -------
        bool
            True if the node is on the target set
        '''
        x = self.sde.get_x(idx)
        for i in range(self.sde.d):
            if (x[i] < self.sde.target_set[i, 0] or
                x[i] > self.sde.target_set[i, 1]):
                return False
        return True

    def get_flatten_idx_from_axis_neighbours(self, idx, i):
        ''' get flatten idx of the neighbours with respect to the i-th coordinate

        Parameters
        ----------
        idx: tuple
            bumpy index of the node
        i: int
            index of the ith coordinate

        Returns
        -------
        tuple
            (k_left, k_right)
        '''

        # find flatten index of left neighbour wrt the i axis
        if idx[i] == 0:
            k_left = None
        else:
            idx_left = list(idx)
            idx_left[i] = idx[i] - 1
            k_left = self.get_flatten_index(tuple(idx_left))

        # find flatten index of right neighbour wrt the i axis
        if idx[i] == self.sde.Nx[i] - 1:
            k_right = None
        else:
            idx_right = list(idx)
            idx_right[i] = idx[i] + 1
            k_right = self.get_flatten_index(tuple(idx_right))

        return (k_left, k_right)

    def get_flatten_idx_from_corner_neighbour(self, k):
        ''' get flatten idx of the corner neighbours

        Parameters
        ----------
        k: int
            flat index

        Returns
        -------
        tuple
            flatten idx of the the corner neighbours
        '''
        idx = self.get_bumpy_index(k)

        idx_inside = [None for i in range(self.sde.d)]
        for i in range(self.sde.d):
            if idx[i] == 0:
                idx_inside[i] = idx[i] + 1
            elif idx[i] == self.sde.Nx[i] - 1 :
                idx_inside[i] = idx[i] - 1
        return self.get_flatten_index(tuple(idx_inside))

    def solve_bvp(self):
        ''' solves the bvp using finite difference
        '''

        # sde equation
        d = self.sde.d
        beta = self.sde.beta
        sigma = self.sde.sigma
        f = self.sde.f
        g = self.sde.g
        gradient = self.sde.gradient

        # discretization
        h = self.sde.h
        Nh = self.sde.Nh

        # assemble linear system of equations: A \psi = b.
        A = sparse.lil_matrix((Nh, Nh))
        b = np.zeros(Nh)

        for k in arange_generator(Nh):

            # get discretized domain index
            idx = self.get_bumpy_index(k)

            # classify type of node
            is_on_ts = self.is_on_ts(idx)
            is_on_boundary = self.is_on_domain_boundary(idx)

            # assemble matrix A and vector b on S
            if not is_on_ts and not is_on_boundary:
                x = self.sde.get_x(idx)
                grad_at_x = gradient(np.array([x]))[0]
                A[k, k] = - (sigma**2 * d) / h**2 - f(x)
                for i in range(d):
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i)
                    A[k, k_left] = sigma**2 / (2 * h**2) + grad_at_x[i] / (2 * h)
                    A[k, k_right] = sigma**2 / (2 * h**2) - grad_at_x[i] / (2 * h)

            # impose condition on ∂S
            elif is_on_ts and not is_on_boundary:
                A[k, k] = 1
                b[k] = np.exp(- g(x))

            # stability condition on the boundary
            elif is_on_boundary:
                neighbour_counter = 0
                for i in range(d):
                    if self.is_on_domain_boundary_i_axis(idx, i):

                        # update counter
                        neighbour_counter += 1

                        # add neighbour
                        k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i)
                        if k_left is not None:
                            A[k, k_left] = - 1
                        elif k_right is not None:
                            A[k, k_right] = - 1

                # normalize
                A[k, k] = neighbour_counter

        psi = linalg.spsolve(A.tocsc(), b)
        self.psi = psi.reshape(self.sde.Nx)
        self.solved = True

    def compute_value_function(self):
        ''' this methos computes the value function
                value_f = - log (psi)
        '''
        assert hasattr(self, 'psi'), ''
        assert self.psi.ndim == self.sde.d, ''
        assert self.psi.shape == self.sde.Nx, ''

        self.value_function =  - np.log(self.psi)

    #TODO! comment this method
    def get_idx_value_f_type(self):
        '''

        Returns
        -------
        list

        '''
        return [slice(self.sde.Nx[i]) for i in range(self.sde.d)]

    #TODO! comment this method
    def get_idx_u_type(self, i):
        '''

        Parameters
        ----------
        i: int
            index of the i-th coordinate

        Returns
        -------
        list

        '''
        idx_u = [None for j in range(self.sde.d + 1)]
        idx_u[-1] = i
        for j in range(self.sde.d):
            idx_u[j] = slice(self.sde.Nx[j])
        return idx_u

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control vector field
                u_opt = - sigma ∇ value_f
        '''
        assert hasattr(self, 'value_function'), ''
        assert self.value_function.ndim == self.sde.d, ''
        assert self.value_function.shape == self.sde.Nx, ''

        # initialize control
        u_opt = np.zeros(self.sde.Nx + (self.sde.d, ))

        for i in range(self.sde.d):

            # compute indices
            idx_value_f_k_plus = self.get_idx_value_f_type()
            idx_value_f_k_minus = self.get_idx_value_f_type()
            idx_u_k = self.get_idx_u_type(i)
            idx_u_0 = self.get_idx_u_type(i)
            idx_u_1 = self.get_idx_u_type(i)
            idx_u_N_minus = self.get_idx_u_type(i)
            idx_u_N = self.get_idx_u_type(i)

            for j in range(self.sde.d):
                if j == i:
                    idx_value_f_k_plus[j] = slice(2, self.sde.Nx[j])
                    idx_value_f_k_minus[j] = slice(0, self.sde.Nx[j] - 2)
                    idx_u_k[j] = slice(1, self.sde.Nx[j] - 1)
                    idx_u_0[j] = 0
                    idx_u_1[j] = 1
                    idx_u_N_minus[j] = self.sde.Nx[j] - 2
                    idx_u_N[j] = self.sde.Nx[j] - 1
                    break

            # generalized central difference
            u_opt[tuple(idx_u_k)] = - self.sde.sigma *(
                self.value_function[tuple(idx_value_f_k_plus)] - self.value_function[tuple(idx_value_f_k_minus)]
            ) / (2 * self.sde.h)
            u_opt[tuple(idx_u_0)] = u_opt[tuple(idx_u_1)]
            u_opt[tuple(idx_u_N)] = u_opt[tuple(idx_u_N_minus)]

        self.u_opt = u_opt

    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save arrays in a npz file
        np.savez(
            os.path.join(self.dir_path, 'hjb-solution.npz'),
            h=self.sde.h,
            domain_h=self.sde.domain_h,
            Nx=self.sde.Nx,
            Nh=self.sde.Nh,
            psi=self.psi,
            value_function=self.value_function,
            u_opt=self.u_opt,
            ct=self.ct,
        )

    def load(self):
        ''' loads the saved arrays and sets them as attributes back
        '''
        try:
            data = np.load(
                os.path.join(self.dir_path, 'hjb-solution.npz'),
                allow_pickle=True,
            )
            for attr_name in data.files:

                # get attribute from data
                if data[attr_name].ndim == 0:
                    attr = data[attr_name][()]
                else:
                    attr = data[attr_name]

                # langevin SDE attribute
                if attr_name in ['h', 'domain_h', 'Nx', 'Nh']:

                    # save attribute
                    setattr(self.sde, attr_name, attr)

                # hjb solver attribute
                else:

                    # save attribute
                    setattr(self, attr_name, attr)

            return True

        except:
            print('no hjb-solution found with h={:.0e}'.format(self.sde.h))
            return False

    def get_psi_at_x(self, x):
        ''' evaluates solution of the BVP at x

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        float
            psi at x
        '''
        # get index of x
        idx = self.sde.get_index(x)

        # evaluate psi at idx
        return self.psi[idx] if hasattr(self, 'psi') else None

    def get_value_function_at_x(self, x):
        ''' evaluates the value function at x

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        float
            value function at x
        '''
        # get index of x
        idx = self.sde.get_index(x)

        # evaluate psi at idx
        return self.value_function[idx] if hasattr(self, 'value_function') else None

    def get_u_opt_at_x(self, x):
        ''' evaluates the optimal control at x

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        array
            optimal control at x
        '''
        # get index of x
        idx = self.sde.get_index(x)

        # evaluate psi at idx
        return self.u_opt[idx] if hasattr(self, 'u_opt') else None

    def get_perturbed_potential_and_drift(self):
        ''' computes the potential, bias potential, perturbed potential, gradient,
            controlled drift
        '''

        # flatten domain_h
        x = self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)

        # potential, bias potential and tilted potential
        V = self.sde.potential(x).reshape(self.sde.Nx)
        self.bias_potential = self.value_function * self.sde.sigma**2
        self.perturbed_potential = V + self.bias_potential

        # gradient and tilted drift
        dV = self.sde.gradient(x).reshape(self.sde.domain_h.shape)
        self.perturbed_drift = - dV + self.sigma * self.u_opt

    def write_report(self, x):
        ''' writes the hjb solver parameters

        Parameters
        ----------
        x: array
            point in the domain
        '''

        # set path
        file_path = os.path.join(self.dir_path, 'report.txt')

        # write file
        f = open(file_path, 'w')

        # space discretization
        f.write('\n space discretization\n')
        f.write('h = {:2.4f}\n'.format(self.sde.h))
        f.write('N_h = {:d}\n'.format(self.sde.Nh))

        # psi, value function and control at x
        f.write('\n psi, value function and optimal control at x\n')

        x_str = 'x: ' + from_1dndarray_to_string(x)
        psi = self.get_psi_at_x(x)
        value_f = self.get_value_function_at_x(x)
        u_opt = self.get_u_opt_at_x(x)
        u_opt_str = 'u_opt(x): ' + from_1dndarray_to_string(u_opt)

        f.write(x_str)
        if psi is not None:
            f.write('psi(x) = {:2.4e}\n'.format(psi))

        if value_f is not None:
            f.write('value_f(x) = {:2.4e}\n'.format(value_f))

        f.write(u_opt_str)

        # maximum value of the control
        if self.sde.d == 1:

            f.write('\n maximum value of the optimal control\n')

            idx_u_max = np.argmax(self.u_opt)
            x_u_max = self.sde.get_x(idx_u_max)
            u_opt_max = self.u_opt[idx_u_max]
            x_u_max_str = 'x: ' + from_1dndarray_to_string(x_u_max)
            u_opt_max_str = 'max u_opt(x): = ' + from_1dndarray_to_string(u_opt_max)

            f.write(x_u_max_str)
            f.write(u_opt_max_str)

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        f.write('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))
        f.close()

        # print file
        f = open(file_path, 'r')
        print(f.read())
        f.close()

    def plot_1d(self, array_name: str, ylim=None, dir_path=None, file_name=None):
        ''' plot chosen array

        Parameters
        ----------
        array_name: str
            name of the array to plot
        ylim: tuple
            limits for the y coordinate
        dir_path: str, optional
            directory path for the figure
        file_name: str, optional
            file name
        '''
        from figures.myfigure import MyFigure

        # check if attribute is computed
        assert array_name in [
            'psi',
            'value_function',
            'perturbed_potential',
            'u_opt',
            'perturbed_drift',
        ], ''
        if not hasattr(self, array_name):
            self.get_perturbed_potential_and_drift()

        # get array
        array = getattr(self, array_name)
        if array_name in ['u_opt', 'perturbed_drift']:
            array = array[:, 0]

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        # set file name
        if file_name is None:
            file_name = array_name

        # discretized domain
        x = self.sde.domain_h[:, 0]

        # initialize figure
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )
        fig.set_xlabel('x')
        fig.set_xlim(-2, 2)
        if ylim is not None:
            fig.set_ylim(ylim[0], ylim[1])

        fig.plot(x, array, labels='Reference solution', colors='tab:grey', linestyles=':')


    def plot_2d_psi(self, dir_path=None, file_name='psi'):
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        # contour plot
        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )
        fig.ax.set_title(TITLES_FIG['psi'])
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_colormap('Blues_r', start=0.10, stop=1.)
        #fig.set_contour_levels_scale('log')
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12)
        fig.contour(X, Y, self.psi)


    def plot_2d_value_function(self, dir_path=None, file_name='value-function'):
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        # contour plot
        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )
        fig.ax.set_title(TITLES_FIG['value-function'])
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_contour_levels_scale('log')
        fig.set_colormap('Blues_r', start=0.10, stop=1.)
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12)
        fig.contour(X, Y, self.value_function)


    def plot_2d_perturbed_potential(self, dir_path=None, file_name='perturbed-potential'):
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        # contour plot
        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )
        fig.ax.set_title(TITLES_FIG['optimal-potential'])
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-2, 2)
        fig.set_ylim(-2, 2)
        fig.set_contour_levels_scale('log')
        fig.set_colormap('Blues_r', start=0.10, stop=1.)
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12)
        fig.contour(X, Y, self.perturbed_potential)


    def plot_2d_control(self, scale=None, width=0.005, dir_path=None, file_name='control'):
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        U = self.u_opt[:, :, 0]
        V = self.u_opt[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )
        fig.ax.set_title(TITLES_FIG['optimal-control'])
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12)
        fig.vector_field(X, Y, U, V, scale=scale, width=width)


    def plot_2d_perturbed_drift(self, scale=None, width=0.005,
                                 dir_path=None, file_name='perturbed-drift'):
        from figures.myfigure import MyFigure

        # set dir path
        if dir_path is None:
            dir_path = self.dir_path

        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        U = self.perturbed_drift[:, :, 0]
        V = self.perturbed_drift[:, :, 1]
        fig = plt.figure(
            FigureClass=MyFigure,
            dir_path=dir_path,
            file_name=file_name,
        )
        fig.ax.set_title(r'$\nabla \widetilde{V}_h$')
        fig.set_xlabel(r'$x_1$')
        fig.set_ylabel(r'$x_2$')
        fig.set_xlim(-1.5, 1.5)
        fig.set_ylim(-1.5, 1.5)
        plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12)
        fig.vector_field(X, Y, U, V, scale=scale, width=width)
