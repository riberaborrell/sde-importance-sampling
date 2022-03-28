import functools
import os
import sys

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from sde_importance_sampling.functions import constant, \
                                              quadratic_one_well, \
                                              double_well, \
                                              double_well_gradient
from sde_importance_sampling.utils_path import get_data_dir


POTENTIAL_NAMES = [
    'nd_2well',
    'nd_2well_asym',
]

class LangevinSDE(object):
    '''
    class representing the overdamped langevin equation.

    Attributes
    ----------
    problem_name: str
        label determining if the problem is elliptic or parabolic
    potential_name: str
        type of potential for the o.l. equation
    d: int
        dimension of the problem
    alpha: array
        parameters of the potential
    beta: float
        inverse of the temperature
    domain: array
        domain of the problem
    is_domain_hypercube: bool
        flag telling if domain is an hypercube
    domain_lb: float
        lower bound of the i-th coordinate of the domain
    domain_ub: float
        upper bound of the i-th coordinate of the domain
    target_set: array
        target set of the problem
    potential: function
        potential function given an alpha fixed
    gradient: function
        gradient function given an alpha fixed
    f: function
        running cost of the work functional
    g: function
        terminal cost of the work functional
    nu: array
        parameters of the g function
    T: float
        deterministic final time
    h: float
        step size
    Nx: tuple
        number of indices per coordinate
    Nh: int
        number of nodes in the grid
    domain_h: array
        discretized domain
    domain_h_flat: array
        flat discretized domain

    Methods
    -------
    set_potential_and_gradient(alpha)

    set_work_path_functional(nu)

    set_settings_dir_path()

    get_nd_doublewell_local_minimums()

    compute_euclidian_distance_to_local_minimums(x)

    is_hypercube(subset)

    discretize_domain(h=None)

    discretize_domain_ith_coordinate(i=0, h=None)

    discretize_domain_i_and_j_th_coordinates(i=0, j=1, x_k=-1., h=None)

    sample_domain_uniformly(K, subset=None)

    sample_multivariate_normal(mean, cov, K)

    sample_domain_boundary_uniformly()

    sample_domain_boundary_uniformly_vec(K)

    sample_S_uniformly(K)

    get_flat_domain_h()

    get_time_index(t)

    get_index(x)

    get_index_using_argmin(x)

    get_index_vectorized(x)

    get_index_vectorized_using_argmin(x)

    get_x(idx)

    get_idx_target_set()

    get_hjb_solver(h=None)

    get_hjb_solver_det(h=0.01, dt=0.005)

    get_not_controlled_sampling(dt, K, seed=None)

    get_hjb_sampling(sol_hjb_dir_path, dt, K, seed=None)

    get_metadynamics_sampling(meta_type, weights_type, omega_0,
                              sigma_i, dt, k, K, seed=None)

    get_metadynamics_nn_sampling(dt, sigma_i, meta_type, k, K)

    get_flat_bias_sampling(dt, k_lim, K)

    write_setting(f)

    print_report(f)
    '''

    def __init__(self, problem_name, potential_name, d, alpha, beta,
                 domain=None, target_set=None, T=None, nu=None):
        ''' init method

        Parameters
        ----------
        problem_name: str
            label determining if the problem is elliptic or parabolic
        potential_name: str
            type of potential for the o.l. equation
        n: int
            dimension of the problem
        alpha: array
            parameters of the potential
        beta: float
            inverse of the temperature
        domain: array
            domain of the problem
        target_set: array
            target set of the problem
        T: float
            deterministic final time
        nu: array
            parameters of the g function

        '''
        # check problem name
        assert problem_name in ['langevin_det-t', 'langevin_stop-t'], ''
        self.problem_name = problem_name

        # check potential name
        assert type(potential_name) == str and potential_name in POTENTIAL_NAMES, ''
        self.potential_name = potential_name

        # check dimension
        assert type(d) == int, ''
        self.d = d

        # get potential and gradient functions
        self.set_potential_and_gradient(alpha)

        # check beta
        type(float) == float, ''
        self.beta = beta

        # check domain 
        self.set_domain(domain)

        # set functions f and g
        self.set_work_path_functional(nu)

        # check target set if we are in the stopping time case
        if target_set is not None and problem_name == 'langevin_stop-t':
            assert type(target_set) == np.ndarray, ''
            assert target_set.ndim == 2, ''
            assert target_set.shape == (d, 2), ''
        elif target_set is None and problem_name == 'langevin_stop-t':
            target_set = np.full((d, 2), [1, 3])
        self.target_set = target_set

        # check final time if we are in the deterministic time case
        if T is not None and problem_name == 'langevin_det-t':
            assert type(T) == float, ''
        elif T is None and problem_name == 'langevin_det-t':
            T = 1.
        self.T = T

        # dir_path
        self.set_settings_dir_path()

    def set_potential_and_gradient(self, alpha):
        ''' given alpha sets a potential and a gradient function

        Parameters
        ----------
        alpha: array
            parameters of the potential
        '''

        # set alpha
        type(alpha) == np.ndarray, ''
        assert alpha.ndim == 1, ''
        assert alpha.shape[0] == self.d, ''
        self.alpha = alpha

        # set potential and gradient
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

    def set_work_path_functional(self, nu=None):
        ''' set work path functional

        Parameters
        ----------
        nu: array, optional
            parameters of the quadratic one well g function
        '''
        # set f=1 and g=0, leading to W=fht
        if self.problem_name == 'langevin_stop-t':
            self.f = functools.partial(constant, a=1.)
            self.g = functools.partial(constant, a=0.)

        # set f=0 and g=quadratic one well, leading to W=g(X_T)
        elif self.problem_name == 'langevin_det-t':

            # set nu
            assert nu is not None, ''
            assert type(nu) == np.ndarray, ''
            assert nu.ndim == 1, ''
            assert nu.shape[0] == self.d, ''
            self.nu = nu

            # set f and g
            self.f = functools.partial(constant, a=0.)
            self.g = functools.partial(quadratic_one_well, nu=nu)

    def set_settings_dir_path(self):
        ''' set directory path for the chosen sde setting
        '''

        # data path
        data_path = get_data_dir()

        # get alpha string
        if self.potential_name == 'nd_2well':
            assert (self.alpha == self.alpha[0]).all(), ''
            alpha_str = 'alpha_i_{}'.format(float(self.alpha[0]))

        elif self.potential_name == 'nd_2well_asym':
            assert self.d > 1, ''
            assert self.alpha[0] != self.alpha[1], ''
            alpha_str = 'alpha_i_{}_j_{}'.format(float(self.alpha[0]), float(self.alpha[1]))

        # get absolute path of the directory for the chosen settings
        self.settings_dir_path = os.path.join(
            data_path,
            self.problem_name,
            self.potential_name,
            'd_{:d}'.format(self.d),
            alpha_str,
            'beta_{}'.format(float(self.beta)),
        )

        # create dir path if not exists
        if not os.path.isdir(self.settings_dir_path):
            os.makedirs(self.settings_dir_path)

    def get_nd_doublewell_local_minimums(self):
        ''' computes the minimums of the double well potential

        Returns
        -------
        array
            (2^n, n)-array with the 2^n local minimums of the double well potential

        '''
        # number of local minimums of the nd double well
        n_local_mins = 2 ** self.d

        # preallocate list for the minimums
        local_mins = []

        # as long as the list is not fulll
        while len(local_mins) < n_local_mins:

            # sample possible minima
            trial = np.random.randint(2, size=self.d).tolist()

            # add if it is not in the list
            if trial not in local_mins:
                local_mins.append(trial)

        # order list
        local_mins.sort()

        # convert into numpy array
        local_mins = np.array(local_mins, dtype=np.float32)

        # substitute 0 for -1
        local_mins[local_mins == 0] = -1.

        return local_mins

    def compute_euclidian_distance_to_local_minimums(self, x):
        ''' computes euclidian distance between x and the local minimums of the double well
            potential

        Parameters
        ----------
        x: array
            K points in the domain

        Returns
        -------
        array
            (2^n, n)-array with the 2^n local minimums of the double well potential
        '''
        assert type(x) == np.ndarray, ''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.d, ''

        # compute local minimums of the nd 2well potential
        mins = self.get_nd_doublewell_local_minimums()

        # minimums expanded
        mins_expand = np.expand_dims(mins, axis=0)

        # x expanded
        x_expand = np.expand_dims(x, axis=1)

        # compute euclidian distance between the expanded mins and x 
        eucl_dist = np.linalg.norm(x_expand - mins_expand, axis=2)

        return eucl_dist

    def is_hypercube(self, subset):
        ''' checks if the given subset (hyperrectangle) an hypercube is.

        Parameters
        ----------
        subset: array
            subset of the domain

        Returns
        -------
        bool
            flag telling us if the subset if an hyperrectangle
        '''
        # get subset dimension
        n = subset.shape[0]

        # 1d case
        if n == 1:
            return True

        # nd case
        lb = subset[0, 0]
        rb = subset[0, 1]
        for i in range(1, n):
            if lb != subset[i, 0] or rb != subset[i, 1]:
                return False
        return True

    def set_domain(self, domain):
        ''' set domain. Check if it is an hypercube.

        Parameters
        ----------
        domain: array
            domain of the o.l. equation

        '''
        # set default domain
        if domain is None:
            self.domain = np.full((self.d, 2), [-3, 3])
            self.domain_lb = -3.
            self.domain_ub = 3.
            self.is_domain_hypercube = True
            return

        # assert domain is an hyperrectangle
        assert type(domain) == np.ndarray, ''
        assert domain.ndim == 2, ''
        assert domain.shape == (self.d, 2), ''
        self.domain = domain

        # check if domain is an hypercube
        self.is_domain_hypercube = self.is_hypercube(self.domain)

        # add lower bound and upper bound
        if self.is_domain_hypercube:
            self.domain_lb = float(domain[0, 0])
            self.domain_ub = float(domain[0, 1])


    def discretize_domain(self, h=None):
        ''' this method discretizes the hyper-rectangular domain uniformly with step-size h

        Parameters
        ----------
        h: float, optional
            step size
        '''
        if h is not None:
            self.h = h
        assert self.h is not None, ''

        # construct not sparse nd grid
        try:
            mgrid_input = []
            for i in range(self.d):
                mgrid_input.append(
                    slice(self.domain[i, 0], self.domain[i, 1] + self.h, self.h)
                )
            self.domain_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)
        except MemoryError as e:
            print('MemoryError: {}'.format(e))
            sys.exit()

        # check shape
        assert self.domain_h.shape[-1] == self.d, ''

        # save number of indices per axis
        self.Nx = self.domain_h.shape[:-1]

        # save number of flattened indices
        N = 1
        for i in range(self.d):
            N *= self.Nx[i]
        self.Nh = N

    def discretize_domain_ith_coordinate(self, i=0, h=None):
        ''' this method discretizes the i-th domain coordinate with discretization step h

        Parameters
        ----------
        i: int, optional
            index of the chosen coordinate
        h: float, optional
            step size
        '''
        assert self.d > 1, ''
        assert i in range(self.d), ''

        if h is not None:
            self.h = h
        assert self.h is not None, ''

        self.domain_i_h = np.arange(self.domain[i, 0], self.domain[i, 1] + h, h)
        self.Nh = self.domain_i_h.shape[0]

    def discretize_domain_i_and_j_th_coordinates(self, i=0, j=1, x_k=-1., h=None):
        ''' this method discretizes the i-th and j-th domain coordinates with discretization step h

        Parameters
        ----------
        i: int, optional
            index of the chosen coordinate
        j: int, optional
            index of the chosen coordinate
        x_k: float, optional
            value of the other coordinates
        h: float, optional
            step size
        '''
        assert self.d > 2, ''
        assert i in range(self.d), ''
        assert j in range(self.d), ''
        assert i != j, ''

        if h is not None:
            self.h = h
        assert self.h is not None, ''

        # construct not sparse nd grid
        try:
            mgrid_input = [
                slice(self.domain[i, 0], self.domain[i, 1] + self.h, self.h),
                slice(self.domain[j, 0], self.domain[j, 1] + self.h, self.h),
            ]
            self.domain_ij_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)
        except MemoryError as e:
            print('MemoryError: {}'.format(e))
            sys.exit()

        # save number of indices per axis
        self.Nx = self.domain_ij_h.shape[:-1]

        # save number of flattened indices
        self.Nh = self.Nx[0] * self.Nx[1]

        # flat domain
        self.domain_h_flat = x_k * np.ones((self.Nh, self.d))
        self.domain_h_flat[:, i] = self.domain_ij_h.reshape(self.Nh, 2)[:, 0]
        self.domain_h_flat[:, j] = self.domain_ij_h.reshape(self.Nh, 2)[:, 1]


    def sample_domain_uniformly(self, K, subset=None):
        ''' samples points from a subset of the domain uniformly

        Parameters
        ----------
        K: int
            number of points to sample
        subset: array, optional
            subset of the domain

        Returns
        -------
        array
            K points
        '''

        # check if domain subset is given
        if subset is None:
            subset = self.domain

        # multivariate uniform sampling
        x = np.random.uniform(
            subset[:, 0],
            subset[:, 1],
            (K, self.d),
        )
        return x


    def sample_multivariate_normal(self, mean, cov, K):
        ''' samples points from a multivariate normal probability distribution function.

        Parameters
        ----------
        mean: (n,)-array
            center of the Gaussian function
        cov: (n, n)-array
            covariance matrix of the Gaussian function
        K: int
            number of points

        Returns
        -------
        array
            K points
        '''
        # tensorize mean and covariance matrix
        mean_tensor = torch.tensor(mean, requires_grad=False)
        cov_tensor = torch.tensor(cov, requires_grad=False)

        # define multivariate normal
        m = MultivariateNormal(mean_tensor, cov_tensor)

        # sample vectorized
        return m.sample((K,)).numpy()

    def sample_domain_boundary_uniformly(self):
        ''' samples a point from the boundary of the domain uniformly

        Returns
        -------
        array
            1 point
        '''
        x = np.empty(self.d)
        i = np.random.randint(self.d)
        for j in range(self.d):
            if j == i:
                k = np.random.randint(2)
                x[j] = self.domain[j, k]
            else:
                x[j] = np.random.uniform(
                    self.domain[j, 0],
                    self.domain[j, 1],
                    1
                )
        return x

    def sample_domain_boundary_uniformly_vec(self, K):
        ''' samples points from the boundary of the domain uniformly

        Parameters
        ----------
        K: int
            number of points to sample

        Returns
        -------
        array
            K points
        '''
        x = self.sample_domain_uniformly(K)
        i = np.random.randint(self.d, size=K)
        k = np.random.randint(2, size=K)
        for j in np.arange(K):
            x[j, i[j]] = self.domain[i[j], k[j]]
        return x

    #TODO: check method
    def sample_S_uniformly(self, K):
        x = np.empty((K, self.d))
        for j in range(K):
            is_in_target_set = True
            while is_in_target_set:
                x_j = self.sample_domain_uniformly(K=1)
                is_in_target_set = (
                    (x_j >= self.target_set[:, 0]) &
                    (x_j <= self.target_set[:, 1])
                ).all()
            x[j] = x_j
        return x

    def get_flat_domain_h(self):
        ''' this method returns the flattened discretized domain

        Returns
        -------
        array
            flat discretize domain
        '''
        return self.domain_h.reshape(self.Nh, self.d)

    def get_time_index(self, t):
        ''' returns time index for the given time t

        Parameters
        ----------
        t: float
            time

        Returns
        -------
        array
            flat discretize domain
        '''
        assert 0 <= t <= self.T, ''

        return int(np.ceil(t / self.dt))

    def get_index(self, x):
        ''' returns the index of the point of the grid closest to x. Assumes the domain is
            an hypercube.

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        array
            tuple with the indices of the point of the grid closest to x.
        '''
        assert x.ndim == 1, ''
        assert x.shape[0] == self.d, ''

        idx = np.floor((
            np.clip(x, self.domain_lb, self.domain_ub - 2 * self.h) + self.domain_ub
        ) / self.h).astype(int)

        return tuple(idx)

    def get_index_using_argmin(self, x):
        ''' returns the index of the point of the grid closest to x. It uses argmin method.

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        array
            tuple with the indices of the point of the grid closest to x.
        '''
        assert x.ndim == 1, ''
        assert x.shape[0] == self.d, ''

        # get index of xzero
        idx = [None for i in range(self.d)]
        for i in range(self.d):
            idx[i] = np.argmin(np.abs(self.domain_i_h - x[i]))

        return tuple(idx)

    def get_index_vectorized(self, x):
        ''' returns the index of the point of the grid closest to x. Vectorized for more than
            one point at a time. Assumes the domain is an hypercube.
        Parameters
        ----------
        x: array
            K points in the domain

        Returns
        -------
        array
            tuple with the indices of the points of the grid closest to x.
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.d, ''

        idx = np.floor((
            np.clip(x, self.domain_lb, self.domain_ub - 2 * self.h) + self.domain_ub
        ) / self.h).astype(int)

        idx = tuple([idx[:, i] for i in range(self.d)])

        return idx

    def get_index_vectorized_using_argmin(self, x):
        ''' returns the index of the point of the grid closest to x. Vectorized for more than
            one point at a time. Assumes the domain is an hypercube.
        Parameters
        ----------
        x: array
            K points in the domain

        Returns
        -------
        array
            tuple with the indices of the points of the grid closest to x.
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.d, ''

        #TODO! try to use directly the discretized domain

        # compute absolute value between each coordinate and the discretized ith grid
        dist = np.abs(self.domain_i_h[np.newaxis, np.newaxis] - x[:, :, np.newaxis])

        # get indeces of the minimum values
        idx = np.argmin(dist, axis=2)

        return tuple(idx)


    def get_x(self, idx):
        ''' returns the coordinates of the point determined by the axis indices idx

        Parameters
        ----------
        idx: tuple
            tuple with the bumpy indices

        Returns
        -------
        array
            point in the grid
        '''
        return self.domain_h[idx]

    def get_idx_target_set(self):
        ''' returns the index of the target set.

        Returns
        -------
        tuple
            indices of the target set
        '''
        assert self.domain_h is not None, ''

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # boolean array telling us if x is in the target set
        is_in_target_set = (
            (x >= self.target_set[:, 0]) &
            (x <= self.target_set[:, 1])
        ).all(axis=1).reshape(self.Nh, 1)

        # get index
        return np.where(is_in_target_set == True)[0]

    def get_hjb_solver(self, h=None) -> None:
        from sde_importance_sampling.hjb_solver import SolverHJB

        if h is None and self.d == 1:
            h = 0.001
        elif h is None and self.d == 2:
            h = 0.005
        elif h is None and self.d == 3:
            h = 0.1
        elif h is None:
            return

        # initialize hjb solver
        sol_hjb = SolverHJB(
            problem_name=self.problem_name,
            potential_name=self.potential_name,
            d=self.d,
            alpha=self.alpha,
            beta=self.beta,
            h=h,
        )

        # load already computed solution
        if sol_hjb.load():
            return sol_hjb

    def get_hjb_solver_det(self, h=0.01, dt=0.005) -> None:
        from sde_importance_sampling.hjb_solver_det import SolverHJBDet

        # initialize hjb solver
        sol_hjb = SolverHJBDet(
            problem_name=self.problem_name,
            potential_name=self.potential_name,
            d=self.d,
            alpha=self.alpha,
            beta=self.beta,
            T=self.T,
            nu=self.nu,
            h=h,
            dt=dt,
        )

        # load already computed solution
        if sol_hjb.load():
            return sol_hjb

    def get_not_controlled_sampling(self, dt, K, seed=None) -> None:
        from sde_importance_sampling.importance_sampling import Sampling

        # initialize not controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = False

        # set Euler-Marujama discretiztion step and number of trajectories
        sample.dt = dt
        sample.K = K
        sample.seed = seed

        # set path
        sample.set_not_controlled_dir_path()

        # load already sampled statistics
        if sample.load():
            return sample

    def get_hjb_sampling(self, sol_hjb_dir_path, dt, K, seed=None) -> None:
        from sde_importance_sampling.importance_sampling import Sampling

        # initialize not controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = True

        # set Euler-Marujama discretiztion step and number of trajectories
        sample.dt = dt
        sample.K = K
        sample.seed = seed

        # set path
        sample.set_controlled_dir_path(sol_hjb_dir_path)

        # load already sampled statistics
        if sample.load():
            return sample

    def get_metadynamics_sampling(self, meta_type, weights_type, omega_0,
                                  sigma_i, dt, k, K, seed=None):
        from sde_importance_sampling.importance_sampling import Sampling
        from sde_importance_sampling.metadynamics import Metadynamics

        # initialize controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = True
        sample.dt = dt

        # initialize meta nd object
        meta = Metadynamics(
            sample=sample,
            meta_type=meta_type,
            weights_type=weights_type,
            omega_0=omega_0,
            sigma_i=sigma_i,
            k=k,
            K=K,
            seed=seed,
        )

        # set path
        meta.set_dir_path()

        # load already sampled trajectories
        meta.load()
        return meta

    def get_metadynamics_nn_sampling(self, dt, sigma_i, meta_type, k, K):
        from mds.langevin_nd_importance_sampling import Sampling
        from mds.langevin_nd_metadynamics_nn import MetadynamicsNN

        # initialize controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = True
        sample.dt = dt

        # initialize meta nd object
        meta = MetadynamicsNN(
            sample=sample,
            k=k,
            K=K,
            sigma_i=sigma_i,
        )
        meta.meta_type = meta_type

        # set path
        meta.set_dir_path()

        # load already sampled trajectories
        meta.load()
        return meta

    def get_flat_bias_sampling(self, dt, k_lim, K):
        from mds.langevin_nd_importance_sampling import Sampling
        from mds.langevin_nd_flat_bias_potential import GetFlatBiasPotential

        # initialize sampling object
        sample = Sampling.new_from(self)
        sample.dt = dt
        sample.k_lim = k_lim
        sample.K = K

        # initialize flatbias object
        flatbias = GetFlatBiasPotential(sample)

        # set path
        flatbias.set_dir_path()

        # load already sampled trajectories
        flatbias.load()
        return flatbias

    def write_setting(self, f):
        ''' writes the setting parameters

        Parameters
        ----------
        f: file
            file where we write our settings report
        '''
        f.write('\nSDE Setting\n')
        f.write('potential: {}\n'.format(self.potential_name))
        f.write('alpha: {}\n'.format(self.alpha))
        f.write('beta: {:2.1f}\n'.format(self.beta))

        target_set = 'target set: ['
        for i in range(self.d):
            i_axis_str = '[{:2.1f}, {:2.1f}]'.format(self.target_set[i, 0], self.target_set[i, 1])
            if i == 0:
                target_set += i_axis_str
            else:
                target_set += ', ' + i_axis_str
        target_set += ']\n'
        f.write(target_set)

    def print_report(self, dir_path):
        try:
            with open(os.path.join(dir_path, 'report.txt'), 'r') as f:
                print(f.read())
        except:
            print('no report file found with path: {}'.format(dir_path))
