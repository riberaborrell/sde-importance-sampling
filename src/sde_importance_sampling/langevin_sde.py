from sde_importance_sampling.functions import constant, \
                                              quadratic_one_well, \
                                              double_well, \
                                              double_well_gradient
from sde_importance_sampling.utils_path import get_data_dir

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import functools
import os
import sys

POTENTIAL_NAMES = [
    'nd_2well',
    'nd_2well_asym',
]

class LangevinSDE(object):
    '''
    '''

    def __init__(self, problem_name, potential_name, n, alpha, beta,
                 domain=None, target_set=None, T=None, nu=None):
        '''
        '''
        # check problem name
        assert problem_name in ['langevin_det-t', 'langevin_stop-t'], ''
        self.problem_name = problem_name

        # check potential name
        assert type(potential_name) == str and potential_name in POTENTIAL_NAMES, ''
        self.potential_name = potential_name

        # check dimension
        assert type(n) == int, ''
        self.n = n

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
            assert target_set.shape == (n, 2), ''
        elif target_set is None and problem_name == 'langevin_stop-t':
            target_set = np.full((n, 2), [1, 3])
        self.target_set = target_set

        # check final time if we are in the deterministic time case
        if T is not None and problem_name == 'langevin_det-t':
            assert type(T) == float, ''
        elif T is None and problem_name == 'langevin_det-t':
            T = 1.
        self.T = T

        # dir_path
        self.set_settings_dir_path()

    @classmethod
    def new_from(cls, obj):
        if issubclass(obj.__class__, LangevinSDE):
            _new = cls(obj.problem_name, obj.potential_name, obj.n, obj.alpha,
                       obj.beta, obj.domain, obj.target_set)
            return _new
        else:
            msg = 'Expected subclass of <class LangevinSDE>, got {}.'.format(type(obj))
            raise TypeError(msg)

    def set_potential_and_gradient(self, alpha):
        '''
        '''
        # set alpha
        type(alpha) == np.ndarray, ''
        assert alpha.ndim == 1, ''
        assert alpha.shape[0] == self.n, ''
        self.alpha = alpha

        # set potential and gradient
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

    def set_work_path_functional(self, nu):
        '''
        '''
        # set f=1 and g=0, leading to W=fht
        if self.problem_name == 'langevin_stop-t':
            self.f = functools.partial(constant, a=1.)
            self.g = functools.partial(constant, a=0.)

        # set f=0 and g=quadratic one well, leading to W=g(X_T)
        elif self.problem_name == 'langevin_det-t':

            # set nu
            type(nu) == np.ndarray, ''
            assert nu.ndim == 1, ''
            assert nu.shape[0] == self.n, ''
            self.nu = nu

            # set f and g
            self.f = functools.partial(constant, a=0.)
            self.g = functools.partial(quadratic_one_well, nu=nu)

    def set_settings_dir_path(self):

        # data path
        data_path = get_data_dir()

        # get alpha string
        if self.potential_name == 'nd_2well':
            assert (self.alpha == self.alpha[0]).all(), ''
            alpha_str = 'alpha_i_{}'.format(float(self.alpha[0]))

        elif self.potential_name == 'nd_2well_asym':
            assert self.n > 1, ''
            assert self.alpha[0] != self.alpha[1], ''
            alpha_str = 'alpha_i_{}_j_{}'.format(float(self.alpha[0]), float(self.alpha[1]))

        # get absolute path of the directory for the chosen settings
        self.settings_dir_path = os.path.join(
            data_path,
            self.problem_name,
            self.potential_name,
            'n_{:d}'.format(self.n),
            alpha_str,
            'beta_{}'.format(float(self.beta)),
        )

        # create dir path if not exists
        if not os.path.isdir(self.settings_dir_path):
            os.makedirs(self.settings_dir_path)

    def get_nd_doublewell_local_minimums(self):
        ''' returns an (2^n, n)-array with the 2^n local minimums of the double well potential
        '''
        # number of local minimums of the nd double well
        n_local_mins = 2 ** self.n

        # preallocate list for the minimums
        local_mins = []

        # as long as the list is not fulll
        while len(local_mins) < n_local_mins:

            # sample possible minima
            trial = np.random.randint(2, size=self.n).tolist()

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
        '''
        '''
        assert type(x) == np.ndarray, ''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        # compute local minimums of the nd 2well potential
        mins = self.get_nd_doublewell_local_minimums()

        # minimums expanded
        mins_expand = np.expand_dims(mins, axis=0)

        # x expanded
        x_expand = np.expand_dims(x, axis=1)

        # compute euclidian distance between the expanded mins and x 
        eucl_dist = np.linalg.norm(x_expand - mins_expand, axis=2)

        return eucl_dist


    def set_domain(self, domain):
        ''' set domain. check if it is an hypercube
        '''
        # set default domain
        if domain is None:
            self.domain = np.full((self.n, 2), [-3, 3])
            self.is_domain_hypercube = True
            return

        # assert domain is an hyperrectangle
        assert type(domain) == np.ndarray, ''
        assert domain.ndim == 2, ''
        assert domain.shape == (self.n, 2), ''
        self.domain = domain

        # check if domain is a rectangle
        # assume it is
        self.is_domain_hypercube = True

        # 1d case
        if self.n == 1:
            return

        # nd case
        lb = domain[0, 0]
        rb = domain[0, 1]
        for i in range(1, self.n):
            if lb != domain[i, 0] or rb != domain[i, 1]:
                self.is_domain_hypercube = False
                return

    def discretize_domain(self, h=None):
        ''' this method discretizes the hyper-rectangular domain uniformly with step-size h
        '''
        if h is not None:
            self.h = h
        assert self.h is not None, ''

        # construct not sparse nd grid
        try:
            mgrid_input = []
            for i in range(self.n):
                mgrid_input.append(
                    slice(self.domain[i, 0], self.domain[i, 1] + self.h, self.h)
                )
            self.domain_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)
        except MemoryError as e:
            print('MemoryError: {}'.format(e))
            sys.exit()

        # check shape
        assert self.domain_h.shape[-1] == self.n, ''

        # save number of indices per axis
        self.Nx = self.domain_h.shape[:-1]

        # save number of flattened indices
        N = 1
        for i in range(self.n):
            N *= self.Nx[i]
        self.Nh = N

    def discretize_domain_1d(self, h=None):
        ''' this method discretizes the i-th domain coordinate
        '''
        assert self.is_domain_hypercube, ''

        if h is not None:
            self.h = h
        assert self.h is not None, ''

        self.domain_i_h = np.arange(self.domain[0, 0], self.domain[0, 1] + h, h)
        #self.domain_i_h = np.expand_dims(self.domain_i_h, 1)
        self.Nh = self.domain_i_h.shape[0]

    def sample_domain_uniformly(self, N):
        x = np.random.uniform(
            self.domain[:, 0],
            self.domain[:, 1],
            (N, self.n),
        )
        return x

    def sample_multivariate_normal(self, mean, cov, N):
        mean_tensor = torch.tensor(mean, requires_grad=False)
        cov_tensor = torch.tensor(cov, requires_grad=False)
        x = np.empty((N, self.n))
        m = MultivariateNormal(mean_tensor, cov_tensor)
        for i in np.arange(N):
            x[i, :] = m.sample().numpy()
        return x

    def sample_domain_boundary_uniformly(self):
        x = np.empty(self.n)
        i = np.random.randint(self.n)
        for j in range(self.n):
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

    def sample_domain_boundary_uniformly_vec(self, N):
        x = self.sample_domain_uniformly(N)
        i = np.random.randint(self.n, size=N)
        k = np.random.randint(2, size=N)
        for j in np.arange(N):
            x[j, i[j]] = self.domain[i[j], k[j]]
        return x

    def sample_S_uniformly(self, N):
        x = np.empty((N, self.n))
        for j in range(N):
            is_in_target_set = True
            while is_in_target_set:
                x_j = self.sample_domain_uniformly(N=1)
                is_in_target_set = (
                    (x_j >= self.target_set[:, 0]) &
                    (x_j <= self.target_set[:, 1])
                ).all()
            x[j] = x_j
        return x

    def get_flat_domain_h(self):
        ''' this method returns the flattened discretized domain
        '''
        return self.domain_h.reshape(self.Nh, self.n)

    def get_time_index(self, t):
        ''' returns time index for the given time t
        '''
        assert 0 <= t <= self.T, ''

        return int(np.ceil(t / self.dt))

    def get_index(self, x):
        ''' returns the index of the point of the grid closest to x
        '''
        assert x.ndim == 1, ''
        assert x.shape[0] == self.n, ''

        idx = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], self.Nx[i])
            idx[i] = np.argmin(np.abs(axis_i - x[i]))

        return tuple(idx)

    def get_index_vectorized(self, x):
        ''' returns the index of the point of the grid closest to x
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.n, ''

        # get index of xzero
        idx = [None for i in range(self.n)]
        for i in range(self.n):
            axis_i = np.linspace(self.domain[i, 0], self.domain[i, 1], self.Nx[i])
            idx[i] = tuple(np.argmin(np.abs(axis_i - x[:, i].reshape(x.shape[0], 1)), axis=1))

        idx = tuple(idx)
        return idx

    def get_x(self, idx):
        ''' returns the coordinates of the point determined by the axis indices idx
        '''
        return self.domain_h[idx]

    def get_idx_target_set(self):
        assert self.domain_h is not None, ''

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.n)

        # boolean array telling us if x is in the target set
        is_in_target_set = (
            (x >= self.target_set[:, 0]) &
            (x <= self.target_set[:, 1])
        ).all(axis=1).reshape(self.Nh, 1)

        # get index
        return np.where(is_in_target_set == True)[0]

    def get_hjb_solver(self, h=None) -> None:
        from sde_importance_sampling.hjb_solver import SolverHJB

        if h is None and self.n == 1:
            h = 0.001
        elif h is None and self.n == 2:
            h = 0.005
        elif h is None and self.n == 3:
            h = 0.1
        elif h is None:
            return

        # initialize hjb solver
        sol_hjb = SolverHJB(
            problem_name=self.problem_name,
            potential_name=self.potential_name,
            n=self.n,
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
            n=self.n,
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

    def get_not_controlled_sampling(self, dt, N, seed=None) -> None:
        from sde_importance_sampling.importance_sampling import Sampling

        # initialize not controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = False

        # set Euler-Marujama discretiztion step and number of trajectories
        sample.dt = dt
        sample.N = N
        sample.seed = seed

        # set path
        sample.set_not_controlled_dir_path()

        # load already sampled statistics
        if sample.load():
            return sample

    def get_hjb_sampling(self, sol_hjb_dir_path, dt, N, seed=None) -> None:
        from sde_importance_sampling.importance_sampling import Sampling

        # initialize not controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = True

        # set Euler-Marujama discretiztion step and number of trajectories
        sample.dt = dt
        sample.N = N
        sample.seed = seed

        # set path
        sample.set_controlled_dir_path(sol_hjb_dir_path)

        # load already sampled statistics
        if sample.load():
            return sample

    def get_metadynamics_sampling(self, meta_type, weights_type, omega_0, k, N, seed=None):
        from sde_importance_sampling.importance_sampling import Sampling
        from sde_importance_sampling.metadynamics import Metadynamics

        # initialize controlled sampling object
        sample = Sampling.new_from(self)
        sample.is_controlled = True

        # initialize meta nd object
        meta = Metadynamics(
            sample=sample,
            meta_type=meta_type,
            weights_type=weights_type,
            omega_0=omega_0,
            k=k,
            N=N,
            seed=seed,
        )

        # set path
        meta.set_dir_path()

        # load already sampled trajectories
        meta.load()
        return meta

    def get_metadynamics_nn_sampling(self, dt, sigma_i, meta_type, k, N):
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
            N=N,
            sigma_i=sigma_i,
        )
        meta.meta_type = meta_type

        # set path
        meta.set_dir_path()

        # load already sampled trajectories
        meta.load()
        return meta

    def get_flat_bias_sampling(self, dt, k_lim, N):
        from mds.langevin_nd_importance_sampling import Sampling
        from mds.langevin_nd_flat_bias_potential import GetFlatBiasPotential

        # initialize sampling object
        sample = Sampling.new_from(self)
        sample.dt = dt
        sample.k_lim = k_lim
        sample.N = N

        # initialize flatbias object
        flatbias = GetFlatBiasPotential(sample)

        # set path
        flatbias.set_dir_path()

        # load already sampled trajectories
        flatbias.load()
        return flatbias

    def write_setting(self, f):
        '''
        '''
        f.write('\nSDE Setting\n')
        f.write('potential: {}\n'.format(self.potential_name))
        f.write('alpha: {}\n'.format(self.alpha))
        f.write('beta: {:2.1f}\n'.format(self.beta))

        target_set = 'target set: ['
        for i in range(self.n):
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
