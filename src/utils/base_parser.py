import argparse

from sde.langevin_sde import POTENTIAL_NAMES

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--problem-name',
        dest='problem_name',
        choices=['langevin_det-t', 'langevin_stop-t'],
        default='langevin_stop-t',
        help='Set type of problem. Default: overdamped langevin with stopping times',
    )
    parser.add_argument(
        '--potential-name',
        dest='potential_name',
        choices=POTENTIAL_NAMES,
        default='nd_2well',
        help='Set type of potential. Default: double well',
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--d',
        dest='d',
        type=int,
        default=1,
        help='Set the dimension d. Default: 1',
    )
    parser.add_argument(
        '--alpha-i',
        dest='alpha_i',
        type=float,
        default=1.,
        help='Set barrier height of the i-th coordinate for the multidimensional extension \
              of the double well potential. Default: 1.',
    )
    parser.add_argument(
        '--alpha-j',
        dest='alpha_j',
        type=float,
        default=1.,
        help='Set barrier height of the j-th coordinate for the multidimensional extension \
              of the double well potential. Default: 1.',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1.,
        help='Set the beta parameter. Default: 1.',
    )
    parser.add_argument(
        '--nu-i',
        dest='nu_i',
        type=float,
        default=3.,
        help='Set nd quadratic one well i-th parameters. Default: 1.',
    )
    parser.add_argument(
        '--nu-j',
        dest='nu_j',
        type=float,
        default=3.,
        help='Set nd quadratic one well j-th parameters. Default: 1.',
    )
    parser.add_argument(
        '--xzero-i',
        dest='xzero_i',
        type=float,
        default=-1.,
        help='Set the initial posicion of the process at each axis. Default: -1',
    )
    parser.add_argument(
        '--h',
        dest='h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--h-hjb',
        dest='h_hjb',
        type=float,
        default=0.1,
        help='Set the discretization step size for the hjb sol. Default: 0.1',
    )
    parser.add_argument(
        '--dt-hjb',
        dest='dt_hjb',
        type=float,
        default=0.005,
        help='Set the time discretization increment for the hjb sol with det time horizont. Default: 0.005',
    )
    parser.add_argument(
        '--K',
        dest='K',
        type=int,
        default=10**3,
        help='Set number of trajectories to sample. Default: 1.000',
    )
    parser.add_argument(
        '--K-batch',
        dest='K_batch',
        type=int,
        default=10**5,
        help='Set number of trajectories to sample. Default: 100.000',
    )
    parser.add_argument(
        '--dt',
        dest='dt',
        type=float,
        default=0.001,
        help='Set dt. Default: 0.001',
    )
    parser.add_argument(
        '--T',
        dest='T',
        type=float,
        default=1.,
        help='Set deterministic time horizont. Default: 1.',
    )
    parser.add_argument(
        '--k-lim',
        dest='k_lim',
        type=int,
        default=10**8,
        help='Set maximal number of time steps. Default: 100.000.000',
    )
    parser.add_argument(
        '--distributed',
        dest='distributed',
        choices=['uniform', 'meta'],
        default='uniform',
        help='Type of ansatz distribution. Default: uniform',
    )
    parser.add_argument(
        '--m-i',
        dest='m_i',
        type=int,
        default=30,
        help='Set number of uniformly distributed ansatz functions per axis \
              that you want to use. Default: 30',
    )
    parser.add_argument(
        '--sigma-i',
        dest='sigma_i',
        type=float,
        default=0.5,
        help='Set the diagonal of the covariance matrix of the ansatz functions. Default: 0.5',
    )
    parser.add_argument(
        '--cv-type',
        dest='cv_type',
        choices=['identity', 'projection'],
        default='identity',
        help='type of metadynamics algorithm. Default: cumulative algorithm',
    )
    parser.add_argument(
        '--meta-type',
        dest='meta_type',
        choices=['independent', 'cumulative'],
        default='cumulative',
        help='type of metadynamics algorithm. Default: cumulative algorithm',
    )
    parser.add_argument(
        '--weights-type',
        dest='weights_type',
        choices=['constant', 'geometric'],
        default='geometric',
        help='type of decay of the gaussian ansatz weights. Default: geometric',
    )
    parser.add_argument(
        '--omega-0-meta',
        dest='omega_0_meta',
        type=float,
        default=1.,
        help='scaling factor of the weights. Default: 1.',
    )
    parser.add_argument(
        '--dt-meta',
        dest='dt_meta',
        type=float,
        default=0.01,
        help='Set dt. Default: 0.01',
    )
    parser.add_argument(
        '--K-meta',
        dest='K_meta',
        type=int,
        default=1,
        help='Set number of trajectories to sample. Default: 1',
    )
    parser.add_argument(
        '--sigma-i-meta',
        dest='sigma_i_meta',
        type=float,
        default=0.5,
        help='Set the diagonal of the covariance matrix of the ansatz functions. Default: 0.5',
    )
    parser.add_argument(
        '--delta-meta',
        dest='delta_meta',
        type=float,
        default=1.,
        help='time interval before adding a bias function. Default: 1.',
    )
    parser.add_argument(
        '--theta',
        dest='theta',
        choices=['random', 'null', 'not-controlled', 'meta', 'hjb'],
        default='null',
        help='Type of control. Default: null',
    )
    parser.add_argument(
        '--target-f',
        dest='target_function',
        choices=['value-f', 'control'],
        default='value-f',
        help='Target function to approximate. Default: value-f',
    )
    parser.add_argument(
        '--train-alg',
        dest='train_alg',
        choices=['classic', 'alternative'],
        default='classic',
        help='Set type of approximation problem training algorithm. Default: "classic"',
    )
    parser.add_argument(
        '--grad-estimator',
        dest='grad_estimator',
        choices=['ipa', 'eff_loss'],
        default='eff_loss',
        help='Set type of grad estimation. Default: "ipa"',
    )
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        choices=['sgd', 'adam'],
        default='adam',
        help='Set type of optimizer. Default: "adam"',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--n-iterations-lim',
        dest='n_iterations_lim',
        type=int,
        default=100,
        help='Set maximal number of sgd iterations. Default: 100',
    )
    parser.add_argument(
        '--n-iterations-backup',
        dest='n_iterations_backup',
        type=int,
        help='Set number of sgd iterations between saving the arrays. Default: None',
    )
    parser.add_argument(
        '--dt-sgd',
        dest='dt_sgd',
        type=float,
        default=0.001,
        help='Set dt in the sgd. Default: 0.001',
    )
    parser.add_argument(
        '--K-sgd',
        dest='K_sgd',
        type=int,
        default=1000,
        help='Set number of trajectories to sample in the sgd. Default: 1000',
    )
    parser.add_argument(
        '--seed-sgd',
        dest='seed_sgd',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--K-grad',
        dest='K_grad',
        type=int,
        default=50,
        help='Set number of times the gradient is sampled. Default: 50',
    )
    parser.add_argument(
        '--K-train',
        dest='K_train',
        type=int,
        default=1000,
        help='Set number of points used in the approximation problem. Default: 1000',
    )
    parser.add_argument(
        '--do-u-l2-error',
        dest='do_u_l2_error',
        action='store_true',
        help='compute u l2 error. Default: False',
    )
    parser.add_argument(
        '--d-layers',
        nargs='+',
        dest='d_layers',
        type=int,
        help='Set dimensions of the NN inner layers',
    )
    parser.add_argument(
        '--n-layers',
        dest='n_layers',
        type=int,
        default=2,
        help='Set number of hidden layers. Default: 2',
    )
    parser.add_argument(
        '--d-layer',
        dest='d_layer',
        type=int,
        default=30,
        help='Set dimension of the hidden layers. Default: 30',
    )
    parser.add_argument(
        '--dense',
        dest='dense',
        action='store_true',
        help='Chooses a dense feed forward NN. Default: False',
    )
    parser.add_argument(
        '--activation',
        dest='activation_type',
        choices=['relu', 'tanh'],
        default='relu',
        help='Type of activation function. Default: relu',
    )
    parser.add_argument(
        '--do-importance-sampling',
        dest='do_importance_sampling',
        action='store_true',
        help='Sample controlled dynamics',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--do-report',
        dest='do_report',
        action='store_true',
        help='Write / Print report. Default: False',
    )
    parser.add_argument(
        '--load',
        dest='load',
        action='store_true',
        help='Load already computed hjb results. Default: False',
    )
    parser.add_argument(
        '--save-trajectory',
        dest='save_trajectory',
        action='store_true',
        help='Save the first trajectory sampled. Default: False',
    )
    return parser
