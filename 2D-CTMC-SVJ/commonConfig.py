# -*- coding: utf-8 -*-
"""
文件名: commonConfig.py
功能描述: 2D-CTMC-SVJ 项目公共配置文件，包含模型参数、网格参数、优化器配置等
作者: [Author]
创建日期: 2026-05-06
"""

HESTON_DEFAULT_PARAMS = {
    'S_0': 100.0,
    'V_0': 0.04,
    'r': 0.03,
    'kappa': 2.0,
    'theta': 0.04,
    'sigma_v': 0.3,
    'rho': -0.7,
    'underlying_type': 'spot',
}

THREE_HALVES_DEFAULT_PARAMS = {
    'S_0': 100.0,
    'V_0': 0.04,
    'r': 0.03,
    'kappa': 2.0,
    'theta': 0.04,
    'sigma_v': 0.3,
    'rho': -0.5,
}

SVJ_JUMP_DEFAULT_PARAMS = {
    'lambda_jump': 0.1,
    'mu_J': -0.05,
    'sigma_J': 0.10,
    'jump_type': 'merton',
}

KOU_JUMP_DEFAULT_PARAMS = {
    'lambda_jump': 0.1,
    'p': 0.3,
    'eta1': 3.0,
    'eta2': 2.0,
    'jump_type': 'kou',
}

LAYER1_GRID_CONFIG = {
    'm': 40,
    'v_lower': None,
    'v_upper': None,
    'grid_type': 'sinh',
    'sinh_alpha': 0.5,
    'v_lower_factor': None,
    'v_upper_factor': None,
    'v_lower_abs': 0.008,
    'v_upper_abs': 0.120,
}

LAYER2_GRID_CONFIG = {
    'N': 200,
    'x_lower': None,
    'x_upper': None,
    'grid_type': 'sinh',
    'sinh_alpha': 1.0,
    'x_lower_factor': 3.0,
    'x_upper_factor': 3.0,
}

OPTION_DEFAULT_PARAMS = {
    'K': 100.0,
    'T': 1.0,
    'r': 0.03,
    'option_type': 'put',
    'option_style': 'eu',
}

CTMC_COMPUTE_PARAMS = {
    'n_time_steps': 400,
    'fast_algorithm': True,
}

OPTIMIZER_CONFIG = {
    'method': 'L-BFGS-B',
    'tol': 1e-8,
    'maxiter': 1000,
    'n_multistart': 5,
    'random_seed': 42,
}

FILE_PATHS = {
    'futures_data': 'input/Futures_au2412.csv',
    'options_data': 'input/Futures_put_options_au2412.csv',
    'result_dir': 'result/',
    'log_dir': 'log/',
    'data_dir': 'data/',
}

LOG_CONFIG = {
    'level': 'INFO',
    'format': '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
              '<level>{level:<8}</level> | '
              '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | '
              '<level>{message}</level>',
    'rotation': '10 MB',
}

VALIDATION_CONFIG = {
    'enabled': True,
    'reference_prices': {
        'call_K100_T1': None,
        'put_K100_T1': None,
    },
    'tolerance_abs': 0.05,
    'tolerance_rel': 0.01,
}
