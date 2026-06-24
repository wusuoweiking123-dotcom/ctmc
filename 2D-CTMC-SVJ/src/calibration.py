# -*- coding: utf-8 -*-
"""
文件名: src/calibration.py
功能描述: 两阶段校准模块
          Stage 1: 校准 Heston (SV) 参数 — kappa, theta, sigma_v, rho, V_0
          Stage 2: 校准 SVJ 跳跃参数 — lambda_jump, mu_J, sigma_J (Heston参数固定)
          使用 OTM 期权价格的最小加权均方误差作为目标函数
作者: [Author]
创建日期: 2026-05-07
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from loguru import logger


# =============================================================================
# Stage 1: Heston 校准
# =============================================================================

_HESTON_BOUNDS = [
    (0.001, 1.0),   # V_0
    (0.1, 20.0),    # kappa
    (0.001, 1.0),   # theta
    (0.05, 2.0),    # sigma_v
    (-0.99, 0.0),   # rho
]


def _heston_objective(params_array, cal_data, base_params):
    """
    Heston 校准目标函数: 加权均方误差 (向量化版本)
    按 (T, option_type) 分组批量计算，消除逐行 Python 循环
    """
    V_0, kappa, theta, sigma_v, rho = params_array

    if V_0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0:
        return 1e10
    if 2 * kappa * theta < sigma_v ** 2 * 0.5:
        return 1e10 + 100 * (sigma_v ** 2 - 2 * kappa * theta)

    from src.heston_analytical import compute_heston_prices_batch

    underlying_type = base_params.get('underlying_type', 'spot')
    S_0 = base_params['S_0']
    r_val = base_params['r']

    total_error = 0.0
    try:
        for (T_val, ot), group in cal_data.groupby(['T', 'option_type']):
            model_prices = compute_heston_prices_batch(
                S_0, group['K'].values, V_0, r_val,
                kappa, theta, sigma_v, rho, T_val,
                option_type=ot, underlying_type=underlying_type,
            )
            diff = (model_prices - group['market_price'].values) * group['weight'].values
            total_error += np.sum(diff ** 2)
    except Exception:
        total_error += 100.0

    if not np.isfinite(total_error):
        return 1e10
    return total_error


def _heston_objective_silent(params_array, cal_data, base_params):
    """静默版目标函数 (不输出日志)"""
    import logging
    prev = logging.getLogger('loguru').level
    logging.getLogger('loguru').setLevel(logging.ERROR)
    try:
        return _heston_objective(params_array, cal_data, base_params)
    finally:
        logging.getLogger('loguru').setLevel(prev)


def calibrate_heston(cal_data, initial_params=None, method='L-BFGS-B',
                     n_multistart=3, use_de=False):
    """
    Stage 1: 校准 Heston 模型参数

    通过最小化模型价格与市场 OTM 期权价格的加权均方误差来拟合参数。

    参数:
        cal_data (pd.DataFrame): 校准数据，需包含 S, K, T, r, market_price, option_type, weight
        initial_params (dict): 初始参数猜测，默认使用 Heston 默认值
        method (str): 优化方法，默认 L-BFGS-B
        n_multistart (int): 多起点数量，默认 3
        use_de (bool): 是否使用差分进化（全局优化），默认 False

    返回值:
        dict: 校准后的 Heston 参数 + 校准误差
    """
    logger.info("=" * 50)
    logger.info("Stage 1: Calibrating Heston parameters")
    logger.info(f"  Data points: {len(cal_data)}, Method: {method}, Multistart: {n_multistart}")

    if initial_params is None:
        from commonConfig import HESTON_DEFAULT_PARAMS
        initial_params = HESTON_DEFAULT_PARAMS.copy()

    x0 = np.array([
        initial_params.get('V_0', 0.04),
        initial_params.get('kappa', 2.0),
        initial_params.get('theta', 0.04),
        initial_params.get('sigma_v', 0.3),
        initial_params.get('rho', -0.7),
    ])

    underlying_type = initial_params.get('underlying_type', 'spot')

    base_params = {
        'S_0': cal_data['S'].iloc[0],
        'r': cal_data['r'].iloc[0],
        'underlying_type': underlying_type,
    }

    if use_de:
        logger.info("  Using differential evolution (global optimizer)")
        result = differential_evolution(
            _heston_objective_silent,
            bounds=_HESTON_BOUNDS,
            args=(cal_data, base_params),
            seed=42, maxiter=500, tol=1e-8,
            polish=True,
        )
        best_x = result.x
        best_fun = result.fun
    else:
        best_x = x0
        best_fun = _heston_objective_silent(x0, cal_data, base_params)
        logger.info(f"  Initial guess error: {best_fun:.6f}")

        starts = [x0]
        rng = np.random.default_rng(42)
        for _ in range(n_multistart - 1):
            perturbed = x0 * (1 + rng.uniform(-0.3, 0.3, size=len(x0)))
            perturbed = np.clip(perturbed, [b[0] for b in _HESTON_BOUNDS],
                                         [b[1] for b in _HESTON_BOUNDS])
            starts.append(perturbed)

        for i, start in enumerate(starts):
            try:
                res = minimize(
                    _heston_objective_silent,
                    start,
                    args=(cal_data, base_params),
                    method=method,
                    bounds=_HESTON_BOUNDS,
                    options={'maxiter': 1000, 'ftol': 1e-10},
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_x = res.x
                    logger.info(f"  Start {i+1}: error={res.fun:.6f} (new best)")
                else:
                    logger.info(f"  Start {i+1}: error={res.fun:.6f}")
            except Exception as e:
                logger.warning(f"  Start {i+1} failed: {e}")

    calibrated = {
        'V_0': best_x[0],
        'kappa': best_x[1],
        'theta': best_x[2],
        'sigma_v': best_x[3],
        'rho': best_x[4],
        'S_0': base_params['S_0'],
        'r': base_params['r'],
        'underlying_type': base_params.get('underlying_type', 'spot'),
    }

    feller = 2 * calibrated['kappa'] * calibrated['theta'] / calibrated['sigma_v'] ** 2
    logger.info(f"  Calibrated: V_0={calibrated['V_0']:.5f}, kappa={calibrated['kappa']:.4f}, "
                f"theta={calibrated['theta']:.5f}, sigma_v={calibrated['sigma_v']:.4f}, "
                f"rho={calibrated['rho']:.4f}")
    logger.info(f"  Feller ratio: {feller:.3f} (should be > 1)")
    logger.info(f"  Objective value: {best_fun:.6f}")

    rmse = np.sqrt(best_fun / len(cal_data)) if len(cal_data) > 0 else 0
    logger.info(f"  RMSE (weighted): {rmse:.6f}")

    return {
        'model_params': calibrated,
        'objective_value': best_fun,
        'rmse': rmse,
        'feller_ratio': feller,
    }


# =============================================================================
# Stage 2: SVJ 跳跃参数校准
# =============================================================================

_SVJ_JUMP_BOUNDS = [
    (0.01, 5.0),    # lambda_jump
    (-0.5, 0.0),    # mu_J
    (0.01, 1.0),    # sigma_J
]


def _svj_objective(params_array, cal_data, heston_params):
    """
    SVJ 跳跃参数校准目标函数 (向量化版本)
    按 (T, option_type) 分组批量计算
    """
    lambda_jump, mu_J, sigma_J = params_array

    if lambda_jump <= 0 or sigma_J <= 0:
        return 1e10

    from src.svj_analytical import compute_svj_prices_batch

    underlying_type = heston_params.get('underlying_type', 'spot')

    total_error = 0.0
    try:
        for (T_val, ot), group in cal_data.groupby(['T', 'option_type']):
            model_prices = compute_svj_prices_batch(
                S0=heston_params['S_0'], K_arr=group['K'].values,
                V0=heston_params['V_0'], r=heston_params['r'],
                kappa=heston_params['kappa'], theta=heston_params['theta'],
                sigma_v=heston_params['sigma_v'], rho=heston_params['rho'],
                T=T_val, lambda_jump=lambda_jump, mu_J=mu_J, sigma_J=sigma_J,
                option_type=ot, underlying_type=underlying_type,
            )
            diff = (model_prices - group['market_price'].values) * group['weight'].values
            total_error += np.sum(diff ** 2)
    except Exception:
        total_error += 100.0

    if not np.isfinite(total_error):
        return 1e10
    return total_error


def _svj_objective_silent(params_array, cal_data, heston_params):
    import logging
    prev = logging.getLogger('loguru').level
    logging.getLogger('loguru').setLevel(logging.ERROR)
    try:
        return _svj_objective(params_array, cal_data, heston_params)
    finally:
        logging.getLogger('loguru').setLevel(prev)


def calibrate_svj(cal_data, heston_params, initial_jump_params=None,
                  method='L-BFGS-B', n_multistart=3, use_de=False):
    """
    Stage 2: 校准 SVJ 跳跃参数 (Heston 参数固定)

    参数:
        cal_data (pd.DataFrame): 校准数据
        heston_params (dict): Stage 1 校准得到的 Heston 参数
        initial_jump_params (dict): 跳跃参数初始猜测
        method (str): 优化方法
        n_multistart (int): 多起点数量
        use_de (bool): 是否使用差分进化

    返回值:
        dict: 校准后的完整 SVJ 参数 + 校准误差
    """
    logger.info("=" * 50)
    logger.info("Stage 2: Calibrating SVJ jump parameters")
    logger.info(f"  Data points: {len(cal_data)}, Method: {method}")

    if initial_jump_params is None:
        from commonConfig import SVJ_JUMP_DEFAULT_PARAMS
        initial_jump_params = SVJ_JUMP_DEFAULT_PARAMS.copy()

    x0 = np.array([
        initial_jump_params.get('lambda_jump', 0.1),
        initial_jump_params.get('mu_J', -0.05),
        initial_jump_params.get('sigma_J', 0.1),
    ])

    if use_de:
        result = differential_evolution(
            _svj_objective_silent,
            bounds=_SVJ_JUMP_BOUNDS,
            args=(cal_data, heston_params),
            seed=42, maxiter=500, tol=1e-8,
            polish=True,
        )
        best_x = result.x
        best_fun = result.fun
    else:
        best_x = x0
        best_fun = _svj_objective_silent(x0, cal_data, heston_params)
        logger.info(f"  Initial guess error: {best_fun:.6f}")

        starts = [x0]
        rng = np.random.default_rng(42)
        for _ in range(n_multistart - 1):
            perturbed = x0 * (1 + rng.uniform(-0.3, 0.3, size=len(x0)))
            perturbed = np.clip(perturbed, [b[0] for b in _SVJ_JUMP_BOUNDS],
                                         [b[1] for b in _SVJ_JUMP_BOUNDS])
            starts.append(perturbed)

        for i, start in enumerate(starts):
            try:
                res = minimize(
                    _svj_objective_silent,
                    start,
                    args=(cal_data, heston_params),
                    method=method,
                    bounds=_SVJ_JUMP_BOUNDS,
                    options={'maxiter': 1000, 'ftol': 1e-10},
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_x = res.x
                    logger.info(f"  Start {i+1}: error={res.fun:.6f} (new best)")
                else:
                    logger.info(f"  Start {i+1}: error={res.fun:.6f}")
            except Exception as e:
                logger.warning(f"  Start {i+1} failed: {e}")

    calibrated_jumps = {
        'lambda_jump': best_x[0],
        'mu_J': best_x[1],
        'sigma_J': best_x[2],
        'jump_type': 'merton',
    }

    k_bar = np.exp(calibrated_jumps['mu_J'] + calibrated_jumps['sigma_J'] ** 2 / 2) - 1
    logger.info(f"  Calibrated: lambda={calibrated_jumps['lambda_jump']:.4f}, "
                f"mu_J={calibrated_jumps['mu_J']:.4f}, "
                f"sigma_J={calibrated_jumps['sigma_J']:.4f}")
    logger.info(f"  Compensator k_bar={k_bar:.6f}")
    logger.info(f"  Objective value: {best_fun:.6f}")

    rmse = np.sqrt(best_fun / len(cal_data)) if len(cal_data) > 0 else 0
    logger.info(f"  RMSE (weighted): {rmse:.6f}")

    full_params = heston_params.copy()

    return {
        'model_params': full_params,
        'jump_params': calibrated_jumps,
        'objective_value': best_fun,
        'rmse': rmse,
    }


# =============================================================================
# 完整两阶段校准
# =============================================================================

def calibrate_full(cal_data, initial_model_params=None, initial_jump_params=None,
                   config=None):
    """
    完整的两阶段校准流程

    Stage 1: 校准 Heston 参数 → Stage 2: 校准跳跃参数

    参数:
        cal_data (pd.DataFrame): 校准数据
        initial_model_params (dict): Heston 初始参数
        initial_jump_params (dict): 跳跃初始参数
        config (dict): 校准配置

    返回值:
        dict: 完整校准结果
    """
    if config is None:
        config = {
            'method': 'L-BFGS-B',
            'n_multistart': 3,
            'use_de': False,
        }

    logger.info("=" * 60)
    logger.info("Two-Stage Calibration: Heston-SVJ")
    logger.info("=" * 60)

    stage1 = calibrate_heston(
        cal_data, initial_model_params,
        method=config.get('method', 'L-BFGS-B'),
        n_multistart=config.get('n_multistart', 3),
        use_de=config.get('use_de', False),
    )

    stage2 = calibrate_svj(
        cal_data, stage1['model_params'], initial_jump_params,
        method=config.get('method', 'L-BFGS-B'),
        n_multistart=config.get('n_multistart', 3),
        use_de=config.get('use_de', False),
    )

    logger.info("=" * 60)
    logger.info("Calibration Complete")
    logger.info(f"  Stage 1 (Heston) RMSE: {stage1['rmse']:.6f}")
    logger.info(f"  Stage 2 (SVJ)    RMSE: {stage2['rmse']:.6f}")
    logger.info(f"  Improvement: {(1 - stage2['rmse']/max(stage1['rmse'],1e-10))*100:.1f}%")
    logger.info("=" * 60)

    return {
        'stage1': stage1,
        'stage2': stage2,
        'model_params': stage2['model_params'],
        'jump_params': stage2['jump_params'],
    }
