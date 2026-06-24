# -*- coding: utf-8 -*-
"""
文件名: src/american_pricing.py
功能描述: 基于双层 CTMC 的美式/百慕大期权定价模块 (张量化版本)

张量化:
    1. np.einsum('lij,lj->li', P_X, B) 批量矩阵-向量乘法
    2. _compute_payoff_and_prices 向量化: 广播替代逐行循环
    3. 方差平移使用 option_pricing._apply_variance_shift (rho≠0 时正确处理)
    4. np.maximum 向量化提前行权检查

作者: [Author]
创建日期: 2026-05-07
张量化: 2026-05-10
"""

import numpy as np
from scipy.linalg import expm
from loguru import logger

from src.option_pricing import (
    _ensure_tensor,
    _apply_variance_shift,
    _build_payoff_matrix,
    _bilinear_interpolate,
    _smoothed_payoff,
)


def _compute_payoff_and_prices(price_grid, variance_grid, model_params, K, option_type,
                                smooth_payoff=False):
    from src.layer2_price import compute_decorrelation_function

    m = len(variance_grid)
    N = len(price_grid)
    rho = model_params['rho']

    gamma_vals, _ = compute_decorrelation_function(variance_grid, model_params)
    S_recovered = np.exp(price_grid[None, :] + rho * gamma_vals[:, None])

    if smooth_payoff:
        payoff = np.zeros((m, N))
        for l in range(m):
            payoff[l, :] = _smoothed_payoff(S_recovered[l], K, option_type)
    else:
        if option_type == 'put':
            payoff = np.maximum(K - S_recovered, 0)
        else:
            payoff = np.maximum(S_recovered - K, 0)

    return payoff, S_recovered


def price_american_fast(Q_variance, G_tensor, price_grid, variance_grid,
                        model_params, option_params, n_time_steps,
                        splitting='strang', smooth_payoff=False):
    G_tensor = _ensure_tensor(G_tensor)
    m = G_tensor.shape[0]
    N = G_tensor.shape[1]
    T = option_params.get('T', 1.0)
    r = model_params['r']
    dt = T / n_time_steps
    K = option_params['K']
    option_type = option_params.get('option_type', 'put')

    logger.info(
        f"American pricing ({splitting}, tensor): {m}x{N}, {n_time_steps} steps, "
        f"dt={dt:.6f}, K={K}, type={option_type}"
    )

    P_V = expm(Q_variance * dt)
    P_X = np.stack([expm(G_tensor[l] * dt) for l in range(m)])

    if splitting == 'strang':
        P_X_half = np.stack([expm(G_tensor[l] * dt / 2.0) for l in range(m)])

    logger.info(f"Transition tensors computed: {m}+1 exponentials")

    payoff, S_recovered = _compute_payoff_and_prices(
        price_grid, variance_grid, model_params, K, option_type,
        smooth_payoff=smooth_payoff
    )

    B_am = payoff.copy()
    B_eu = payoff.copy()

    exercise_boundary = np.full((n_time_steps, m), np.nan)
    discount = np.exp(-r * dt)

    use_shift = abs(model_params.get('rho', 0.0)) > 1e-14

    for z in range(n_time_steps - 1, -1, -1):
        if splitting == 'strang':
            E = np.einsum('lij,lj->li', P_X_half, B_am)
            if use_shift:
                B_am = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
            else:
                B_am = P_V @ E
            B_am = np.einsum('lij,lj->li', P_X_half, B_am)
        else:
            E = np.einsum('lij,lj->li', P_X, B_am)
            if use_shift:
                B_am = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
            else:
                B_am = P_V @ E

        B_am *= discount

        exercise_mask = (payoff > B_am) & (payoff > 0)
        B_am = np.maximum(B_am, payoff)

        if option_type == 'put':
            for l in range(m):
                indices = np.where(exercise_mask[l])[0]
                if len(indices) > 0:
                    exercise_boundary[z, l] = S_recovered[l, indices[-1]]
        else:
            for l in range(m):
                indices = np.where(exercise_mask[l])[0]
                if len(indices) > 0:
                    exercise_boundary[z, l] = S_recovered[l, indices[0]]

        if splitting == 'strang':
            E = np.einsum('lij,lj->li', P_X_half, B_eu)
            if use_shift:
                B_eu = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
            else:
                B_eu = P_V @ E
            B_eu = np.einsum('lij,lj->li', P_X_half, B_eu)
        else:
            E = np.einsum('lij,lj->li', P_X, B_eu)
            if use_shift:
                B_eu = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
            else:
                B_eu = P_V @ E

        B_eu *= discount

    from src.layer2_price import compute_initial_auxiliary_value
    from src.layer1_variance import get_variance_state_index
    from src.layer2_price import get_price_state_index

    X_0 = compute_initial_auxiliary_value(
        model_params['S_0'], model_params['V_0'], model_params
    )

    var_idx = get_variance_state_index(model_params['V_0'], variance_grid)
    price_idx = get_price_state_index(X_0, price_grid)

    american_price = _bilinear_interpolate(
        B_am, price_grid, variance_grid, X_0,
        model_params['V_0'], m, N
    )
    european_price = _bilinear_interpolate(
        B_eu, price_grid, variance_grid, X_0,
        model_params['V_0'], m, N
    )
    eep = american_price - european_price

    logger.info(
        f"American: {american_price:.6f}, European: {european_price:.6f}, "
        f"EEP: {eep:.6f} ({eep / max(european_price, 1e-10) * 100:.2f}%)"
    )

    return {
        'american_price': american_price,
        'european_price': european_price,
        'early_exercise_premium': eep,
        'exercise_boundary': exercise_boundary,
        'var_idx': var_idx,
        'price_idx': price_idx,
    }
