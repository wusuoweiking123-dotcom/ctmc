# -*- coding: utf-8 -*-
"""
文件名: src/tensor_pricing.py
功能描述: 张量加速的二维 CTMC 期权定价模块 (SVD 压缩层)

核心创新:
    1. 价值矩阵 B ∈ R^{m×N} 的截断 SVD 低秩表示 (张量压缩)
    2. 每步 SVD 重压缩保持低秩结构 + 隐式去噪
    3. 基础张量操作 (einsum, variance shift) 复用 option_pricing 模块

依赖:
    src/option_pricing 提供 _ensure_tensor, _apply_variance_shift,
    _build_payoff_matrix, _bilinear_interpolate, _batch_linterp

作者: [Author]
创建日期: 2026-05-09
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
)


def _randomized_svd(B, rank, n_oversamples=5, n_power_iters=2):
    m, N = B.shape
    k = min(rank + n_oversamples, min(m, N))

    rng = np.random.RandomState(42)
    Omega = rng.randn(N, k)
    Y = B @ Omega

    for _ in range(n_power_iters):
        Y = B @ (B.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B_proj = Q.T @ B
    U_small, s, Vt = np.linalg.svd(B_proj, full_matrices=False)
    U = Q @ U_small

    r = min(rank, len(s))
    return U[:, :r], s[:r], Vt[:r, :]


def _svd_compress(B, max_rank=None, tol=1e-6):
    m, N = B.shape
    use_randomized = max_rank is not None and max_rank < min(m, N) // 3

    if use_randomized:
        U, s, Vt = _randomized_svd(B, max_rank)
    else:
        U, s, Vt = np.linalg.svd(B, full_matrices=False)

    if max_rank is not None:
        r = min(max_rank, len(s))
    else:
        total_energy = np.sum(s ** 2)
        if total_energy > 0:
            cumulative = np.cumsum(s ** 2) / total_energy
            r = max(np.searchsorted(cumulative, 1.0 - tol) + 1, 1)
        else:
            r = 1

    r = min(r, len(s))
    return U[:, :r], s[:r], Vt[:r, :], r


def _reconstruct(U, s, Vt):
    return U @ np.diag(s) @ Vt


def price_european_tensor(Q_variance, G_tensor, price_grid, variance_grid,
                           model_params, option_params, n_time_steps,
                           max_rank=None, svd_tol=1e-6,
                           smooth_payoff=False, splitting='lie'):
    G_tensor = _ensure_tensor(G_tensor)
    m = G_tensor.shape[0]
    N = G_tensor.shape[1]
    T = option_params.get('T', 1.0)
    r = model_params['r']
    dt = T / n_time_steps

    logger.info(
        f"Tensor European pricing: {m}x{N}, {n_time_steps} steps, "
        f"splitting={splitting}, max_rank={max_rank}, tol={svd_tol}"
    )

    P_V = expm(Q_variance * dt)
    P_X = np.stack([expm(G_tensor[l] * dt) for l in range(m)])

    if splitting == 'strang':
        P_X_half = np.stack([expm(G_tensor[l] * dt / 2.0) for l in range(m)])

    B = _build_payoff_matrix(price_grid, variance_grid, model_params, option_params,
                              m, N, smooth_payoff=smooth_payoff)
    discount = np.exp(-r * dt)

    use_shift = abs(model_params.get('rho', 0.0)) > 1e-14

    rank_history = []

    for z in range(n_time_steps - 1, -1, -1):
        if splitting == 'strang':
            E = np.einsum('lij,lj->li', P_X_half, B)
            if use_shift:
                B = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
            else:
                B = P_V @ E
            B = np.einsum('lij,lj->li', P_X_half, B)
        else:
            E = np.einsum('lij,lj->li', P_X, B)
            if use_shift:
                B = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
            else:
                B = P_V @ E

        B *= discount

        if max_rank is not None or svd_tol is not None:
            U, s, Vt, rank = _svd_compress(B, max_rank=max_rank, tol=svd_tol)
            B = _reconstruct(U, s, Vt)
            rank_history.append(rank)

    from src.layer2_price import compute_initial_auxiliary_value

    X_0 = compute_initial_auxiliary_value(
        model_params['S_0'], model_params['V_0'], model_params
    )
    price = _bilinear_interpolate(B, price_grid, variance_grid, X_0,
                                   model_params['V_0'], m, N)

    if rank_history:
        logger.info(
            f"Tensor pricing done: price={price:.6f}, "
            f"rank: min={min(rank_history)}, max={max(rank_history)}, "
            f"mean={np.mean(rank_history):.1f}"
        )

    return price, rank_history


def price_american_tensor(Q_variance, G_tensor, price_grid, variance_grid,
                           model_params, option_params, n_time_steps,
                           max_rank=None, svd_tol=1e-6,
                           smooth_payoff=False, splitting='strang'):
    from src.layer2_price import (
        compute_initial_auxiliary_value,
        compute_decorrelation_function,
    )
    from src.layer1_variance import get_variance_state_index
    from src.layer2_price import get_price_state_index
    from src.option_pricing import _smoothed_payoff

    G_tensor = _ensure_tensor(G_tensor)
    m = G_tensor.shape[0]
    N = G_tensor.shape[1]
    T = option_params.get('T', 1.0)
    r_rate = model_params['r']
    dt = T / n_time_steps
    K = option_params['K']
    option_type = option_params.get('option_type', 'put')

    logger.info(
        f"Tensor American pricing ({splitting}): {m}x{N}, {n_time_steps} steps, "
        f"K={K}, max_rank={max_rank}"
    )

    P_V = expm(Q_variance * dt)
    P_X = np.stack([expm(G_tensor[l] * dt) for l in range(m)])

    if splitting == 'strang':
        P_X_half = np.stack([expm(G_tensor[l] * dt / 2.0) for l in range(m)])

    rho = model_params['rho']
    gamma_vals, _ = compute_decorrelation_function(variance_grid, model_params)
    S_recovered = np.exp(price_grid[None, :] + rho * gamma_vals[:, None])

    payoff = np.zeros((m, N))
    if smooth_payoff:
        for l in range(m):
            payoff[l, :] = _smoothed_payoff(S_recovered[l], K, option_type)
    else:
        if option_type == 'put':
            payoff = np.maximum(K - S_recovered, 0)
        else:
            payoff = np.maximum(S_recovered - K, 0)

    B_am = payoff.copy()
    B_eu = payoff.copy()
    exercise_boundary = np.full((n_time_steps, m), np.nan)
    discount = np.exp(-r_rate * dt)

    use_shift = abs(model_params.get('rho', 0.0)) > 1e-14

    rank_history_am = []
    rank_history_eu = []

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

        if max_rank is not None or svd_tol is not None:
            U, s, Vt, rank = _svd_compress(B_am, max_rank=max_rank, tol=svd_tol)
            B_am = _reconstruct(U, s, Vt)
            rank_history_am.append(rank)

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

        if max_rank is not None or svd_tol is not None:
            U, s, Vt, rank = _svd_compress(B_eu, max_rank=max_rank, tol=svd_tol)
            B_eu = _reconstruct(U, s, Vt)
            rank_history_eu.append(rank)

    X_0 = compute_initial_auxiliary_value(
        model_params['S_0'], model_params['V_0'], model_params
    )

    american_price = _bilinear_interpolate(
        B_am, price_grid, variance_grid, X_0, model_params['V_0'], m, N
    )
    european_price = _bilinear_interpolate(
        B_eu, price_grid, variance_grid, X_0, model_params['V_0'], m, N
    )
    eep = american_price - european_price

    logger.info(
        f"Tensor American: Am={american_price:.6f}, Eu={european_price:.6f}, "
        f"EEP={eep:.6f} ({eep / max(european_price, 1e-10) * 100:.2f}%)"
    )

    return {
        'american_price': american_price,
        'european_price': european_price,
        'early_exercise_premium': eep,
        'exercise_boundary': exercise_boundary,
        'rank_history_am': rank_history_am,
        'rank_history_eu': rank_history_eu,
    }
