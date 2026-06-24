# -*- coding: utf-8 -*-
"""
文件名: src/option_pricing.py
功能描述: 基于双层 CTMC 的期权定价模块 (张量化版本)

核心张量化:
    1. G_tensor: (m, N, N) 体制生成元张量, 替代 list[(N,N)]
    2. np.einsum('lij,lj->li', P_X, B): 批量矩阵-向量乘法, 替代 Python for 循环
    3. _apply_variance_shift: 向量化方差平移, 消除 m×m 双重循环
    4. _build_payoff_matrix: 向量化支付矩阵构建, 广播替代逐行循环
    5. 向后兼容: 通过 _ensure_tensor 自动转换 list → tensor

作者: [Author]
创建日期: 2026-05-06
张量化: 2026-05-10
"""

import numpy as np
from scipy.linalg import expm
from loguru import logger


def _ensure_tensor(G):
    if isinstance(G, np.ndarray) and G.ndim == 3:
        return G
    return np.stack(G)


def _batch_linterp(xp, fp, x):
    batch, M = x.shape
    idx = np.searchsorted(xp, x.ravel()).reshape(batch, M)
    idx = np.clip(idx, 1, len(xp) - 1)

    x_lo = xp[idx - 1]
    x_hi = xp[idx]
    span = x_hi - x_lo
    span = np.where(span > 0, span, 1.0)
    w = np.clip((x - x_lo) / span, 0.0, 1.0)

    f_lo = fp[idx - 1]
    f_hi = fp[idx]
    result = f_lo + w * (f_hi - f_lo)

    result = np.where(x < xp[0], fp[0], result)
    result = np.where(x > xp[-1], fp[-1], result)
    return result


def _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params):
    rho = model_params.get('rho', 0.0)
    sigma_v = model_params.get('sigma_v', 1.0)
    m, N = E.shape

    if abs(rho) < 1e-14:
        return P_V @ E

    rho_sv = rho / sigma_v
    DX = rho_sv * (variance_grid[:, None] - variance_grid[None, :])

    B = np.zeros((m, N))
    sparsity_threshold = 1e-15

    for j in range(m):
        p_col = P_V[:, j]
        active = np.abs(p_col) > sparsity_threshold
        if not np.any(active):
            continue

        l_idx = np.where(active)[0]
        p_vals = p_col[l_idx]
        dx_vals = DX[l_idx, j]

        zero_shift = np.abs(dx_vals) < 1e-14
        nonzero_shift = ~zero_shift

        if np.any(zero_shift):
            zs_l = l_idx[zero_shift]
            zs_p = p_vals[zero_shift]
            B[zs_l] += zs_p[:, None] * E[j]

        if np.any(nonzero_shift):
            ns_l = l_idx[nonzero_shift]
            ns_p = p_vals[nonzero_shift]
            ns_dx = dx_vals[nonzero_shift]

            targets = price_grid[None, :] + ns_dx[:, None]
            shifted = _batch_linterp(price_grid, E[j], targets)
            B[ns_l] += ns_p[:, None] * shifted

    return B


def _smoothed_payoff(S_vec, K, option_type, epsilon=None):
    if epsilon is None:
        epsilon = K * 0.01

    B = np.zeros_like(S_vec)

    if option_type == 'put':
        far_itm = S_vec <= K - epsilon
        far_otm = S_vec >= K + epsilon
        smooth = (~far_itm) & (~far_otm)

        B[far_itm] = K - S_vec[far_itm]
        B[far_otm] = 0.0
        if np.any(smooth):
            S_s = S_vec[smooth]
            B[smooth] = (K + epsilon - S_s) ** 2 / (4 * epsilon)
    else:
        far_otm = S_vec <= K - epsilon
        far_itm = S_vec >= K + epsilon
        smooth = (~far_otm) & (~far_itm)

        B[far_otm] = 0.0
        B[far_itm] = S_vec[far_itm] - K
        if np.any(smooth):
            S_s = S_vec[smooth]
            B[smooth] = (S_s - K + epsilon) ** 2 / (4 * epsilon)

    return B


def _build_payoff_matrix(price_grid, variance_grid, model_params, option_params,
                         m, N, smooth_payoff=False):
    from src.layer2_price import compute_decorrelation_function

    rho = model_params['rho']
    K = option_params['K']
    option_type = option_params.get('option_type', 'put')

    gamma_vals, _ = compute_decorrelation_function(variance_grid, model_params)
    S_matrix = np.exp(price_grid[None, :] + rho * gamma_vals[:, None])

    if smooth_payoff:
        B = np.zeros((m, N))
        for l in range(m):
            B[l, :] = _smoothed_payoff(S_matrix[l], K, option_type)
    else:
        if option_type == 'put':
            B = np.maximum(K - S_matrix, 0)
        else:
            B = np.maximum(S_matrix - K, 0)
    return B


def _bilinear_interpolate(B, price_grid, variance_grid, X_0, V_0, m, N):
    pi = np.searchsorted(price_grid, X_0)
    pi = max(1, min(pi, N - 1))
    wp = (X_0 - price_grid[pi - 1]) / (price_grid[pi] - price_grid[pi - 1])
    wp = max(0.0, min(wp, 1.0))

    vi = np.searchsorted(variance_grid, V_0)
    vi = max(1, min(vi, m - 1))
    wv = (V_0 - variance_grid[vi - 1]) / (variance_grid[vi] - variance_grid[vi - 1])
    wv = max(0.0, min(wv, 1.0))

    p00 = B[vi - 1, pi - 1]
    p01 = B[vi - 1, pi]
    p10 = B[vi, pi - 1]
    p11 = B[vi, pi]

    return ((1 - wv) * (1 - wp) * p00
            + (1 - wv) * wp * p01
            + wv * (1 - wp) * p10
            + wv * wp * p11)


def build_payoff_vector(price_grid, variance_grid, model_params, option_params):
    m = len(variance_grid)
    N = len(price_grid)
    K = option_params['K']
    option_type = option_params.get('option_type', 'put')

    from src.layer2_price import compute_decorrelation_function

    rho = model_params['rho']
    gamma_vals, _ = compute_decorrelation_function(variance_grid, model_params)
    S_all = np.exp(price_grid[None, :] + rho * gamma_vals[:, None])

    if option_type == 'put':
        H = np.maximum(K - S_all, 0).ravel()
    else:
        H = np.maximum(S_all - K, 0).ravel()

    logger.info(f"Payoff vector built: {m*N} states, type={option_type}, K={K}")
    return H


def price_european_regular(G_combined, payoff_vector, r, T, initial_state_idx):
    total_states = G_combined.shape[0]

    logger.info(
        f"Regular pricing: matrix exp of {total_states}x{total_states} matrix..."
    )

    P_T = expm(G_combined * T)
    V_T = P_T @ payoff_vector
    price = np.exp(-r * T) * V_T[initial_state_idx]

    logger.info(f"Regular pricing result: {price:.6f}")
    return price


def price_european_fast(Q_variance, G_tensor, price_grid, variance_grid,
                         model_params, option_params, n_time_steps,
                         smooth_payoff=False):
    G_tensor = _ensure_tensor(G_tensor)
    m = G_tensor.shape[0]
    N = G_tensor.shape[1]
    T = option_params.get('T', 1.0)
    r = model_params['r']
    dt = T / n_time_steps

    logger.info(
        f"Fast pricing (tensor): {m}x{N}, {n_time_steps} steps, dt={dt:.6f}"
    )

    P_V = expm(Q_variance * dt)
    P_X = np.stack([expm(G_tensor[l] * dt) for l in range(m)])

    logger.info(f"Transition tensors pre-computed: {m}+1 exponentials")

    B = _build_payoff_matrix(price_grid, variance_grid, model_params, option_params,
                              m, N, smooth_payoff=smooth_payoff)
    discount = np.exp(-r * dt)

    use_shift = abs(model_params.get('rho', 0.0)) > 1e-14

    for z in range(n_time_steps - 1, -1, -1):
        E = np.einsum('lij,lj->li', P_X, B)
        if use_shift:
            B = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
        else:
            B = P_V @ E
        B *= discount

    from src.layer2_price import compute_initial_auxiliary_value

    X_0 = compute_initial_auxiliary_value(
        model_params['S_0'], model_params['V_0'], model_params
    )

    price = _bilinear_interpolate(B, price_grid, variance_grid, X_0,
                                   model_params['V_0'], m, N)

    logger.info(f"Fast pricing result: {price:.6f}")
    return price


def price_european_strang(Q_variance, G_tensor, price_grid, variance_grid,
                           model_params, option_params, n_time_steps,
                           smooth_payoff=False):
    G_tensor = _ensure_tensor(G_tensor)
    m = G_tensor.shape[0]
    N = G_tensor.shape[1]
    T = option_params.get('T', 1.0)
    r = model_params['r']
    dt = T / n_time_steps
    half_dt = dt / 2.0

    logger.info(
        f"Strang pricing (tensor): {m}x{N}, {n_time_steps} steps, dt={dt:.6f}"
    )

    P_V = expm(Q_variance * dt)
    P_X_half = np.stack([expm(G_tensor[l] * half_dt) for l in range(m)])

    B = _build_payoff_matrix(price_grid, variance_grid, model_params, option_params,
                              m, N, smooth_payoff=smooth_payoff)
    discount = np.exp(-r * dt)

    use_shift = abs(model_params.get('rho', 0.0)) > 1e-14

    for z in range(n_time_steps - 1, -1, -1):
        E = np.einsum('lij,lj->li', P_X_half, B)
        if use_shift:
            B = _apply_variance_shift(E, P_V, price_grid, variance_grid, model_params)
        else:
            B = P_V @ E
        B = np.einsum('lij,lj->li', P_X_half, B)
        B *= discount

    from src.layer2_price import compute_initial_auxiliary_value
    X_0 = compute_initial_auxiliary_value(
        model_params['S_0'], model_params['V_0'], model_params
    )

    price = _bilinear_interpolate(B, price_grid, variance_grid, X_0,
                                   model_params['V_0'], m, N)

    logger.info(f"Strang pricing result: {price:.6f}")
    return price


def compute_pricing_errors(ctmc_prices, benchmark_prices, strikes):
    abs_errors = np.abs(ctmc_prices - benchmark_prices)
    rel_errors = abs_errors / np.maximum(benchmark_prices, 1e-10)

    errors = {
        'MAE': np.mean(abs_errors),
        'RMSE': np.sqrt(np.mean((ctmc_prices - benchmark_prices) ** 2)),
        'MaxAE': np.max(abs_errors),
        'MeanRE': np.mean(rel_errors) * 100,
        'MaxRE': np.max(rel_errors) * 100,
    }

    logger.info("Pricing error summary:")
    logger.info(f"  MAE:    {errors['MAE']:.6f}")
    logger.info(f"  RMSE:   {errors['RMSE']:.6f}")
    logger.info(f"  MaxAE:  {errors['MaxAE']:.6f}")
    logger.info(f"  MeanRE: {errors['MeanRE']:.4f}%")
    logger.info(f"  MaxRE:  {errors['MaxRE']:.4f}%")

    return errors


def price_european_richardson(model_params, layer1_config, layer2_config,
                               option_params, n_time_steps, splitting='strang'):
    from src.grid_construction import build_variance_grid, build_price_grid
    from src.layer1_variance import construct_variance_generator
    from src.layer2_price import construct_all_regime_generators

    m_fine = layer1_config['m']
    N_fine = layer2_config['N']

    m_coarse = max(m_fine // 2, 5)
    N_coarse = max(N_fine // 2, 10)

    l1_coarse_v = dict(layer1_config, m=m_coarse)
    l2_coarse_x = dict(layer2_config, N=N_coarse)

    price_fn = price_european_strang if splitting == 'strang' else price_european_fast

    def _price(l1, l2):
        vg = build_variance_grid(l1, model_params)
        pg = build_price_grid(l2, model_params, vg)
        Q = construct_variance_generator(vg, model_params)
        Gt = construct_all_regime_generators(pg, vg, model_params, Q_variance=Q)
        return price_fn(Q, Gt, pg, vg, model_params, option_params, n_time_steps)

    p_fine = _price(layer1_config, layer2_config)
    p_cx = _price(layer1_config, l2_coarse_x)
    p_cv = _price(l1_coarse_v, layer2_config)

    ext_x = 2 * p_fine - p_cx
    ext_v = 2 * p_fine - p_cv
    ext_2d = 4 * p_fine - 2 * p_cx - 2 * p_cv

    K = option_params['K']
    S_0 = model_params['S_0']
    moneyness = K / S_0

    if moneyness < 0.92:
        price_ext = ext_x
        method = 'price_extrap'
    elif moneyness > 1.08:
        price_ext = ext_v
        method = 'var_extrap'
    else:
        w_v = min(max((moneyness - 0.85) / 0.3, 0.0), 1.0)
        price_ext = (1 - w_v) * ext_x + w_v * ext_v
        method = 'hybrid(w={:.2f})'.format(w_v)

    logger.info(
        "Richardson ({}, {}): fine={:.6f} cx={:.6f} cv={:.6f} | "
        "ext_x={:.6f} ext_v={:.6f} ext_2d={:.6f} -> {}={:.6f}".format(
            splitting, method, p_fine, p_cx, p_cv,
            ext_x, ext_v, ext_2d, method, price_ext))

    return {
        'price': price_ext,
        'price_fine': p_fine,
        'price_cx': p_cx,
        'price_cv': p_cv,
        'price_2d': ext_2d,
        'price_var_extrap': ext_v,
        'price_price_extrap': ext_x,
        'method': method,
    }
