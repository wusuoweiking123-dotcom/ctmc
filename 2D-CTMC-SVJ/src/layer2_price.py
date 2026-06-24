# -*- coding: utf-8 -*-
"""
文件名: src/layer2_price.py
功能描述: Layer 2 辅助过程 CTMC 近似模块
         基于 Heston 模型的去相关变换，构造体制依赖的生成元矩阵G_l^(N)
         通过辅助变量 X = ln(S) - rho*gamma(V) 实现 (X, V) 的独立性
         基于 Mackay, Vachon & Cui (2023) Eq. (6.1)-(6.9)
作者: [Author]
创建日期: 2026-05-06
"""

import numpy as np
from loguru import logger


def _cost_of_carry(model_params):
    """
    计算持有成本 (cost of carry)

    - 'spot':   b = r  (现货期权, dS/S = r dt + ...)
    - 'futures': b = 0  (期货期权, dF/F = 0 dt + ...)
    """
    if model_params.get('underlying_type', 'spot') == 'futures':
        return 0.0
    return model_params['r']


def compute_decorrelation_function(v, model_params):
    """
    计算 Heston 模型的去相关函数 gamma(v) 及其导数

    对于 Heston 模型:
        sigma_S(v) = sqrt(v), sigma_V(v) = sigma_v * sqrt(v)
        gamma(v) = integral of sigma_S/sigma_V dv = v / sigma_v

    参数:
        v (float or np.ndarray): 方差值        model_params (dict): 模型参数，需包含 sigma_v

    返回值:
        tuple: (gamma_val, gamma_prime)
            - gamma_val: gamma(v) 的值
            - gamma_prime: gamma'(v) 的值    """
    sigma_v = model_params['sigma_v']

    gamma_val = v / sigma_v
    gamma_prime = 1.0 / sigma_v

    return gamma_val, gamma_prime


def compute_auxiliary_coefficients(x, v, model_params):
    """
    计算辅助过程 X 的漂移和扩散系数

    辅助过程 (Eq. 6.3):
        dX_t = mu_X(X_t, V_t) dt + sigma_X(V_t) dW*_t
        其中 W* 独立于驱动的V 的布朗运动
    对于 Heston 模型:
        sigma_X(v) = sqrt(1 - rho^2) * sqrt(v)
        mu_X(x, v) = b - v/2 - rho * psi(v)   (b=r for spot, b=0 for futures)
        psi(v) = kappa*(theta-v)/sigma_v

    参数:
        x (float or np.ndarray): 对数价格（辅助变量）
        v (float or np.ndarray): 方差
        model_params (dict): 模型参数

    返回值:
        tuple: (mu_X, sigma_X_sq)
            - mu_X: 辅助过程的漂移
            - sigma_X_sq: 辅助过程扩散系数的平方
    """
    b = _cost_of_carry(model_params)
    rho = model_params['rho']
    kappa = model_params['kappa']
    theta = model_params['theta']
    sigma_v = model_params['sigma_v']

    psi_v = kappa * (theta - v) / sigma_v

    mu_X = b - v / 2 - rho * psi_v
    sigma_X_sq = (1 - rho ** 2) * np.maximum(v, 0)

    return mu_X, sigma_X_sq


def _solve_5pt_rates(dists, mu, sigma_sq):
    """
    求解 5 点模板的转移速率

    匹配 4 阶条件:
      Σ d_k * q_k = μ           (一阶矩: 漂移)
      Σ d_k² * q_k = σ²         (二阶矩: 扩散)
      Σ d_k³ * q_k = 0           (三阶矩: 零偏度)
      Σ d_k⁴ * q_k = 0           (四阶矩: 零超额峰度)

    参数:
        dists: [d_{i-2}, d_{i-1}, d_{i+1}, d_{i+2}] — 到各邻居的距离 (有符号)
        mu: 漂移系数
        sigma_sq: 扩散系数

    返回值:
        list: [q_{i,i-2}, q_{i,i-1}, q_{i,i+1}, q_{i,i+2}] 或 None (无有效解)
    """
    E_d = [np.exp(d) - 1.0 for d in dists]
    A = np.array([
        [E_d[0], E_d[1], E_d[2], E_d[3]],
        [dists[0]**2, dists[1]**2, dists[2]**2, dists[3]**2],
        [dists[0]**3, dists[1]**3, dists[2]**3, dists[3]**3],
        [dists[0]**4, dists[1]**4, dists[2]**4, dists[3]**4],
    ])
    b = np.array([mu + sigma_sq / 2.0, sigma_sq, 0.0, 0.0])

    try:
        rates = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    if np.all(rates >= -1e-12):
        rates = np.maximum(rates, 0.0)
        return rates.tolist()
    return None


def _solve_3pt_exp_rates(dl, dr, mu, sigma_sq):
    ds = dl + dr
    q_l = (sigma_sq - dr * mu) / (dl * ds)
    q_r = (sigma_sq + dl * mu) / (dr * ds)

    E_l = np.exp(-dl) - 1.0
    E_r = np.exp(dr) - 1.0
    c_exp = mu + sigma_sq / 2.0
    eps = q_l * E_l + q_r * E_r - c_exp
    denom = E_l ** 2 + E_r ** 2
    if abs(denom) > 1e-30:
        q_l -= eps * E_l / denom
        q_r -= eps * E_r / denom

    if q_l < 0 or q_r < 0:
        q_l = (sigma_sq - dr * mu) / (dl * ds)
        q_r = (sigma_sq + dl * mu) / (dr * ds)

    return max(q_l, 0.0), max(q_r, 0.0)


def construct_regime_generator(price_grid, v_l, model_params, stencil='auto'):
    """
    构造Layer 2 在特定方差体制v_l 下的生成元矩阵G_l^(N)

    stencil 选项:
      '3pt'  — 经典 3 点模板 (匹配漂移+扩散)
      '5pt'  — 5 点模板 (匹配前 4 阶矩)
      'auto' — 优先 5 点，无有效解时退回 3 点

    参数:
        price_grid (np.ndarray): 对数价格状态空间网格，形状 (N,)
        v_l (float): 当前方差体制值
        model_params (dict): 模型参数
        stencil (str): 模板类型

    返回值:
        np.ndarray: 体制 l 下的生成元矩阵G_l^(N)，形状(N, N)
    """
    N = len(price_grid)
    if N < 3:
        raise ValueError(f"Need at least 3 price grid points, got {N}")

    deltas = np.diff(price_grid)
    G = np.zeros((N, N))

    mu_X_0, sigma_X_sq_0 = compute_auxiliary_coefficients(price_grid[0], v_l, model_params)
    q_01 = sigma_X_sq_0 / deltas[0] ** 2 + max(mu_X_0, 0) / deltas[0]
    G[0, 1] = max(q_01, 1e-12)
    G[0, 0] = -G[0, 1]

    mu_X_1, sigma_X_sq_1 = compute_auxiliary_coefficients(price_grid[1], v_l, model_params)
    G[1, 0], G[1, 2] = _solve_3pt_exp_rates(deltas[0], deltas[1], mu_X_1, sigma_X_sq_1)
    G[1, 1] = -(G[1, 0] + G[1, 2])

    mu_X_N, sigma_X_sq_N = compute_auxiliary_coefficients(price_grid[-1], v_l, model_params)
    q_N_N1 = sigma_X_sq_N / deltas[-1] ** 2 + max(-mu_X_N, 0) / deltas[-1]
    G[N - 1, N - 2] = max(q_N_N1, 1e-12)
    G[N - 1, N - 1] = -G[N - 1, N - 2]

    mu_X_N1, sigma_X_sq_N1 = compute_auxiliary_coefficients(price_grid[-2], v_l, model_params)
    G[N - 2, N - 3], G[N - 2, N - 1] = _solve_3pt_exp_rates(
        deltas[-2], deltas[-1], mu_X_N1, sigma_X_sq_N1)
    G[N - 2, N - 2] = -(G[N - 2, N - 3] + G[N - 2, N - 1])

    use_5pt = stencil in ('5pt', 'auto')
    fallback_count = 0

    for i in range(2, N - 2):
        mu_X_i, sigma_X_sq_i = compute_auxiliary_coefficients(
            price_grid[i], v_l, model_params
        )

        applied_5pt = False
        if use_5pt:
            dists = [
                price_grid[i - 2] - price_grid[i],
                price_grid[i - 1] - price_grid[i],
                price_grid[i + 1] - price_grid[i],
                price_grid[i + 2] - price_grid[i],
            ]
            rates = _solve_5pt_rates(dists, mu_X_i, sigma_X_sq_i)
            if rates is not None:
                G[i, i - 2] = rates[0]
                G[i, i - 1] = rates[1]
                G[i, i + 1] = rates[2]
                G[i, i + 2] = rates[3]
                G[i, i] = -(G[i, i - 2] + G[i, i - 1] + G[i, i + 1] + G[i, i + 2])
                applied_5pt = True
            else:
                fallback_count += 1

        if not applied_5pt:
            G[i, i - 1], G[i, i + 1] = _solve_3pt_exp_rates(
                deltas[i - 1], deltas[i], mu_X_i, sigma_X_sq_i)
            G[i, i] = -(G[i, i - 1] + G[i, i + 1])

    if use_5pt:
        logger.debug(
            "Layer 2 generator: stencil={}, 5pt interior points: {}/{}, fallbacks: {}".format(
                stencil, N - 4 - fallback_count, N - 4, fallback_count))

    return G


def construct_regime_generator_with_drift(price_grid, v_l, model_params,
                                           drift_correction, stencil='auto'):
    """
    构造带额外漂移修正的Layer 2 体制生成元
    用于 SVJ 模型: mu_X_SVJ = mu_X_SV + drift_correction
    其中 drift_correction = -lambda * k_bar (跳跃补偿漂移修正)

    stencil 选项:
      '3pt'  — 经典 3 点模板
      '5pt'  — 5 点模板 (匹配前 4 阶矩)
      'auto' — 优先 5 点，无有效解时退回 3 点

    参数:
        price_grid (np.ndarray): 对数价格网格, 形状 (N,)
        v_l (float): 当前方差体制值
        model_params (dict): 模型参数
        drift_correction (float): 额外漂移修正值
        stencil (str): 模板类型
    返回值:
        np.ndarray: 带漂移修正的体制生成元矩阵 形状 (N, N)
    """
    N = len(price_grid)
    if N < 3:
        raise ValueError(f"Need at least 3 price grid points, got {N}")

    b = _cost_of_carry(model_params)
    rho = model_params['rho']
    kappa = model_params['kappa']
    theta = model_params['theta']
    sigma_v = model_params['sigma_v']

    psi_v = kappa * (theta - v_l) / sigma_v
    mu_X = b - v_l / 2 - rho * psi_v + drift_correction
    sigma_X_sq = (1 - rho ** 2) * max(v_l, 0)

    deltas = np.diff(price_grid)
    G = np.zeros((N, N))

    G[0, 1] = max(sigma_X_sq / deltas[0] ** 2 + max(mu_X, 0) / deltas[0], 1e-12)
    G[0, 0] = -G[0, 1]

    G[1, 0], G[1, 2] = _solve_3pt_exp_rates(deltas[0], deltas[1], mu_X, sigma_X_sq)
    G[1, 1] = -(G[1, 0] + G[1, 2])

    G[N - 1, N - 2] = max(sigma_X_sq / deltas[-1] ** 2 + max(-mu_X, 0) / deltas[-1], 1e-12)
    G[N - 1, N - 1] = -G[N - 1, N - 2]

    G[N - 2, N - 3], G[N - 2, N - 1] = _solve_3pt_exp_rates(
        deltas[-2], deltas[-1], mu_X, sigma_X_sq)
    G[N - 2, N - 2] = -(G[N - 2, N - 3] + G[N - 2, N - 1])

    use_5pt = stencil in ('5pt', 'auto')
    fallback_count = 0

    for i in range(2, N - 2):
        applied_5pt = False
        if use_5pt:
            dists = [
                price_grid[i - 2] - price_grid[i],
                price_grid[i - 1] - price_grid[i],
                price_grid[i + 1] - price_grid[i],
                price_grid[i + 2] - price_grid[i],
            ]
            rates = _solve_5pt_rates(dists, mu_X, sigma_X_sq)
            if rates is not None:
                G[i, i - 2] = rates[0]
                G[i, i - 1] = rates[1]
                G[i, i + 1] = rates[2]
                G[i, i + 2] = rates[3]
                G[i, i] = -(G[i, i - 2] + G[i, i - 1] + G[i, i + 1] + G[i, i + 2])
                applied_5pt = True
            else:
                fallback_count += 1

        if not applied_5pt:
            G[i, i - 1], G[i, i + 1] = _solve_3pt_exp_rates(
                deltas[i - 1], deltas[i], mu_X, sigma_X_sq)
            G[i, i] = -(G[i, i - 1] + G[i, i + 1])

    if use_5pt:
        logger.debug(
            "Layer 2 drift generator: stencil={}, 5pt: {}/{}, fallbacks: {}".format(
                stencil, N - 4 - fallback_count, N - 4, fallback_count))

    return G


def construct_all_regime_generators(price_grid, variance_grid, model_params,
                                     stencil='auto', Q_variance=None):
    """
    构造所有方差体制下的Layer 2 生成元矩阵
    对方差网格中的每个状态v_l (l=1,...,m)，分别构造一个N×N 的生成元矩阵。

    当 Q_variance 非空时，计算 V 生成元的指数矩误差 epsilon_v(l)，
    并用 drift_correction = -epsilon_v(l) 修正 X 生成元的漂移，
    使得组合生成元满足鞅条件: cv_actual(l) + cx_corrected(l,i) = r

    参数:
        price_grid (np.ndarray): 对数价格网格，形状(N,)
        variance_grid (np.ndarray): 方差网格，形状(m,)
        model_params (dict): 模型参数
        stencil (str): 模板类型
        Q_variance (np.ndarray or None): Layer 1 方差生成元，形状 (m, m)

    返回值:
        list: 包含 m 个生成元矩阵的列表，每个形状为(N, N)
    """
    m = len(variance_grid)
    generators = []

    if Q_variance is not None:
        from src.layer1_variance import compute_v_generator_exp_moment
        _, _, epsilon_v = compute_v_generator_exp_moment(
            Q_variance, variance_grid, model_params
        )
        n_corrected = np.sum(np.abs(epsilon_v) > 1e-14)
        logger.info(
            f"Drift correction: {n_corrected}/{m} regimes corrected, "
            f"epsilon_v range=[{epsilon_v.min():.6e}, {epsilon_v.max():.6e}]"
        )
    else:
        epsilon_v = np.zeros(m)

    for l in range(m):
        if abs(epsilon_v[l]) > 1e-14:
            G_l = construct_regime_generator_with_drift(
                price_grid, variance_grid[l], model_params,
                drift_correction=-epsilon_v[l], stencil=stencil
            )
        else:
            G_l = construct_regime_generator(
                price_grid, variance_grid[l], model_params, stencil=stencil
            )
        generators.append(G_l)

    generators = np.stack(generators)

    logger.info(
        f"Layer 2 generators built: tensor {generators.shape}"
    )
    return generators


def recover_price_from_auxiliary(x, v, model_params):
    """
    从辅助变量恢复原始资产价格
    S = exp(X + rho * gamma(V))

    对于 Heston: S = exp(X + rho * V / sigma_v)

    参数:
        x (float or np.ndarray): 辅助变量值        v (float or np.ndarray): 方差值        model_params (dict): 模型参数

    返回值:
        float or np.ndarray: 恢复的资产价格    """
    rho = model_params['rho']
    gamma_val, _ = compute_decorrelation_function(v, model_params)
    return np.exp(x + rho * gamma_val)


def get_price_state_index(log_price, price_grid):
    """
    在对数价格网格中查找最近邻状态的索引

    参数:
        log_price (float): 对数价格值        price_grid (np.ndarray): 对数价格网格

    返回值:
        int: 最近邻状态的索引
    """
    return int(np.argmin(np.abs(price_grid - log_price)))


def compute_initial_auxiliary_value(S_0, V_0, model_params):
    """
    计算初始辅助变量值
    X_0 = ln(S_0) - rho * gamma(V_0)

    参数:
        S_0 (float): 初始资产价格
        V_0 (float): 初始方差
        model_params (dict): 模型参数

    返回值:
        float: 初始辅助变量值X_0
    """
    gamma_val, _ = compute_decorrelation_function(V_0, model_params)
    rho = model_params['rho']
    X_0 = np.log(S_0) - rho * gamma_val

    logger.debug(f"Initial auxiliary: X_0={X_0:.6f}, S_0={S_0:.4f}, V_0={V_0:.6f}")
    return X_0
