# -*- coding: utf-8 -*-
"""
文件名: src/layer1_variance.py
功能描述: Layer 1 方差过程 CTMC 近似模块
         将连续方差过程（CIR/Heston）离散化为有限状态CTMC，构造生成元矩阵 Q^(m)
         基于 Mackay, Vachon & Cui (2023) Eq. (5.3)-(5.4)
作者: [Author]
创建日期: 2026-05-06
"""

import numpy as np
from loguru import logger


def compute_cir_coefficients(v, model_params):
    """
    计算 CIR 过程的漂移和扩散系数

    CIR 过程: dV_t = kappa*(theta - V_t)*dt + sigma_v*sqrt(V_t)*dW_t

    参数:
        v (float or np.ndarray): 方差值        model_params (dict): 模型参数，需包含 kappa, theta, sigma_v

    返回值:
        tuple: (mu_V, sigma_V_sq)
            - mu_V (float or np.ndarray): 漂移系数
            - sigma_V_sq (float or np.ndarray): 扩散系数的平方    """
    kappa = model_params['kappa']
    theta = model_params['theta']
    sigma_v = model_params['sigma_v']

    mu_V = kappa * (theta - v)
    sigma_V_sq = (sigma_v ** 2) * np.maximum(v, 0)

    return mu_V, sigma_V_sq


def _solve_5pt_rates_variance(dists, mu, sigma_sq, rho=0.0, sigma_v=1.0):
    """
    求解 Layer 1 方差过程的 5 点模板转移速率

    匹配指数矩 + 高阶多项式矩:
      Σ q_k * (exp(ρ*d_k/σ_v) - 1) = ρ*μ/σ_v + ρ²*σ²/(2σ_v²)  (指数矩)
      Σ d_k² * q_k = σ²         (二阶矩: 扩散)
      Σ d_k³ * q_k = 0           (三阶矩: 零偏度)
      Σ d_k⁴ * q_k = 0           (四阶矩: 零超额峰度)

    参数:
        dists: [d_{i-2}, d_{i-1}, d_{i+1}, d_{i+2}] — 到各邻居的距离 (有符号)
        mu: 漂移系数
        sigma_sq: 扩散系数
        rho: 相关系数 (ρ≈0 时退回多项式矩)
        sigma_v: 方差波动率

    返回值:
        list: [q_{i,i-2}, q_{i,i-1}, q_{i,i+1}, q_{i,i+2}] 或 None (无有效解)
    """
    if abs(rho) > 1e-10:
        c_v = rho * mu / sigma_v + rho ** 2 * sigma_sq / (2.0 * sigma_v ** 2)
        Ev = [np.exp(rho * d / sigma_v) - 1.0 for d in dists]
        row1 = [Ev[0], Ev[1], Ev[2], Ev[3]]
        b0 = c_v
    else:
        row1 = [dists[0], dists[1], dists[2], dists[3]]
        b0 = mu

    A = np.array([
        row1,
        [dists[0]**2, dists[1]**2, dists[2]**2, dists[3]**2],
        [dists[0]**3, dists[1]**3, dists[2]**3, dists[3]**3],
        [dists[0]**4, dists[1]**4, dists[2]**4, dists[3]**4],
    ])
    b = np.array([b0, sigma_sq, 0.0, 0.0])

    try:
        rates = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    if np.all(rates >= -1e-12):
        rates = np.maximum(rates, 0.0)
        return rates.tolist()
    return None


def _solve_3pt_exp_rates_variance(dl, dr, mu_V, sigma_V_sq, rho, sigma_v):
    ds = dl + dr
    q_l = (sigma_V_sq - dr * mu_V) / (dl * ds)
    q_r = (sigma_V_sq + dl * mu_V) / (dr * ds)

    if abs(rho) > 1e-10:
        Ev_l = np.exp(-rho * dl / sigma_v) - 1.0
        Ev_r = np.exp(rho * dr / sigma_v) - 1.0
        c_v = rho * mu_V / sigma_v + rho ** 2 * sigma_V_sq / (2.0 * sigma_v ** 2)
        eps = q_l * Ev_l + q_r * Ev_r - c_v
        denom = Ev_l ** 2 + Ev_r ** 2
        if abs(denom) > 1e-30:
            q_l -= eps * Ev_l / denom
            q_r -= eps * Ev_r / denom

    if q_l < 0 or q_r < 0:
        q_l = (sigma_V_sq - dr * mu_V) / (dl * ds)
        q_r = (sigma_V_sq + dl * mu_V) / (dr * ds)

    return max(q_l, 0.0), max(q_r, 0.0)


def construct_variance_generator(grid, model_params, stencil='auto'):
    """
    构造Layer 1 方差过程的CTMC 生成元矩阵Q^(m)

    stencil 选项:
      '3pt'  — 经典 3 点模板 (匹配漂移+扩散)
      '5pt'  — 5 点模板 (匹配前 4 阶矩)
      'auto' — 优先 5 点，无有效解时退回 3 点

    参数:
        grid (np.ndarray): 方差状态空间网格，形状 (m,)
        model_params (dict): 模型参数
        stencil (str): 模板类型

    返回值:
        np.ndarray: 生成元矩阵Q^(m)，形状(m, m)
    """
    m = len(grid)
    if m < 3:
        raise ValueError(f"Need at least 3 grid points, got {m}")

    rho = model_params.get('rho', 0.0)
    sigma_v = model_params.get('sigma_v', 1.0)

    deltas = np.diff(grid)
    Q = np.zeros((m, m))

    mu_V_0, sigma_V_sq_0 = compute_cir_coefficients(grid[0], model_params)

    if abs(rho) > 1e-10:
        cv_tgt_0 = rho * mu_V_0 / sigma_v + rho**2 * sigma_V_sq_0 / (2.0 * sigma_v**2)
        exp_fac_0 = np.exp(rho * deltas[0] / sigma_v) - 1.0
        if abs(exp_fac_0) > 1e-30 and cv_tgt_0 / exp_fac_0 > 0:
            Q[0, 1] = cv_tgt_0 / exp_fac_0
        else:
            Q[0, 1] = max(sigma_V_sq_0 / deltas[0] ** 2 + max(mu_V_0, 0) / deltas[0], 1e-12)
    else:
        Q[0, 1] = max(sigma_V_sq_0 / deltas[0] ** 2 + max(mu_V_0, 0) / deltas[0], 1e-12)
    Q[0, 0] = -Q[0, 1]

    mu_V_1, sigma_V_sq_1 = compute_cir_coefficients(grid[1], model_params)
    Q[1, 0], Q[1, 2] = _solve_3pt_exp_rates_variance(
        deltas[0], deltas[1], mu_V_1, sigma_V_sq_1, rho, sigma_v)
    Q[1, 1] = -(Q[1, 0] + Q[1, 2])

    mu_V_m, sigma_V_sq_m = compute_cir_coefficients(grid[-1], model_params)

    if abs(rho) > 1e-10:
        cv_tgt_m = rho * mu_V_m / sigma_v + rho**2 * sigma_V_sq_m / (2.0 * sigma_v**2)
        exp_fac_m = np.exp(-rho * deltas[-1] / sigma_v) - 1.0
        if abs(exp_fac_m) > 1e-30 and cv_tgt_m / exp_fac_m > 0:
            Q[m - 1, m - 2] = cv_tgt_m / exp_fac_m
        else:
            Q[m - 1, m - 2] = max(
                sigma_V_sq_m / deltas[-1] ** 2 + max(-mu_V_m, 0) / deltas[-1], 1e-12)
    else:
        Q[m - 1, m - 2] = max(
            sigma_V_sq_m / deltas[-1] ** 2 + max(-mu_V_m, 0) / deltas[-1], 1e-12)
    Q[m - 1, m - 1] = -Q[m - 1, m - 2]

    mu_V_m1, sigma_V_sq_m1 = compute_cir_coefficients(grid[-2], model_params)
    Q[m - 2, m - 3], Q[m - 2, m - 1] = _solve_3pt_exp_rates_variance(
        deltas[-2], deltas[-1], mu_V_m1, sigma_V_sq_m1, rho, sigma_v)
    Q[m - 2, m - 2] = -(Q[m - 2, m - 3] + Q[m - 2, m - 1])

    use_5pt = stencil in ('5pt', 'auto')
    fallback_count = 0

    for i in range(2, m - 2):
        mu_V_i, sigma_V_sq_i = compute_cir_coefficients(grid[i], model_params)

        applied_5pt = False
        if use_5pt:
            dists = [
                grid[i - 2] - grid[i],
                grid[i - 1] - grid[i],
                grid[i + 1] - grid[i],
                grid[i + 2] - grid[i],
            ]
            rates = _solve_5pt_rates_variance(dists, mu_V_i, sigma_V_sq_i, rho, sigma_v)
            if rates is not None:
                Q[i, i - 2] = rates[0]
                Q[i, i - 1] = rates[1]
                Q[i, i + 1] = rates[2]
                Q[i, i + 2] = rates[3]
                Q[i, i] = -(Q[i, i - 2] + Q[i, i - 1] + Q[i, i + 1] + Q[i, i + 2])
                applied_5pt = True
            else:
                fallback_count += 1

        if not applied_5pt:
            Q[i, i - 1], Q[i, i + 1] = _solve_3pt_exp_rates_variance(
                deltas[i - 1], deltas[i], mu_V_i, sigma_V_sq_i, rho, sigma_v)
            Q[i, i] = -(Q[i, i - 1] + Q[i, i + 1])

    if use_5pt:
        logger.debug(
            "Layer 1 generator: stencil={}, 5pt interior: {}/{}, fallbacks: {}".format(
                stencil, m - 4 - fallback_count, m - 4, fallback_count))

    logger.info(
        f"Layer 1 generator Q^(m) built: shape={Q.shape}, "
        f"off-diag range=[{Q[Q > 0].min():.6f}, {Q[Q > 0].max():.6f}]"
    )

    _validate_generator(Q, 'Layer 1 variance')

    return Q


def compute_v_generator_exp_moment(Q, variance_grid, model_params):
    """
    计算 V 生成元 Q 在每个方差状态 l 处的实际指数矩 cv_actual(l)

    理论目标: cv_target(l) = rho*kappa*(theta-v_l)/sigma_v + rho^2*v_l/2
    实际值:   cv_actual(l) = sum_k Q[l,k] * exp(rho*(v_k - v_l)/sigma_v)

    指数矩误差: epsilon_v(l) = cv_actual(l) - cv_target(l)

    参数:
        Q (np.ndarray): Layer 1 方差生成元, 形状 (m, m)
        variance_grid (np.ndarray): 方差网格, 形状 (m,)
        model_params (dict): 模型参数

    返回值:
        tuple: (cv_actual, cv_target, epsilon_v)
            - cv_actual (np.ndarray): 实际指数矩, 形状 (m,)
            - cv_target (np.ndarray): 理论指数矩, 形状 (m,)
            - epsilon_v (np.ndarray): 误差, 形状 (m,)
    """
    rho = model_params.get('rho', 0.0)
    sigma_v = model_params.get('sigma_v', 1.0)
    kappa = model_params['kappa']
    theta = model_params['theta']

    m = len(variance_grid)

    f_V = np.exp(rho * variance_grid / sigma_v)

    Qf = Q @ f_V
    cv_actual = Qf / f_V

    cv_target = rho * kappa * (theta - variance_grid) / sigma_v + rho**2 * variance_grid / 2.0

    epsilon_v = cv_actual - cv_target

    logger.info(
        f"V-generator exp moment: epsilon_v range=[{epsilon_v.min():.6e}, {epsilon_v.max():.6e}], "
        f"|max|={np.abs(epsilon_v).max():.6e}"
    )

    if np.abs(epsilon_v).max() > 1e-6:
        worst_l = np.argmax(np.abs(epsilon_v))
        logger.debug(
            f"  Worst regime l={worst_l}, v_l={variance_grid[worst_l]:.6f}, "
            f"cv_actual={cv_actual[worst_l]:.6f}, cv_target={cv_target[worst_l]:.6f}, "
            f"epsilon={epsilon_v[worst_l]:.6e}"
        )

    return cv_actual, cv_target, epsilon_v


def compute_variance_stationary_distribution(Q, grid, model_params):
    """
    计算方差过程的平稳分布（用于网格范围验证）
    CIR 过程的平稳分布为 Gamma 分布:
        V ~ Gamma(2*kappa*theta/sigma_v^2, sigma_v^2/(2*kappa))

    参数:
        Q (np.ndarray): 生成元矩阵（未使用，保留接口一致性）
        grid (np.ndarray): 方差网格
        model_params (dict): 模型参数

    返回值:
        np.ndarray: 平稳分布概率向量
    """
    kappa = model_params['kappa']
    theta = model_params['theta']
    sigma_v = model_params['sigma_v']

    from scipy.stats import gamma as gamma_dist

    shape_param = 2 * kappa * theta / (sigma_v ** 2)
    scale_param = sigma_v ** 2 / (2 * kappa)

    probs = gamma_dist.pdf(grid, a=shape_param, scale=scale_param)
    total = np.sum(probs)
    if total > 0:
        probs = probs / total

    logger.debug(
        f"Stationary distribution: peak at grid[{np.argmax(probs)}]={grid[np.argmax(probs)]:.6f}"
    )
    return probs


def get_variance_state_index(value, grid):
    """
    在方差网格中查找最近邻状态的索引

    参数:
        value (float): 方差值        grid (np.ndarray): 方差网格

    返回值:
        int: 最近邻状态的索引
    """
    idx = int(np.argmin(np.abs(grid - value)))
    logger.debug(f"Variance state index: V={value:.6f} -> index={idx}, grid[{idx}]={grid[idx]:.6f}")
    return idx


def _validate_generator(Q, label='generator'):
    """
    验证生成元矩阵是否满足基本性质

    参数:
        Q (np.ndarray): 生成元矩阵        label (str): 标签（用于日志）

    返回值:
        bool: 是否有效
    """
    diag_valid = np.all(np.diag(Q) <= 1e-10)
    off_diag = Q.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_valid = np.all(off_diag >= -1e-10)
    row_sums = np.abs(Q.sum(axis=1))
    row_sum_valid = np.all(row_sums < 1e-8)

    is_valid = diag_valid and off_diag_valid and row_sum_valid

    if not is_valid:
        logger.warning(f"{label} generator validation FAILED")
        if not diag_valid:
            logger.warning(f"  Positive diagonal elements detected")
        if not off_diag_valid:
            logger.warning(f"  Negative off-diagonal elements detected")
        if not row_sum_valid:
            logger.warning(f"  Max row-sum deviation: {row_sums.max():.2e}")
    else:
        logger.debug(f"{label} generator validated OK")

    return is_valid
