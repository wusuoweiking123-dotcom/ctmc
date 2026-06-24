# -*- coding: utf-8 -*-
"""
文件名: src/grid_construction.py
功能描述: 自适应状态空间网格构建模块，支持均匀网格和sinh 变换非均匀网格
         Layer 1（方差维度）和Layer 2（价格维度）使用不同的网格策略
作者: [Author]
创建日期: 2026-05-06
修改历史:
    2026-05-06 - 初始版本，实现均匀和sinh 网格
"""

import numpy as np
from loguru import logger


def generate_uniform_grid(lower, upper, num_points):
    """
    生成均匀分布的状态空间网格
    参数:
        lower (float): 网格下界
        upper (float): 网格上界
        num_points (int): 网格点数量
    返回值:
        np.ndarray: 网格点数组，形状 (num_points,)
    """
    grid = np.linspace(lower, upper, num_points)
    logger.debug(
        f"Uniform grid: [{lower:.4f}, {upper:.4f}], "
        f"{num_points} points, spacing={grid[1] - grid[0]:.6f}"
    )
    return grid


def generate_sinh_grid(lower, upper, num_points, center, alpha):
    """
    使用 sinh (双曲正弦) 变换生成非均匀网格

    网格在center 附近密集，远离center 处稀疏。    基于 Tavella-Randall 非均匀网格方法 (Eq. 5.1 in Mackay et al., 2023)。
    公式:
        grid_i = center + alpha * sinh(c2 * u_i + c1 * (1 - u_i))
        其中 u_i = i / (num_points - 1), i = 0, ..., num_points-1
        c1 = arcsinh((lower - center) / alpha)
        c2 = arcsinh((upper - center) / alpha)

    参数:
        lower (float): 网格下界
        upper (float): 网格上界
        num_points (int): 网格点数量        center (float): 网格集中点（通常是初始值或长期均值）
        alpha (float): 集中度参数，alpha 越小网格越集中，alpha→∞ 时退化为均匀网格

    返回值:
        np.ndarray: 非均匀网格点数组，形状 (num_points,)
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    c1 = np.arcsinh((lower - center) / alpha)
    c2 = np.arcsinh((upper - center) / alpha)

    u = np.linspace(0, 1, num_points)
    weights = c2 * u + c1 * (1 - u)
    grid = center + alpha * np.sinh(weights)

    spacings = np.diff(grid)
    logger.debug(
        f"Sinh grid: [{lower:.4f}, {upper:.4f}], center={center:.4f}, "
        f"alpha={alpha}, spacing range=[{spacings.min():.6f}, {spacings.max():.6f}]"
    )
    return grid


def build_variance_grid(config_dict, model_params):
    """
    构建 Layer 1 方差过程的网格
    网格范围根据方差过程的长期均值theta 和初始方差V_0 确定。
    参数:
        config_dict (dict): Layer 1 网格配置，包含
            - m (int): 网格点数
            - grid_type (str): 'uniform' 或 'sinh'
            - sinh_alpha (float): sinh 集中度参数            - v_lower_factor (float): 下界系数（乘以theta）            - v_upper_factor (float): 上界系数（乘以theta）        model_params (dict): 模型参数，需包含 theta

    返回值:
        np.ndarray: 方差网格点数组，形状 (m,)
    """
    m = config_dict['m']
    theta = model_params['theta']
    V_0 = model_params['V_0']

    v_lower = config_dict.get('v_lower')
    v_upper = config_dict.get('v_upper')

    if v_lower is None:
        v_lower = config_dict.get('v_lower_abs')
    if v_lower is None:
        v_lower = theta * config_dict.get('v_lower_factor', 0.05)
    if v_upper is None:
        v_upper = config_dict.get('v_upper_abs')
    if v_upper is None:
        v_upper = theta * config_dict.get('v_upper_factor', 5.0)

    v_lower = max(v_lower, 1e-6)

    center = V_0
    grid_type = config_dict.get('grid_type', 'sinh')

    if grid_type == 'sinh':
        alpha = config_dict.get('sinh_alpha', 0.5)
        grid = generate_sinh_grid(v_lower, v_upper, m, center, alpha)
    else:
        grid = generate_uniform_grid(v_lower, v_upper, m)

    logger.info(
        f"Layer 1 variance grid built: {m} points, "
        f"range=[{grid[0]:.6f}, {grid[-1]:.6f}]"
    )
    return grid


def build_price_grid(config_dict, model_params, variance_grid):
    """
    构建 Layer 2 价格（对数价格）维度的网格
    网格范围根据当前资产价格 S_0 和方差网格的范围确定。
    参数:
        config_dict (dict): Layer 2 网格配置，包含
            - N (int): 网格点数
            - grid_type (str): 'uniform' 或 'sinh'
            - sinh_alpha (float): sinh 集中度参数        model_params (dict): 模型参数，需包含 S_0
        variance_grid (np.ndarray): Layer 1 方差网格

    返回值:
        np.ndarray: 对数价格网格点数组，形状 (N,)
    """
    N = config_dict['N']
    S_0 = model_params['S_0']
    V_0 = model_params.get('V_0', 0.04)
    T = model_params.get('T', 1.0)

    from src.layer2_price import compute_decorrelation_function
    gamma_val, _ = compute_decorrelation_function(V_0, model_params)
    rho = model_params.get('rho', 0.0)
    center = np.log(S_0) - rho * gamma_val

    x_lower = config_dict.get('x_lower')
    x_upper = config_dict.get('x_upper')

    if x_lower is None:
        max_vol = np.sqrt(variance_grid[-1])
        x_lower = center - max_vol * np.sqrt(T) * config_dict.get('x_lower_factor', 3.0)
    if x_upper is None:
        max_vol = np.sqrt(variance_grid[-1])
        x_upper = center + max_vol * np.sqrt(T) * config_dict.get('x_upper_factor', 3.0)

    grid_type = config_dict.get('grid_type', 'sinh')

    if grid_type == 'sinh':
        alpha = config_dict.get('sinh_alpha', 1.0)
        grid = generate_sinh_grid(x_lower, x_upper, N, center, alpha)
    else:
        grid = generate_uniform_grid(x_lower, x_upper, N)

    logger.info(
        f"Layer 2 price grid built: {N} points, "
        f"range=[{grid[0]:.4f}, {grid[-1]:.4f}] "
        f"(S range=[{np.exp(grid[0]):.2f}, {np.exp(grid[-1]):.2f}])"
    )
    return grid


def build_strike_locked_price_grid(config_dict, model_params, variance_grid, strikes=None):
    """
    构建行权价精确对齐的价格网格

    策略: 先生成标准 sinh 网格，然后将最近的网格点精确移动到行权价对应位置。
    保证 payoff kink (S=K) 处有网格点，消除插值误差。

    参数:
        config_dict (dict): Layer 2 网格配置
        model_params (dict): 模型参数
        variance_grid (np.ndarray): 方差网格
        strikes (list or None): 行权价列表

    返回值:
        np.ndarray: 对齐后的对数价格网格
    """
    if strikes is None or len(strikes) == 0:
        return build_price_grid(config_dict, model_params, variance_grid)

    from src.layer2_price import compute_decorrelation_function

    grid = build_price_grid(config_dict, model_params, variance_grid)

    rho = model_params.get('rho', 0.0)
    V_0 = model_params.get('V_0', 0.04)
    gamma_val, _ = compute_decorrelation_function(V_0, model_params)

    log_strikes_x = [np.log(K) - rho * gamma_val for K in strikes if K > 0]

    grid = grid.copy()
    for x_k in log_strikes_x:
        if x_k <= grid[0] or x_k >= grid[-1]:
            continue
        idx = np.argmin(np.abs(grid - x_k))
        if abs(grid[idx] - x_k) > 1e-12:
            grid[idx] = x_k

    if not np.all(np.diff(grid) > 0):
        grid = np.sort(np.unique(grid))
        logger.warning("Strike locking caused non-monotone grid, re-sorted")

    logger.info(
        f"Strike-locked price grid: {len(grid)} points, "
        f"{len(log_strikes_x)} strikes aligned, "
        f"range=[{grid[0]:.4f}, {grid[-1]:.4f}]"
    )
    return grid


def build_adaptive_price_grid(config_dict, model_params, variance_grid, strikes=None):
    """
    构建自适应价格网格 — 在行权价附近增加网格密度

    策略:
    1. 构建基础 sinh 网格 (70% 点数预算)
    2. 在每个行权价对应的 X 空间位置插入局部密集点簇
    3. 合并、排序、去重

    参数:
        config_dict (dict): Layer 2 网格配置
        model_params (dict): 模型参数
        variance_grid (np.ndarray): 方差网格
        strikes (list or None): 行权价列表，None 则退回标准网格

    返回值:
        np.ndarray: 自适应对数价格网格
    """
    if strikes is None or len(strikes) == 0:
        return build_price_grid(config_dict, model_params, variance_grid)

    from src.layer2_price import compute_decorrelation_function

    N_target = config_dict['N']
    n_base = max(int(N_target * 0.7), 20)
    base_config = dict(config_dict, N=n_base)
    base_grid = build_price_grid(base_config, model_params, variance_grid)

    rho = model_params.get('rho', 0.0)
    V_0 = model_params.get('V_0', 0.04)
    gamma_val, _ = compute_decorrelation_function(V_0, model_params)

    log_strikes_x = [np.log(K) - rho * gamma_val for K in strikes if K > 0]

    if not log_strikes_x:
        return base_grid

    x_min, x_max = base_grid[0], base_grid[-1]
    n_per_strike = max(int(N_target * 0.3 / len(log_strikes_x)), 5)
    half_width = 0.08

    extra_points = []
    for x_k in log_strikes_x:
        lo = max(x_k - half_width, x_min)
        hi = min(x_k + half_width, x_max)
        if hi <= lo:
            continue
        cluster = generate_sinh_grid(lo, hi, n_per_strike, x_k, 0.02)
        extra_points.extend(cluster.tolist())

    if not extra_points:
        return base_grid

    all_points = np.concatenate([base_grid, np.array(extra_points)])
    grid = np.sort(np.unique(np.round(all_points, 12)))

    grid = grid[(grid >= x_min) & (grid <= x_max)]

    logger.info(
        f"Adaptive price grid built: {len(grid)} points "
        f"(base={n_base} + {len(grid) - n_base} strike-aware), "
        f"{len(log_strikes_x)} strikes, "
        f"range=[{grid[0]:.4f}, {grid[-1]:.4f}]"
    )
    return grid


def compute_grid_spacings(grid):
    """
    计算网格间距数组

    参数:
        grid (np.ndarray): 网格点数组
    返回值:
        np.ndarray: 间距数组，长度为 len(grid) - 1
    """
    return np.diff(grid)


def validate_grid(grid, min_spacing=1e-10):
    """
    验证网格的有效性
    参数:
        grid (np.ndarray): 网格点数组        min_spacing (float): 最小允许间距
    返回值:
        bool: 网格是否有效
    """
    spacings = np.diff(grid)

    is_monotone = np.all(spacings > 0)
    is_finite = np.all(np.isfinite(grid))
    is_positive_spacing = np.all(spacings > min_spacing)

    if not is_monotone:
        logger.error("Grid is not strictly monotone increasing")
    if not is_finite:
        logger.error("Grid contains non-finite values")
    if not is_positive_spacing:
        logger.error(f"Grid spacing below minimum {min_spacing}")

    is_valid = is_monotone and is_finite and is_positive_spacing

    if is_valid:
        logger.debug(
            f"Grid validated: {len(grid)} points, "
            f"spacing range=[{spacings.min():.6e}, {spacings.max():.6e}]"
        )
    return is_valid
