# -*- coding: utf-8 -*-
"""
文件名: src/jump_generator.py
功能描述: 体制依赖的跳跃生成元模块
         将Layer 2 体制生成元添加跳跃分量Lambda_J
         支持 Merton（对数正态）和Kou（双指数）跳跃规格
         基于 Eq. (3.1)-(3.4): 跳跃仅影响资产价格，不影响方差过程
作者: [Author]
创建日期: 2026-05-07
"""

import numpy as np
from scipy.stats import norm
from loguru import logger


def compute_merton_compensator(mu_J, sigma_J):
    """
    计算 Merton 跳跃的补偿器 k_bar = E[e^Z - 1]

    对于 Z ~ N(mu_J, sigma_J^2):
        k_bar = exp(mu_J + sigma_J^2/2) - 1

    参数:
        mu_J (float): 对数跳跃大小的均值        sigma_J (float): 对数跳跃大小的标准差

    返回值:
        float: 补偿器k_bar
    """
    return np.exp(mu_J + sigma_J ** 2 / 2) - 1


def compute_kou_compensator(p, eta1, eta2):
    """
    计算 Kou 双指数跳跃的补偿器k_bar = E[e^Z - 1]

    对于 Z ~ DoubleExponential(p, eta1, eta2):
        k_bar = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1

    参数:
        p (float): 上涨概率
        eta1 (float): 上涨指数参数 (>1)
        eta2 (float): 下跌指数参数 (>0)

    返回值:
        float: 补偿器k_bar
    """
    return p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1


def construct_merton_jump_matrix(price_grid, lambda_jump, mu_J, sigma_J):
    """
    构造Merton 对数正态跳跃的跳跃生成元矩阵Lambda_J

    对于对数价格网格 {x_1, ..., x_N}，从状态x_i 跳跃到状态x_j
    对应对数跳跃大小 Z = x_j - x_i。
    通过在对数跳跃大小的 bins 上积分正态密度计算跳跃速率:
        Lambda_J[i,j] = lambda_jump * [Phi((z_hi - mu_J)/sigma_J)
                                     - Phi((z_lo - mu_J)/sigma_J)]

    其中 z_lo, z_hi 是bin j 相对于源状态i 的边界。
    参数:
        price_grid (np.ndarray): 对数价格网格, 形状 (N,)
        lambda_jump (float): 泊松跳跃强度
        mu_J (float): 对数跳跃大小的均值        sigma_J (float): 对数跳跃大小的标准差

    返回值:
        np.ndarray: 跳跃矩阵 Lambda_J, 形状 (N, N)
    """
    N = len(price_grid)

    bin_lower = np.zeros(N)
    bin_upper = np.zeros(N)

    bin_lower[0] = price_grid[0] - (price_grid[1] - price_grid[0])
    bin_upper[N - 1] = price_grid[N - 1] + (price_grid[N - 1] - price_grid[N - 2])

    for j in range(1, N):
        bin_lower[j] = (price_grid[j - 1] + price_grid[j]) / 2
    for j in range(N - 1):
        bin_upper[j] = (price_grid[j] + price_grid[j + 1]) / 2

    Lambda_J = np.zeros((N, N))

    for i in range(N):
        z_lo = bin_lower - price_grid[i]
        z_hi = bin_upper - price_grid[i]

        p_lo = norm.cdf((z_lo - mu_J) / sigma_J)
        p_hi = norm.cdf((z_hi - mu_J) / sigma_J)

        Lambda_J[i, :] = lambda_jump * (p_hi - p_lo)

    np.fill_diagonal(Lambda_J, 0)

    row_sums = Lambda_J.sum(axis=1)
    Lambda_J = Lambda_J - np.diag(row_sums)

    logger.debug(
        f"Merton jump matrix built: shape=({N},{N}), "
        f"off-diag range=[{Lambda_J[Lambda_J > 0].min():.6f}, "
        f"{Lambda_J.max():.6f}], "
        f"max_exit_rate={row_sums.max():.6f}"
    )

    return Lambda_J


def construct_kou_jump_matrix(price_grid, lambda_jump, p, eta1, eta2):
    """
    构造Kou 双指数跳跃的跳跃生成元矩阵Lambda_J

    使用 bin 中心比率方法，参考1D-CTMC 项目中的实现模式。
    参数:
        price_grid (np.ndarray): 对数价格网格, 形状 (N,)
        lambda_jump (float): 泊松跳跃强度
        p (float): 上涨跳跃概率
        eta1 (float): 上涨指数参数 (>1)
        eta2 (float): 下跌指数参数 (>0)

    返回值:
        np.ndarray: 跳跃矩阵 Lambda_J, 形状 (N, N)
    """
    N = len(price_grid)

    bin_lower = np.zeros(N)
    bin_upper = np.zeros(N)

    bin_lower[0] = price_grid[0] - (price_grid[1] - price_grid[0])
    bin_upper[N - 1] = price_grid[N - 1] + (price_grid[N - 1] - price_grid[N - 2])

    for j in range(1, N):
        bin_lower[j] = (price_grid[j - 1] + price_grid[j]) / 2
    for j in range(N - 1):
        bin_upper[j] = (price_grid[j] + price_grid[j + 1]) / 2

    Lambda_J = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            y_lo = np.exp(bin_lower[j] - price_grid[i]) - 1
            y_hi = np.exp(bin_upper[j] - price_grid[i]) - 1

            rate = 0.0

            if y_lo >= 0:
                rate = p * eta1 * (np.exp(-eta1 * y_lo) - np.exp(-eta1 * y_hi))
            elif y_hi <= 0:
                rate = (1 - p) * eta2 * (np.exp(eta2 * y_hi) - np.exp(eta2 * y_lo))
            else:
                rate = (p * (1 - np.exp(-eta1 * y_hi))
                        + (1 - p) * (np.exp(eta2 * y_lo) - 1))

            Lambda_J[i, j] = lambda_jump * max(rate, 0)

    row_sums = Lambda_J.sum(axis=1)
    Lambda_J = Lambda_J - np.diag(row_sums)

    logger.debug(f"Kou jump matrix built: shape=({N},{N})")

    return Lambda_J


def compute_jump_drift_correction(jump_params):
    """
    计算跳跃对辅助过程漂移的修正值
    SVJ 模型下(Eq. 3.2, 6.6):
        mu_X_SVJ(x, v) = mu_X_SV(x, v) - lambda * k_bar

    参数:
        jump_params (dict): 跳跃参数, 需包含 jump_type 和相应参数
    返回值:
        float: 漂移修正值，-lambda * k_bar
    """
    lambda_jump = jump_params.get('lambda_jump', 0)
    if lambda_jump == 0:
        return 0.0

    jump_type = jump_params.get('jump_type', 'merton')

    if jump_type == 'merton':
        k_bar = compute_merton_compensator(
            jump_params['mu_J'], jump_params['sigma_J']
        )
    elif jump_type == 'kou':
        k_bar = compute_kou_compensator(
            jump_params['p'], jump_params['eta1'], jump_params['eta2']
        )
    else:
        raise ValueError(f"Unknown jump type: {jump_type}")

    return -lambda_jump * k_bar


def construct_jump_matrix(price_grid, jump_params):
    """
    根据跳跃类型分派构造跳跃矩阵
    参数:
        price_grid (np.ndarray): 对数价格网格
        jump_params (dict): 跳跃参数

    返回值:
        np.ndarray: 跳跃矩阵, 形状 (N, N)
    """
    jump_type = jump_params.get('jump_type', 'merton')
    lambda_jump = jump_params.get('lambda_jump', 0)

    if lambda_jump == 0:
        N = len(price_grid)
        return np.zeros((N, N))

    if jump_type == 'merton':
        return construct_merton_jump_matrix(
            price_grid, lambda_jump, jump_params['mu_J'], jump_params['sigma_J']
        )
    elif jump_type == 'kou':
        return construct_kou_jump_matrix(
            price_grid, lambda_jump,
            jump_params['p'], jump_params['eta1'], jump_params['eta2']
        )
    else:
        raise ValueError(f"Unknown jump type: {jump_type}")


def add_jump_to_regime_generator(G_diffusion, price_grid, jump_params):
    """
    为单个体制的扩散生成元添加跳跃分量
    G_l^(SVJ) = G_l^(SV, drift-corrected) + Lambda_J

    方案A: 跳跃的漂移修正直接嵌入生成元的漂移中, 不通过 FD 近似。
    即先修改三对角生成元的mu_X, 再叠加跳跃矩阵。
    参数:
        G_diffusion (np.ndarray): 纯SV 体制生成元，形状 (N, N)
        price_grid (np.ndarray): 对数价格网格
        jump_params (dict): 跳跃参数

    返回值:
        np.ndarray: 含跳跃的体制生成元，形状 (N, N)

    注意: 此函数仅叠加 Lambda_J, 漂移修正由调用方通过
          construct_regime_generator_svj 在构造G_diffusion 时完成。    """
    Lambda_J = construct_jump_matrix(price_grid, jump_params)

    G_svj = G_diffusion.copy() + Lambda_J

    N = len(price_grid)
    for i in range(N):
        G_svj[i, i] = -(G_svj[i, :].sum() - G_svj[i, i])

    logger.debug(f"Jump component added to regime generator")

    return G_svj


def construct_all_regime_generators_svj(
    price_grid, variance_grid, model_params, jump_params, stencil='auto'
):
    """
    为所有方差体制构造含跳跃的生成元

    方案A: 对每个体制，先构造带漂移修正的扩散生成元
    (mu_X_SVJ = mu_X_SV - lambda*k_bar), 再叠加跳跃矩阵Lambda_J。
    Layer 1 (方差生成元) 完全不变。

    stencil 选项:
      '3pt'  — 3 点模板
      '5pt'  — 5 点模板 (匹配前 4 阶矩)
      'auto' — 优先 5 点，退回 3 点

    参数:
        price_grid (np.ndarray): 对数价格网格, 形状 (N,)
        variance_grid (np.ndarray): 方差网格, 形状 (m,)
        model_params (dict): 模型参数
        jump_params (dict): 跳跃参数
        stencil (str): 模板类型

    返回值:
        list: 含跳跃的体制生成元列表
    """
    m = len(variance_grid)
    N = len(price_grid)
    generators_svj = []

    Lambda_J = construct_jump_matrix(price_grid, jump_params)
    drift_correction = compute_jump_drift_correction(jump_params)

    from src.layer2_price import construct_regime_generator_with_drift
    for l in range(m):
        G_drift = construct_regime_generator_with_drift(
            price_grid, variance_grid[l], model_params, drift_correction,
            stencil=stencil,
        )
        G_svj = G_drift + Lambda_J
        for i in range(N):
            G_svj[i, i] = -(G_svj[i, :].sum() - G_svj[i, i])
        generators_svj.append(G_svj)

    generators_svj = np.stack(generators_svj)

    logger.info(
        f"SVJ regime generators built: tensor {generators_svj.shape}, "
        f"jump_type={jump_params.get('jump_type', 'merton')}, "
        f"lambda={jump_params.get('lambda_jump', 0)}, "
        f"drift_correction={drift_correction:.6f}"
    )

    return generators_svj
