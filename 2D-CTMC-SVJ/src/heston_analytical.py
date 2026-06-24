# -*- coding: utf-8 -*-
"""
文件名: src/heston_analytical.py
功能描述: Heston 模型闭式解模块
         基于 Heston (1993) + Albrecher et al. (2007) 的稳定旋转公式
         参考 Rouah (2013), "The Heston Model and Its Extensions"
作者: [Author]
创建日期: 2026-05-06
修改历史:
    2026-05-06 - 使用 Heston 双参数(u=1/2, u=-1/2)公式重写
"""

import numpy as np
from scipy.integrate import quad
from loguru import logger


def _heston_integrand(phi, S_0, K, V_0, b, kappa, theta, sigma_v, rho, T, p_num):
    """
    Heston 定价积分核函数
    使用标准 Heston (1993) 公式，配合 Albrecher et al. (2007) 旋转对数
        P_j = 1/2 + 1/pi * integral_0^inf Re[integrand_j(phi)] dphi

    参数区别:
        p_num=1 (P1，以S为计价单位):
            u = 0.5, b_param = kappa + rho*sigma_v
        p_num=2 (P2，风险中性测度):
            u = -0.5, b_param = kappa

    参数:
        phi (float): 积分变量
        b (float): 持有成本 (r for spot, 0 for futures)
        p_num (int): 1 或 2

    返回值:
        float: 积分核的实部
    """
    i = 1j

    if p_num == 1:
        u_param = 0.5
        b_param = kappa + rho * sigma_v
    else:
        u_param = -0.5
        b_param = kappa

    d = np.sqrt(
        (rho * sigma_v * i * phi - b_param) ** 2
        - sigma_v ** 2 * (2 * u_param * i * phi - phi ** 2)
    )

    g = (b_param - rho * sigma_v * i * phi - d) / (b_param - rho * sigma_v * i * phi + d)

    exp_dT = np.exp(-d * T)
    g_exp = g * exp_dT

    log_term = np.log((1 - g_exp) / (1 - g))

    C = (b * i * phi * T
         + kappa * theta / sigma_v ** 2 * (
             (b_param - rho * sigma_v * i * phi - d) * T
             - 2 * log_term
         ))

    D = ((b_param - rho * sigma_v * i * phi - d) / sigma_v ** 2
         * (1 - exp_dT) / (1 - g_exp))

    f = np.exp(C + D * V_0 + i * phi * np.log(S_0))

    return np.real(np.exp(-i * phi * np.log(K)) * f / (i * phi))


def compute_heston_price(S_0, K, V_0, r, kappa, theta, sigma_v, rho, T,
                          option_type='call', underlying_type='spot'):
    """
    计算 Heston 模型下的欧式期权价格（闭式解）

    Spot:   Call = S_0 * P1 - K * exp(-rT) * P2
    Futures: Call = exp(-rT) * [F * P1 - K * P2]

    参数:
        S_0 (float): 初始资产价格 (spot) 或期货价格 (futures)
        K (float): 行权价格
        V_0 (float): 初始方差
        r (float): 无风险利率
        kappa (float): 均值回归速度
        theta (float): 长期方差均值
        sigma_v (float): 方差波动率
        rho (float): 相关系数
        T (float): 到期时间（年化）
        option_type (str): 'call' 或 'put'
        underlying_type (str): 'spot' 或 'futures'

    返回值:
        float: 期权价格
    """
    b = 0.0 if underlying_type == 'futures' else r

    if T <= 0:
        if option_type == 'call':
            return max(S_0 - K, 0)
        else:
            return max(K - S_0, 0)

    P2_int, _ = quad(
        _heston_integrand, 1e-8, 200,
        args=(S_0, K, V_0, b, kappa, theta, sigma_v, rho, T, 2),
        limit=500
    )
    P2 = 0.5 + P2_int / np.pi

    P1_int, _ = quad(
        _heston_integrand, 1e-8, 200,
        args=(S_0, K, V_0, b, kappa, theta, sigma_v, rho, T, 1),
        limit=500
    )
    P1 = 0.5 + P1_int / np.pi

    if underlying_type == 'futures':
        call_price = np.exp(-r * T) * (S_0 * P1 - K * P2)
    else:
        call_price = S_0 * P1 - K * np.exp(-r * T) * P2

    if option_type == 'call':
        price = max(call_price, 0)
    else:
        if underlying_type == 'futures':
            put_price = np.exp(-r * T) * (K * (1 - P2) - S_0 * (1 - P1))
        else:
            put_price = call_price - S_0 + K * np.exp(-r * T)
        price = max(put_price, 0)

    logger.info(
        f"Heston analytical: {option_type} S={S_0} K={K} T={T:.4f} -> "
        f"price={price:.6f} (P1={P1:.6f}, P2={P2:.6f})"
    )
    return price


def compute_heston_price_vector(strikes, model_params, T, option_type='call'):
    """
    批量计算不同行权价下的Heston 期权价格

    参数:
        strikes (np.ndarray): 行权价数组        model_params (dict): 模型参数
        T (float): 到期时间
        option_type (str): 'call' 或 'put'

    返回值:
        np.ndarray: 期权价格数组
    """
    prices = np.array([
        compute_heston_price(
            S_0=model_params['S_0'],
            K=K,
            V_0=model_params['V_0'],
            r=model_params['r'],
            kappa=model_params['kappa'],
            theta=model_params['theta'],
            sigma_v=model_params['sigma_v'],
            rho=model_params['rho'],
            T=T,
            option_type=option_type,
        )
        for K in strikes
    ])

    logger.info(f"Heston analytical prices computed for {len(strikes)} strikes")
    return prices


# =============================================================================
# 向量化快速定价 (Gauss-Legendre 固定节点积分)
# =============================================================================

from numpy.polynomial.legendre import leggauss

_GL_N = 96
_gl_nodes_raw, _gl_weights_raw = leggauss(_GL_N)
_PHI_LO, _PHI_HI = 1e-8, 200.0
_GL_PHI = (_PHI_HI - _PHI_LO) / 2 * _gl_nodes_raw + (_PHI_HI + _PHI_LO) / 2
_GL_W = (_PHI_HI - _PHI_LO) / 2 * _gl_weights_raw


def _heston_cf(phi, b, kappa, theta, sigma_v, rho, T, p_num):
    """
    Heston 特征函数核心计算 (向量化)
    phi: (N_phi,) 数组
    返回: C, D — 均为 (N_phi,) 复数数组
    """
    i = 1j
    u_param = 0.5 if p_num == 1 else -0.5
    b_param = kappa + rho * sigma_v if p_num == 1 else kappa

    d = np.sqrt(
        (rho * sigma_v * i * phi - b_param) ** 2
        - sigma_v ** 2 * (2 * u_param * i * phi - phi ** 2)
    )
    g = (b_param - rho * sigma_v * i * phi - d) / (b_param - rho * sigma_v * i * phi + d)
    exp_dT = np.exp(-d * T)
    g_exp = g * exp_dT
    log_term = np.log((1 - g_exp) / (1 - g))

    C = (b * i * phi * T
         + kappa * theta / sigma_v ** 2 * (
             (b_param - rho * sigma_v * i * phi - d) * T
             - 2 * log_term
         ))
    D = ((b_param - rho * sigma_v * i * phi - d) / sigma_v ** 2
         * (1 - exp_dT) / (1 - g_exp))
    return C, D


def compute_heston_price_fast(S_0, K, V_0, r, kappa, theta, sigma_v, rho, T,
                               option_type='call', underlying_type='spot'):
    """
    快速单期权定价 — Gauss-Legendre 固定节点积分 (替代 quad)
    接口与 compute_heston_price 完全一致
    """
    b = 0.0 if underlying_type == 'futures' else r
    if T <= 0:
        return max(S_0 - K, 0) if option_type == 'call' else max(K - S_0, 0)

    phi = _GL_PHI
    w = _GL_W

    C2, D2 = _heston_cf(phi, b, kappa, theta, sigma_v, rho, T, 2)
    f2 = np.exp(C2 + D2 * V_0 + 1j * phi * np.log(S_0))
    integrand2 = np.real(np.exp(-1j * phi * np.log(K)) * f2 / (1j * phi))
    P2 = 0.5 + np.dot(w, integrand2) / np.pi

    C1, D1 = _heston_cf(phi, b, kappa, theta, sigma_v, rho, T, 1)
    f1 = np.exp(C1 + D1 * V_0 + 1j * phi * np.log(S_0))
    integrand1 = np.real(np.exp(-1j * phi * np.log(K)) * f1 / (1j * phi))
    P1 = 0.5 + np.dot(w, integrand1) / np.pi

    if underlying_type == 'futures':
        call_price = np.exp(-r * T) * (S_0 * P1 - K * P2)
    else:
        call_price = S_0 * P1 - K * np.exp(-r * T) * P2

    if option_type == 'call':
        return max(call_price, 0)
    else:
        if underlying_type == 'futures':
            put_price = np.exp(-r * T) * (K * (1 - P2) - S_0 * (1 - P1))
        else:
            put_price = call_price - S_0 + K * np.exp(-r * T)
        return max(put_price, 0)


def compute_heston_prices_batch(S_0, K_arr, V_0, r, kappa, theta, sigma_v, rho, T,
                                 option_type='put', underlying_type='spot'):
    """
    批量定价 — 同一 T 下多个行权价，利用 numpy 广播

    phi (N_phi,) × K (N_K,) → integrand (N_K, N_phi) → 矩阵乘法求积分

    参数:
        K_arr: 行权价数组
        其余参数同 compute_heston_price
    返回值:
        np.ndarray: 期权价格数组, shape (N_K,)
    """
    b = 0.0 if underlying_type == 'futures' else r
    K_arr = np.asarray(K_arr, dtype=float)

    if T <= 0:
        if option_type == 'put':
            return np.maximum(K_arr - S_0, 0)
        return np.maximum(S_0 - K_arr, 0)

    phi = _GL_PHI   # (N_phi,)
    w = _GL_W       # (N_phi,)
    logS = np.log(S_0)
    logK = np.log(K_arr)  # (N_K,)

    # P2: 特征函数对 phi 向量化, log(K) 对 K 向量化 → 外积
    C2, D2 = _heston_cf(phi, b, kappa, theta, sigma_v, rho, T, 2)
    f2 = np.exp(C2 + D2 * V_0 + 1j * phi * logS)          # (N_phi,)
    integrand2 = np.real(np.exp(-1j * np.outer(logK, phi)) * f2 / (1j * phi))  # (N_K, N_phi)
    P2 = 0.5 + integrand2 @ w / np.pi                       # (N_K,)

    # P1
    C1, D1 = _heston_cf(phi, b, kappa, theta, sigma_v, rho, T, 1)
    f1 = np.exp(C1 + D1 * V_0 + 1j * phi * logS)
    integrand1 = np.real(np.exp(-1j * np.outer(logK, phi)) * f1 / (1j * phi))
    P1 = 0.5 + integrand1 @ w / np.pi

    if underlying_type == 'futures':
        call_prices = np.exp(-r * T) * (S_0 * P1 - K_arr * P2)
    else:
        call_prices = S_0 * P1 - K_arr * np.exp(-r * T) * P2

    if option_type == 'call':
        return np.maximum(call_prices, 0)
    else:
        if underlying_type == 'futures':
            put_prices = np.exp(-r * T) * (K_arr * (1 - P2) - S_0 * (1 - P1))
        else:
            put_prices = call_prices - S_0 + K_arr * np.exp(-r * T)
        return np.maximum(put_prices, 0)
