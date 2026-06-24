# -*- coding: utf-8 -*-
"""
文件名: src/svj_analytical.py
功能描述: Heston-SVJ (Bates 1996) 半闭式解模块
         基于 Heston 特征函数 + Merton 跳跃修正
         用于验证 CTMC-SVJ 实现的正确性
作者: [Author]
创建日期: 2026-05-07
"""

import numpy as np
from scipy.integrate import quad
from loguru import logger


def _svj_integrand(phi, S0, V0, b, kappa, theta, sigma_v, rho, T,
                    lambda_jump, mu_J, sigma_J, K, p_num):
    """
    Heston-SVJ 定价积分核函数
    在 Heston 积分核的基础上添加 Merton 跳跃修正:
        f_SVJ(phi) = f_Heston(phi) * phi_Jump(phi)

    参数:
        phi (float): 积分变量
        b (float): 持有成本 (r for spot, 0 for futures)
        p_num (int): 1 (P1) 或 2 (P2)

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

    k_bar = np.exp(mu_J + sigma_J ** 2 / 2) - 1

    b_adj = b - lambda_jump * k_bar

    d = np.sqrt(
        (rho * sigma_v * i * phi - b_param) ** 2
        - sigma_v ** 2 * (2 * u_param * i * phi - phi ** 2)
    )

    g = (b_param - rho * sigma_v * i * phi - d) / (b_param - rho * sigma_v * i * phi + d)

    exp_dT = np.exp(-d * T)
    g_exp = g * exp_dT
    log_term = np.log((1 - g_exp) / (1 - g))

    C = (b_adj * i * phi * T
         + kappa * theta / sigma_v ** 2 * (
             (b_param - rho * sigma_v * i * phi - d) * T
             - 2 * log_term
         ))

    D = ((b_param - rho * sigma_v * i * phi - d) / sigma_v ** 2
         * (1 - exp_dT) / (1 - g_exp))

    f_heston = np.exp(C + D * V0 + i * phi * np.log(S0))

    phi_jump = np.exp(
        lambda_jump * T * (
            np.exp(i * phi * mu_J - sigma_J ** 2 * phi ** 2 / 2)
            - 1
        )
    )

    f_svj = f_heston * phi_jump

    return np.real(np.exp(-i * phi * np.log(K)) * f_svj / (i * phi))


def compute_svj_price(S0, K, V0, r, kappa, theta, sigma_v, rho, T,
                       lambda_jump, mu_J, sigma_J, option_type='call',
                       underlying_type='spot'):
    """
    计算 Heston-SVJ 模型的欧式期权价格 (Bates 1996)

    Spot:   Call = S0 * P1 - K * exp(-rT) * P2
    Futures: Call = exp(-rT) * [F * P1 - K * P2]

    参数:
        S0 (float): 初始资产价格或期货价格
        K (float): 行权价
        V0 (float): 初始方差
        r (float): 无风险利率
        kappa (float): 方差均值回归速度
        theta (float): 方差长期均值
        sigma_v (float): 方差波动率
        rho (float): 价格-方差相关系数
        T (float): 到期时间
        lambda_jump (float): 跳跃强度
        mu_J (float): 对数跳跃大小均值
        sigma_J (float): 对数跳跃大小标准差
        option_type (str): 'call' 或 'put'
        underlying_type (str): 'spot' 或 'futures'

    返回值:
        float: 期权价格
    """
    b = 0.0 if underlying_type == 'futures' else r

    P2_int, _ = quad(
        _svj_integrand, 1e-8, 200,
        args=(S0, V0, b, kappa, theta, sigma_v, rho, T,
              lambda_jump, mu_J, sigma_J, K, 2),
        limit=500
    )
    P2 = 0.5 + P2_int / np.pi

    P1_int, _ = quad(
        _svj_integrand, 1e-8, 200,
        args=(S0, V0, b, kappa, theta, sigma_v, rho, T,
              lambda_jump, mu_J, sigma_J, K, 1),
        limit=500
    )
    P1 = 0.5 + P1_int / np.pi

    if underlying_type == 'futures':
        call_price = np.exp(-r * T) * (S0 * P1 - K * P2)
    else:
        call_price = S0 * P1 - K * np.exp(-r * T) * P2

    if option_type == 'call':
        price = call_price
    else:
        if underlying_type == 'futures':
            price = np.exp(-r * T) * (K * (1 - P2) - S0 * (1 - P1))
        else:
            price = call_price - S0 + K * np.exp(-r * T)

    logger.info(
        f"SVJ analytical: {option_type} S={S0} K={K} T={T:.4f} "
        f"-> price={price:.6f} (P1={P1:.6f}, P2={P2:.6f})"
    )

    return price


# =============================================================================
# 向量化快速定价 (Gauss-Legendre 固定节点积分)
# =============================================================================

from numpy.polynomial.legendre import leggauss as _leggauss_svj

_SVJ_GL_N = 96
_svj_nodes_raw, _svj_weights_raw = _leggauss_svj(_SVJ_GL_N)
_SVJ_PHI_LO, _SVJ_PHI_HI = 1e-8, 200.0
_SVJ_GL_PHI = (_SVJ_PHI_HI - _SVJ_PHI_LO) / 2 * _svj_nodes_raw + (_SVJ_PHI_HI + _SVJ_PHI_LO) / 2
_SVJ_GL_W = (_SVJ_PHI_HI - _SVJ_PHI_LO) / 2 * _svj_weights_raw


def _svj_cf(phi, S0, V0, b, kappa, theta, sigma_v, rho, T,
            lambda_jump, mu_J, sigma_J, p_num):
    """
    SVJ 特征函数 (向量化)
    phi: (N_phi,) 数组
    返回: f_svj (N_phi,) 复数数组
    """
    i = 1j
    u_param = 0.5 if p_num == 1 else -0.5
    b_param = kappa + rho * sigma_v if p_num == 1 else kappa

    k_bar = np.exp(mu_J + sigma_J ** 2 / 2) - 1
    b_adj = b - lambda_jump * k_bar

    d = np.sqrt(
        (rho * sigma_v * i * phi - b_param) ** 2
        - sigma_v ** 2 * (2 * u_param * i * phi - phi ** 2)
    )
    g = (b_param - rho * sigma_v * i * phi - d) / (b_param - rho * sigma_v * i * phi + d)
    exp_dT = np.exp(-d * T)
    g_exp = g * exp_dT
    log_term = np.log((1 - g_exp) / (1 - g))

    C = (b_adj * i * phi * T
         + kappa * theta / sigma_v ** 2 * (
             (b_param - rho * sigma_v * i * phi - d) * T
             - 2 * log_term
         ))
    D = ((b_param - rho * sigma_v * i * phi - d) / sigma_v ** 2
         * (1 - exp_dT) / (1 - g_exp))

    f_heston = np.exp(C + D * V0 + i * phi * np.log(S0))

    phi_jump = np.exp(
        lambda_jump * T * (
            np.exp(i * phi * mu_J - sigma_J ** 2 * phi ** 2 / 2)
            - 1
        )
    )

    return f_heston * phi_jump


def compute_svj_price_fast(S0, K, V0, r, kappa, theta, sigma_v, rho, T,
                            lambda_jump, mu_J, sigma_J, option_type='call',
                            underlying_type='spot'):
    """
    快速单期权定价 — Gauss-Legendre 固定节点积分
    接口与 compute_svj_price 完全一致
    """
    b = 0.0 if underlying_type == 'futures' else r
    if T <= 0:
        return max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)

    phi = _SVJ_GL_PHI
    w = _SVJ_GL_W

    f_svj_2 = _svj_cf(phi, S0, V0, b, kappa, theta, sigma_v, rho, T,
                       lambda_jump, mu_J, sigma_J, 2)
    integrand2 = np.real(np.exp(-1j * phi * np.log(K)) * f_svj_2 / (1j * phi))
    P2 = 0.5 + np.dot(w, integrand2) / np.pi

    f_svj_1 = _svj_cf(phi, S0, V0, b, kappa, theta, sigma_v, rho, T,
                       lambda_jump, mu_J, sigma_J, 1)
    integrand1 = np.real(np.exp(-1j * phi * np.log(K)) * f_svj_1 / (1j * phi))
    P1 = 0.5 + np.dot(w, integrand1) / np.pi

    if underlying_type == 'futures':
        call_price = np.exp(-r * T) * (S0 * P1 - K * P2)
    else:
        call_price = S0 * P1 - K * np.exp(-r * T) * P2

    if option_type == 'call':
        price = call_price
    else:
        if underlying_type == 'futures':
            price = np.exp(-r * T) * (K * (1 - P2) - S0 * (1 - P1))
        else:
            price = call_price - S0 + K * np.exp(-r * T)

    return max(price, 0)


def compute_svj_prices_batch(S0, K_arr, V0, r, kappa, theta, sigma_v, rho, T,
                              lambda_jump, mu_J, sigma_J, option_type='put',
                              underlying_type='spot'):
    """
    批量定价 — 同一 T 下多个行权价

    参数:
        K_arr: 行权价数组
    返回值:
        np.ndarray: 期权价格数组, shape (N_K,)
    """
    b = 0.0 if underlying_type == 'futures' else r
    K_arr = np.asarray(K_arr, dtype=float)

    if T <= 0:
        if option_type == 'put':
            return np.maximum(K_arr - S0, 0)
        return np.maximum(S0 - K_arr, 0)

    phi = _SVJ_GL_PHI
    w = _SVJ_GL_W
    logS = np.log(S0)
    logK = np.log(K_arr)

    f_svj_2 = _svj_cf(phi, S0, V0, b, kappa, theta, sigma_v, rho, T,
                       lambda_jump, mu_J, sigma_J, 2)
    integrand2 = np.real(np.exp(-1j * np.outer(logK, phi)) * f_svj_2 / (1j * phi))
    P2 = 0.5 + integrand2 @ w / np.pi

    f_svj_1 = _svj_cf(phi, S0, V0, b, kappa, theta, sigma_v, rho, T,
                       lambda_jump, mu_J, sigma_J, 1)
    integrand1 = np.real(np.exp(-1j * np.outer(logK, phi)) * f_svj_1 / (1j * phi))
    P1 = 0.5 + integrand1 @ w / np.pi

    if underlying_type == 'futures':
        call_prices = np.exp(-r * T) * (S0 * P1 - K_arr * P2)
    else:
        call_prices = S0 * P1 - K_arr * np.exp(-r * T) * P2

    if option_type == 'call':
        return np.maximum(call_prices, 0)
    else:
        if underlying_type == 'futures':
            put_prices = np.exp(-r * T) * (K_arr * (1 - P2) - S0 * (1 - P1))
        else:
            put_prices = call_prices - S0 + K_arr * np.exp(-r * T)
        return np.maximum(put_prices, 0)
