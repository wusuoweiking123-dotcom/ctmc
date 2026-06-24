# -*- coding: utf-8 -*-
"""
diagnostic_bias.py — 诊断 CTMC 系统性偏差来源
测试: (1) 鞅条件 (2) 指数矩误差分布 (3) 有/无漂移修正对比 (4) Regular vs Fast
"""

import sys
import copy
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

from commonConfig import (
    HESTON_DEFAULT_PARAMS, SVJ_JUMP_DEFAULT_PARAMS,
    LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG,
)
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import (
    construct_variance_generator,
    compute_v_generator_exp_moment,
)
from src.layer2_price import (
    construct_all_regime_generators,
    compute_initial_auxiliary_value,
    compute_decorrelation_function,
)
from src.option_pricing import price_european_fast, price_european_strang
from src.heston_analytical import compute_heston_price

MP = copy.deepcopy(HESTON_DEFAULT_PARAMS)
JP = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)
L1 = copy.deepcopy(LAYER1_GRID_CONFIG)
L2 = copy.deepcopy(LAYER2_GRID_CONFIG)


def build_grids(l1_cfg, l2_cfg, mp):
    vg = build_variance_grid(l1_cfg, mp)
    pg = build_price_grid(l2_cfg, mp, vg)
    return vg, pg


def diag_exp_moment_errors():
    """诊断 1: 指数矩误差 ε_v(l) 的分布"""
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC 1: Exponential moment errors ε_v(l)")
    logger.info("=" * 60)

    vg, pg = build_grids(L1, L2, MP)
    Q = construct_variance_generator(vg, MP)

    cv_actual, cv_target, eps_v = compute_v_generator_exp_moment(Q, vg, MP)

    m = len(vg)
    n_corrected = np.sum(np.abs(eps_v) > 1e-14)
    logger.info(f"Grid: m={m}, N={len(pg)}")
    logger.info(f"States with |ε_v| > 1e-14: {n_corrected}/{m}")
    logger.info(f"ε_v range: [{eps_v.min():.6e}, {eps_v.max():.6e}]")
    logger.info(f"|ε_v| max: {np.abs(eps_v).max():.6e}")

    boundary_eps = [eps_v[0], eps_v[1], eps_v[-2], eps_v[-1]]
    interior_eps = eps_v[2:-2]
    logger.info(f"Boundary ε_v: {boundary_eps}")
    logger.info(f"Interior |ε_v| max: {np.abs(interior_eps).max():.6e}")

    large_eps_idx = np.where(np.abs(eps_v) > 1e-10)[0]
    if len(large_eps_idx) > 0:
        logger.info(f"States with |ε_v| > 1e-10:")
        for idx in large_eps_idx:
            logger.info(
                f"  l={idx}, v={vg[idx]:.6f}, ε_v={eps_v[idx]:.6e}, "
                f"cv_actual={cv_actual[idx]:.6f}, cv_target={cv_target[idx]:.6f}"
            )

    return Q, vg, pg, eps_v


def diag_martingale_check(Q, vg, pg, eps_v):
    """诊断 2: 鞅条件检查 — 定价 K→0 call 检验 E[S_T] = S0*exp(rT)"""
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 2: Martingale condition check")
    logger.info("=" * 60)

    r = MP['r']
    S0 = MP['S_0']
    T = 1.0
    expected_forward = S0 * np.exp(r * T)

    G_list = construct_all_regime_generators(pg, vg, MP, Q_variance=Q)

    opt_call_zero = {'K': 1e-10, 'T': T, 'option_type': 'call'}

    ctmc_call0 = price_european_fast(
        Q, G_list, pg, vg, MP, opt_call_zero, 400
    )

    forward_implied = ctmc_call0 * np.exp(r * T)
    forward_error = (forward_implied - expected_forward) / expected_forward * 100

    logger.info(f"Expected forward: {expected_forward:.6f}")
    logger.info(f"CTMC implied forward (via K→0 call): {forward_implied:.6f}")
    logger.info(f"Forward error: {forward_error:+.4f}%")

    opt_call_100 = {'K': 100.0, 'T': T, 'option_type': 'call'}
    opt_put_100 = {'K': 100.0, 'T': T, 'option_type': 'put'}

    ctmc_call = price_european_fast(Q, G_list, pg, vg, MP, opt_call_100, 400)
    ctmc_put = price_european_fast(Q, G_list, pg, vg, MP, opt_put_100, 400)

    parity_ctmc = ctmc_call - ctmc_put
    parity_theory = S0 - 100.0 * np.exp(-r * T)
    parity_error = parity_ctmc - parity_theory

    logger.info(f"\nPut-Call Parity (K=100):")
    logger.info(f"  CTMC Call={ctmc_call:.6f}, Put={ctmc_put:.6f}, C-P={parity_ctmc:.6f}")
    logger.info(f"  Theoretical C-P = {parity_theory:.6f}")
    logger.info(f"  Parity error = {parity_error:.6f}")


def diag_with_without_correction():
    """诊断 3: 对比有/无指数矩漂移修正的定价"""
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 3: With vs Without exponential moment correction")
    logger.info("=" * 60)

    vg, pg = build_grids(L1, L2, MP)
    Q = construct_variance_generator(vg, MP)

    G_with = construct_all_regime_generators(pg, vg, MP, Q_variance=Q)
    G_without = construct_all_regime_generators(pg, vg, MP, Q_variance=None)

    strikes = [85, 90, 95, 100, 105, 110, 115]
    T = 1.0

    logger.info(
        f"{'K':>5s} | {'Heston':>10s} | {'With_corr':>10s} | {'RE_corr%':>10s} | "
        f"{'No_corr':>10s} | {'RE_nocorr%':>10s} | {'Delta%':>10s}"
    )
    logger.info("-" * 80)

    for K in strikes:
        opt_p = {'K': K, 'T': T, 'option_type': 'put'}

        heston = compute_heston_price(
            S_0=MP['S_0'], K=K, V_0=MP['V_0'], r=MP['r'],
            kappa=MP['kappa'], theta=MP['theta'], sigma_v=MP['sigma_v'],
            rho=MP['rho'], T=T, option_type='put'
        )

        p_with = price_european_fast(
            Q, G_with, pg, vg, MP, opt_p, 400
        )
        p_without = price_european_fast(
            Q, G_without, pg, vg, MP, opt_p, 400
        )

        re_with = (p_with - heston) / max(heston, 1e-10) * 100
        re_without = (p_without - heston) / max(heston, 1e-10) * 100
        delta = re_with - re_without

        logger.info(
            f"{K:5d} | {heston:10.4f} | {p_with:10.4f} | {re_with:+10.2f} | "
            f"{p_without:10.4f} | {re_without:+10.2f} | {delta:+10.2f}"
        )


def diag_lie_vs_strang_vs_regular():
    """诊断 4: Lie-Trotter vs Strang vs Regular (小网格) 定价对比"""
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 4: Lie vs Strang vs Regular (small grid 10x50)")
    logger.info("=" * 60)

    l1_small = dict(L1, m=10)
    l2_small = dict(L2, N=50)
    vg, pg = build_grids(l1_small, l2_small, MP)
    Q = construct_variance_generator(vg, MP)
    G_list = construct_all_regime_generators(pg, vg, MP, Q_variance=Q)

    strikes = [85, 100, 115]
    T = 1.0

    from src.combined_generator import construct_combined_generator
    from src.option_pricing import (
        price_european_regular,
        build_payoff_vector,
    )
    from src.layer2_price import compute_initial_auxiliary_value, compute_decorrelation_function

    G_combined = construct_combined_generator(Q, G_list, use_sparse=True)

    X_0 = compute_initial_auxiliary_value(MP['S_0'], MP['V_0'], MP)
    rho = MP['rho']
    from src.layer1_variance import get_variance_state_index
    from src.layer2_price import get_price_state_index
    v_idx = get_variance_state_index(MP['V_0'], vg)
    p_idx = get_price_state_index(X_0, pg)
    init_idx = v_idx * len(pg) + p_idx

    logger.info(
        f"{'K':>5s} | {'Heston':>10s} | {'Regular':>10s} | {'RE_reg%':>10s} | "
        f"{'Lie':>10s} | {'RE_lie%':>10s} | {'Strang':>10s} | {'RE_str%':>10s}"
    )
    logger.info("-" * 90)

    for K in strikes:
        opt_p = {'K': K, 'T': T, 'option_type': 'put'}

        heston = compute_heston_price(
            S_0=MP['S_0'], K=K, V_0=MP['V_0'], r=MP['r'],
            kappa=MP['kappa'], theta=MP['theta'], sigma_v=MP['sigma_v'],
            rho=MP['rho'], T=T, option_type='put'
        )

        H = build_payoff_vector(pg, vg, MP, opt_p)
        p_reg = price_european_regular(G_combined.toarray(), H, MP['r'], T, init_idx)

        p_lie = price_european_fast(Q, G_list, pg, vg, MP, opt_p, 200)
        p_strang = price_european_strang(Q, G_list, pg, vg, MP, opt_p, 200)

        re_reg = (p_reg - heston) / max(heston, 1e-10) * 100
        re_lie = (p_lie - heston) / max(heston, 1e-10) * 100
        re_str = (p_strang - heston) / max(heston, 1e-10) * 100

        logger.info(
            f"{K:5d} | {heston:10.4f} | {p_reg:10.4f} | {re_reg:+10.2f} | "
            f"{p_lie:10.4f} | {re_lie:+10.2f} | {p_strang:10.4f} | {re_str:+10.2f}"
        )


def diag_variance_distribution():
    """诊断 5: CTMC 方差终端分布 vs 真实 CIR 分布"""
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC 5: Variance terminal distribution (CTMC vs CIR)")
    logger.info("=" * 60)

    vg, pg = build_grids(L1, L2, MP)
    Q = construct_variance_generator(vg, MP)
    m = len(vg)

    T = 1.0
    from scipy.linalg import expm
    P_V_T = expm(Q * T)

    v0_idx = np.argmin(np.abs(vg - MP['V_0']))
    pi_V_T = P_V_T[v0_idx, :]

    from scipy.stats import gamma as gamma_dist
    kappa = MP['kappa']
    theta = MP['theta']
    sigma_v = MP['sigma_v']

    d = 4 * kappa * theta / sigma_v**2
    c = sigma_v**2 / (4 * kappa)
    df_nc = 4 * kappa * MP['V_0'] * np.exp(-kappa * T) / (sigma_v**2 * (1 - np.exp(-kappa * T)))

    from scipy.stats import ncx2
    scale = sigma_v**2 * (1 - np.exp(-kappa * T)) / (4 * kappa)
    true_cdf = ncx2.cdf(4 * kappa * vg / (sigma_v**2 * (1 - np.exp(-kappa * T))),
                         df=d, nc=df_nc)

    ctmc_mean = np.sum(pi_V_T * vg)
    ctmc_var = np.sum(pi_V_T * vg**2) - ctmc_mean**2

    true_mean = theta + (MP['V_0'] - theta) * np.exp(-kappa * T)
    true_var = (vg[1] - vg[0])**2 * 0

    logger.info(f"CTMC E[V_T] = {ctmc_mean:.6f}")
    logger.info(f"True  E[V_T] = {true_mean:.6f}")
    logger.info(f"Error in E[V_T] = {(ctmc_mean - true_mean)/true_mean*100:+.4f}%")

    logger.info(f"\nCTMC probability distribution at T={T}:")
    logger.info(f"  v range: [{vg[0]:.6f}, {vg[-1]:.6f}]")
    logger.info(f"  P(V_T < v_0) = {np.sum(pi_V_T[:v0_idx]):.6f}")
    logger.info(f"  P(V_T > 2*theta) = {np.sum(pi_V_T[vg > 2*theta]):.6f}")
    logger.info(f"  P(V_T > v_upper*0.8) = {np.sum(pi_V_T[vg > 0.8*vg[-1]]):.6f}")

    prob_10 = np.sum(pi_V_T[vg > 0.10])
    prob_12 = np.sum(pi_V_T[vg > 0.12])
    logger.info(f"  P(V_T > 0.10) = {prob_10:.6e}")
    logger.info(f"  P(V_T > 0.12) = {prob_12:.6e}")

    top5_idx = np.argsort(pi_V_T)[-5:][::-1]
    logger.info(f"\n  Top 5 variance states by probability:")
    for idx in top5_idx:
        logger.info(f"    v[{idx}]={vg[idx]:.6f}, P={pi_V_T[idx]:.6f}")


if __name__ == "__main__":
    Q, vg, pg, eps_v = diag_exp_moment_errors()
    diag_martingale_check(Q, vg, pg, eps_v)
    diag_with_without_correction()
    diag_lie_vs_strang_vs_regular()
    diag_variance_distribution()
