# -*- coding: utf-8 -*-
"""
test_shift_fix.py — 验证 X-shift 修复对翼部精度的改善
对比 rho=-0.7 下修复前后的定价误差
"""

import sys
import copy
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

from commonConfig import HESTON_DEFAULT_PARAMS, LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
from src.option_pricing import price_european_fast, price_european_strang
from src.heston_analytical import compute_heston_price

MP = copy.deepcopy(HESTON_DEFAULT_PARAMS)
L1 = copy.deepcopy(LAYER1_GRID_CONFIG)
L2 = copy.deepcopy(LAYER2_GRID_CONFIG)
STRIKES = [85, 90, 95, 100, 105, 110, 115]
T = 1.0


def test_fix():
    logger.info("=" * 80)
    logger.info("X-SHIFT FIX VERIFICATION (rho=-0.7, 40x200, Lie + Strang)")
    logger.info("=" * 80)

    vg = build_variance_grid(L1, MP)
    pg = build_price_grid(L2, MP, vg)
    Q = construct_variance_generator(vg, MP)
    G_list = construct_all_regime_generators(pg, vg, MP, Q_variance=Q)

    logger.info(
        f"{'K':>4s} | {'Heston':>10s} | {'CTMC_Lie':>10s} | {'RE_Lie%':>9s} | "
        f"{'CTMC_Strang':>11s} | {'RE_Str%':>9s}"
    )
    logger.info("-" * 75)

    for K in STRIKES:
        opt_p = {'K': K, 'T': T, 'option_type': 'put'}

        heston = compute_heston_price(
            S_0=MP['S_0'], K=K, V_0=MP['V_0'], r=MP['r'],
            kappa=MP['kappa'], theta=MP['theta'], sigma_v=MP['sigma_v'],
            rho=MP['rho'], T=T, option_type='put'
        )

        ctmc_lie = price_european_fast(Q, G_list, pg, vg, MP, opt_p, 400)
        ctmc_strang = price_european_strang(Q, G_list, pg, vg, MP, opt_p, 400)

        re_lie = (ctmc_lie - heston) / max(heston, 1e-10) * 100
        re_str = (ctmc_strang - heston) / max(heston, 1e-10) * 100

        logger.info(
            f"{K:4d} | {heston:10.4f} | {ctmc_lie:10.4f} | {re_lie:+9.2f} | "
            f"{ctmc_strang:11.4f} | {re_str:+9.2f}"
        )

    logger.info("\n--- rho sensitivity (K=85, K=115) ---")
    for rho in [0.0, -0.3, -0.5, -0.7, -0.9]:
        mp = copy.deepcopy(MP)
        mp['rho'] = rho
        vg2 = build_variance_grid(L1, mp)
        pg2 = build_price_grid(L2, mp, vg2)
        Q2 = construct_variance_generator(vg2, mp)
        G2 = construct_all_regime_generators(pg2, vg2, mp, Q_variance=Q2)

        results = {}
        for K in [85, 100, 115]:
            opt = {'K': K, 'T': T, 'option_type': 'put'}
            h = compute_heston_price(
                S_0=mp['S_0'], K=K, V_0=mp['V_0'], r=mp['r'],
                kappa=mp['kappa'], theta=mp['theta'], sigma_v=mp['sigma_v'],
                rho=rho, T=T, option_type='put'
            )
            c = price_european_fast(Q2, G2, pg2, vg2, mp, opt, 400)
            results[K] = (c - h) / max(h, 1e-10) * 100

        logger.info(
            f"rho={rho:5.1f} | K85: {results[85]:+8.2f}% | "
            f"K100: {results[100]:+8.2f}% | K115: {results[115]:+8.2f}%"
        )


if __name__ == "__main__":
    test_fix()
