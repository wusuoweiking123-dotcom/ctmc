# -*- coding: utf-8 -*-
"""
diagnostic_correlation.py — 诊断 rho 对翼部偏差的影响
对比 rho=0, -0.3, -0.5, -0.7, -0.9 的定价误差
"""

import sys
import copy
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

from commonConfig import (
    HESTON_DEFAULT_PARAMS, LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG,
)
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
from src.option_pricing import price_european_fast
from src.heston_analytical import compute_heston_price

L1 = copy.deepcopy(LAYER1_GRID_CONFIG)
L2 = copy.deepcopy(LAYER2_GRID_CONFIG)

RHO_VALUES = [0.0, -0.3, -0.5, -0.7, -0.9]
STRIKES = [85, 90, 95, 100, 105, 110, 115]
T = 1.0


def test_rho_sensitivity():
    logger.info("=" * 90)
    logger.info("CORRELATION SENSITIVITY: Wing error vs rho")
    logger.info("=" * 90)

    header = f"{'rho':>6s} | {'K':>4s} | {'Heston':>10s} | {'CTMC':>10s} | {'RE%':>10s}"
    logger.info(header)
    logger.info("-" * 90)

    for rho in RHO_VALUES:
        mp = copy.deepcopy(HESTON_DEFAULT_PARAMS)
        mp['rho'] = rho

        vg = build_variance_grid(L1, mp)
        pg = build_price_grid(L2, mp, vg)
        Q = construct_variance_generator(vg, mp)
        G_list = construct_all_regime_generators(pg, vg, mp, Q_variance=Q)

        for K in STRIKES:
            opt_p = {'K': K, 'T': T, 'option_type': 'put'}

            heston = compute_heston_price(
                S_0=mp['S_0'], K=K, V_0=mp['V_0'], r=mp['r'],
                kappa=mp['kappa'], theta=mp['theta'], sigma_v=mp['sigma_v'],
                rho=rho, T=T, option_type='put'
            )

            ctmc = price_european_fast(
                Q, G_list, pg, vg, mp, opt_p, 400
            )

            re = (ctmc - heston) / max(heston, 1e-10) * 100

            logger.info(f"{rho:6.1f} | {K:4d} | {heston:10.4f} | {ctmc:10.4f} | {re:+10.2f}")

        logger.info("-" * 90)

    logger.info("\nSummary: RE% at K=85 (OTM put) across rho values:")
    logger.info("-" * 50)


if __name__ == "__main__":
    test_rho_sensitivity()
