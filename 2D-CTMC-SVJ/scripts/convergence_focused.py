# -*- coding: utf-8 -*-
import sys, copy, time
import numpy as np
from loguru import logger
from commonConfig import HESTON_DEFAULT_PARAMS, SVJ_JUMP_DEFAULT_PARAMS, LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
from src.jump_generator import construct_all_regime_generators_svj
from src.tensor_pricing import price_european_tensor
from src.heston_analytical import compute_heston_price
from src.svj_analytical import compute_svj_price

logger.remove()
logger.add(sys.stderr, level="WARNING")

mp = copy.deepcopy(HESTON_DEFAULT_PARAMS)
jp = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)

configs = [
    {'label': '40x200_base',     'm': 40, 'N': 200, 'ns': 400, 'v_upper': 0.12, 'x_factor': 3.0},
    {'label': '60x300_base',     'm': 60, 'N': 300, 'ns': 400, 'v_upper': 0.12, 'x_factor': 3.0},
    {'label': '40x200_wide',     'm': 40, 'N': 200, 'ns': 400, 'v_upper': 0.25, 'x_factor': 5.0},
    {'label': '60x300_wide',     'm': 60, 'N': 300, 'ns': 400, 'v_upper': 0.25, 'x_factor': 5.0},
    {'label': '80x400_wide',     'm': 80, 'N': 400, 'ns': 400, 'v_upper': 0.25, 'x_factor': 5.0},
]

strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
T_val = 1.0

print("=" * 100)
print("Convergence Study: Grid Refinement + Grid Widening (T=1.0, SV+SVJ, Lie)")
print("Params: rho=-0.7, kappa=2.0, sigma_v=0.3, lambda=0.1")
print("=" * 100)

header = (f"{'Config':>18s} {'K':>4s} | "
          f"{'CTMC_SV':>10s} {'Heston':>10s} {'RE_SV':>8s} | "
          f"{'CTMC_SVJ':>10s} {'SVJ_ana':>10s} {'RE_SVJ':>8s} | "
          f"{'Time':>6s} {'Rank':>4s}")
print(header)
print("-" * 100)

for cfg in configs:
    m, N, ns = cfg['m'], cfg['N'], cfg['ns']
    l1 = copy.deepcopy(LAYER1_GRID_CONFIG)
    l1['m'] = m
    l1['v_upper_abs'] = cfg['v_upper']
    l2 = copy.deepcopy(LAYER2_GRID_CONFIG)
    l2['N'] = N
    l2['x_lower_factor'] = cfg['x_factor']
    l2['x_upper_factor'] = cfg['x_factor']

    t0 = time.time()
    vg = build_variance_grid(l1, mp)
    pg = build_price_grid(l2, mp, vg)
    Q = construct_variance_generator(vg, mp)
    G_sv = construct_all_regime_generators(pg, vg, mp, Q_variance=Q)
    G_svj = construct_all_regime_generators_svj(pg, vg, mp, jp)
    t_build = time.time() - t0

    for K in strikes:
        opt = {'K': K, 'T': T_val, 'option_type': 'put'}

        heston_p = compute_heston_price(
            S_0=mp['S_0'], K=K, V_0=mp['V_0'],
            r=mp['r'], kappa=mp['kappa'], theta=mp['theta'],
            sigma_v=mp['sigma_v'], rho=mp['rho'], T=T_val, option_type='put',
        )
        svj_p = compute_svj_price(
            S0=mp['S_0'], K=K, V0=mp['V_0'],
            r=mp['r'], kappa=mp['kappa'], theta=mp['theta'],
            sigma_v=mp['sigma_v'], rho=mp['rho'], T=T_val,
            lambda_jump=jp['lambda_jump'], mu_J=jp['mu_J'], sigma_J=jp['sigma_J'],
            option_type='put',
        )

        t0 = time.time()
        p_sv, r_sv = price_european_tensor(Q, G_sv, pg, vg, mp, opt, ns, max_rank=None, svd_tol=1e-6)
        t0 = time.time()
        p_svj, r_svj = price_european_tensor(Q, G_svj, pg, vg, mp, opt, ns, max_rank=None, svd_tol=1e-6)
        t_price = time.time() - t0

        re_sv = (p_sv - heston_p) / max(heston_p, 1e-10) * 100
        re_svj = (p_svj - svj_p) / max(svj_p, 1e-10) * 100

        print(f"{cfg['label']:>18s} {K:4d} | "
              f"{p_sv:10.4f} {heston_p:10.4f} {re_sv:+7.2f}% | "
              f"{p_svj:10.4f} {svj_p:10.4f} {re_svj:+7.2f}% | "
              f"{t_price:5.1f}s {r_svj[-1]:4d}")

    print()

print("=" * 100)
print("SUMMARY: |RE_SV| by config and strike")
print("=" * 100)
