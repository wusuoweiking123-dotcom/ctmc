# -*- coding: utf-8 -*-
import sys
import copy
import time
import numpy as np
from loguru import logger

from commonConfig import (
    HESTON_DEFAULT_PARAMS, SVJ_JUMP_DEFAULT_PARAMS,
    LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG,
)
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
from src.jump_generator import construct_all_regime_generators_svj
from src.option_pricing import price_european_fast, price_european_strang, _apply_variance_shift
from src.tensor_pricing import price_european_tensor
from src.heston_analytical import compute_heston_price
from src.svj_analytical import compute_svj_price

logger.remove()
logger.add(sys.stderr, level="WARNING")

mp = copy.deepcopy(HESTON_DEFAULT_PARAMS)
jp = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)

print("=" * 80)
print("Validation: Tensor Pricing vs Standard Pricing vs Analytical")
print("=" * 80)

for m, N, ns in [(20, 100, 200), (40, 200, 400)]:
    l1 = copy.deepcopy(LAYER1_GRID_CONFIG)
    l1['m'] = m
    l2 = copy.deepcopy(LAYER2_GRID_CONFIG)
    l2['N'] = N

    vg = build_variance_grid(l1, mp)
    pg = build_price_grid(l2, mp, vg)
    Q = construct_variance_generator(vg, mp)
    G_sv = construct_all_regime_generators(pg, vg, mp, Q_variance=Q)
    G_svj = construct_all_regime_generators_svj(pg, vg, mp, jp)

    print(f"\n--- Grid: {m}x{N}, n_steps={ns}, G_sv shape={G_sv.shape} ---")

    for K in [85, 100, 115]:
        for T_val in [1.0]:
            opt = {'K': K, 'T': T_val, 'option_type': 'put'}

            heston_p = compute_heston_price(
                S_0=mp['S_0'], K=K, V_0=mp['V_0'],
                r=mp['r'], kappa=mp['kappa'], theta=mp['theta'],
                sigma_v=mp['sigma_v'], rho=mp['rho'], T=T_val, option_type='put',
            )

            t0 = time.time()
            p_std = price_european_fast(Q, G_sv, pg, vg, mp, opt, ns)
            t_std = time.time() - t0

            t0 = time.time()
            p_tensor, ranks = price_european_tensor(
                Q, G_sv, pg, vg, mp, opt, ns,
                max_rank=None, svd_tol=1e-6, splitting='lie',
            )
            t_tensor = time.time() - t0

            re_std = (p_std - heston_p) / max(heston_p, 1e-10) * 100
            re_tensor = (p_tensor - heston_p) / max(heston_p, 1e-10) * 100
            diff = abs(p_tensor - p_std)

            print(
                f"  K={K:3d} T={T_val} | "
                f"Std: {p_std:.6f} ({re_std:+.2f}%) {t_std:.3f}s | "
                f"Tensor: {p_tensor:.6f} ({re_tensor:+.2f}%) {t_tensor:.3f}s | "
                f"Diff: {diff:.2e} rank: {ranks[-1] if ranks else 'N/A'}"
            )

print("\n" + "=" * 80)
print("Speed Benchmark: _apply_variance_shift (vectorized)")
print("=" * 80)

from scipy.linalg import expm

m, N = 40, 200
l1 = copy.deepcopy(LAYER1_GRID_CONFIG)
l1['m'] = m
l2 = copy.deepcopy(LAYER2_GRID_CONFIG)
l2['N'] = N

vg = build_variance_grid(l1, mp)
pg = build_price_grid(l2, mp, vg)
Q = construct_variance_generator(vg, mp)
G_sv = construct_all_regime_generators(pg, vg, mp, Q_variance=Q)

P_V = expm(Q * 0.0025)
E = np.random.randn(m, N)

n_iter = 50

t0 = time.time()
for _ in range(n_iter):
    B = _apply_variance_shift(E, P_V, pg, vg, mp)
t_elapsed = (time.time() - t0) / n_iter

print(f"  Vectorized variance shift: {t_elapsed*1000:.1f} ms/call")
print(f"  Shape check: B.shape={B.shape}")
print("\nDone.")
