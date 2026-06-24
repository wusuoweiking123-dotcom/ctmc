# -*- coding: utf-8 -*-
import sys
import copy
import time
import numpy as np
import pandas as pd
from loguru import logger

from commonConfig import (
    HESTON_DEFAULT_PARAMS,
    SVJ_JUMP_DEFAULT_PARAMS,
    LAYER1_GRID_CONFIG,
    LAYER2_GRID_CONFIG,
)
from src.grid_construction import (
    build_variance_grid,
    build_price_grid,
)
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
from src.jump_generator import construct_all_regime_generators_svj
from src.option_pricing import price_european_fast
from src.heston_analytical import compute_heston_price
from src.svj_analytical import compute_svj_price

logger.remove()
logger.add(sys.stderr, level="WARNING")


def run_grid(m, N, n_steps, model_params, jump_params):
    l1 = copy.deepcopy(LAYER1_GRID_CONFIG)
    l1['m'] = m
    l2 = copy.deepcopy(LAYER2_GRID_CONFIG)
    l2['N'] = N

    t0 = time.time()
    vg = build_variance_grid(l1, model_params)
    pg = build_price_grid(l2, model_params, vg)
    Q = construct_variance_generator(vg, model_params)
    G_sv = construct_all_regime_generators(pg, vg, model_params, Q_variance=Q)
    G_svj = construct_all_regime_generators_svj(pg, vg, model_params, jump_params)
    t_build = time.time() - t0

    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    T_val = 1.0
    rows = []

    for K in strikes:
        opt = {'K': K, 'T': T_val, 'option_type': 'put'}

        heston_p = compute_heston_price(
            S_0=model_params['S_0'], K=K, V_0=model_params['V_0'],
            r=model_params['r'], kappa=model_params['kappa'],
            theta=model_params['theta'], sigma_v=model_params['sigma_v'],
            rho=model_params['rho'], T=T_val, option_type='put',
        )
        svj_p = compute_svj_price(
            S0=model_params['S_0'], K=K, V0=model_params['V_0'],
            r=model_params['r'], kappa=model_params['kappa'],
            theta=model_params['theta'], sigma_v=model_params['sigma_v'],
            rho=model_params['rho'], T=T_val,
            lambda_jump=jump_params['lambda_jump'],
            mu_J=jump_params['mu_J'],
            sigma_J=jump_params['sigma_J'],
            option_type='put',
        )

        t0 = time.time()
        ctmc_sv = price_european_fast(
            Q, G_sv, pg, vg, model_params, opt, n_steps
        )
        t_sv = time.time() - t0

        t0 = time.time()
        ctmc_svj = price_european_fast(
            Q, G_svj, pg, vg, model_params, opt, n_steps
        )
        t_svj = time.time() - t0

        re_sv = (ctmc_sv - heston_p) / max(heston_p, 1e-10) * 100
        re_svj = (ctmc_svj - svj_p) / max(svj_p, 1e-10) * 100

        rows.append({
            'm': m, 'N': N, 'n_steps': n_steps,
            'K': K, 'T': T_val,
            'CTMC_SV': ctmc_sv, 'Heston': heston_p, 'RE_SV_%': re_sv,
            'CTMC_SVJ': ctmc_svj, 'SVJ': svj_p, 'RE_SVJ_%': re_svj,
            'Time_SV_s': t_sv, 'Time_SVJ_s': t_svj, 'Build_s': t_build,
            'Total_States': m * N,
        })

        print(
            f"  {m:>2}x{N:<3} K={K:3d} | "
            f"SV: {ctmc_sv:8.4f} vs {heston_p:8.4f} ({re_sv:+7.2f}%) | "
            f"SVJ: {ctmc_svj:8.4f} vs {svj_p:8.4f} ({re_svj:+7.2f}%)"
        )

    return rows


if __name__ == "__main__":
    mp = copy.deepcopy(HESTON_DEFAULT_PARAMS)
    jp = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-grid', type=int, default=40)
    args = parser.parse_args()

    all_configs = [
        (20, 100, 200),
        (30, 150, 300),
        (40, 200, 400),
        (60, 300, 400),
        (80, 400, 400),
    ]
    configs = [c for c in all_configs if c[0] <= args.max_grid]

    print("=" * 90)
    print("Grid Convergence Study: 2D-CTMC-SVJ (T=1.0, Lie splitting)")
    print("=" * 90)

    all_rows = []
    for m, N, ns in configs:
        print(f"\n--- Grid: {m}x{N}, n_steps={ns}, states={m*N} ---")
        rows = run_grid(m, N, ns, mp, jp)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = f"result/grid_convergence_{ts}.csv"
    df.to_csv(out, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 90)
    print("CONVERGENCE SUMMARY (T=1.0)")
    print("=" * 90)
    print(f"{'Grid':>10s} {'States':>8s} | "
          f"{'SV_MEAN':>8s} {'SV_MAX':>8s} {'SV_K85':>8s} {'SV_K120':>8s} | "
          f"{'SVJ_MEAN':>8s} {'SVJ_MAX':>8s} {'SVJ_K85':>8s} {'SVJ_K120':>8s} | "
          f"{'Time_s':>8s}")
    print("-" * 110)

    for m, N, ns in configs:
        sub = df[(df['m'] == m) & (df['N'] == N)]
        if len(sub) == 0:
            continue
        sv_mean = sub['RE_SV_%'].abs().mean()
        sv_max = sub['RE_SV_%'].abs().max()
        sv_k85 = sub.loc[sub['K'] == 85, 'RE_SV_%'].values
        sv_k85 = sv_k85[0] if len(sv_k85) else float('nan')
        sv_k120 = sub.loc[sub['K'] == 120, 'RE_SV_%'].values
        sv_k120 = sv_k120[0] if len(sv_k120) else float('nan')
        svj_mean = sub['RE_SVJ_%'].abs().mean()
        svj_max = sub['RE_SVJ_%'].abs().max()
        svj_k85 = sub.loc[sub['K'] == 85, 'RE_SVJ_%'].values
        svj_k85 = svj_k85[0] if len(svj_k85) else float('nan')
        svj_k120 = sub.loc[sub['K'] == 120, 'RE_SVJ_%'].values
        svj_k120 = svj_k120[0] if len(svj_k120) else float('nan')
        t_total = sub['Time_SV_s'].mean() + sub['Time_SVJ_s'].mean()

        print(f"{m:>3}x{N:<3}   {m*N:>8d} | "
              f"{sv_mean:+7.2f}% {sv_max:+7.2f}% {sv_k85:+7.2f}% {sv_k120:+7.2f}% | "
              f"{svj_mean:+7.2f}% {svj_max:+7.2f}% {svj_k85:+7.2f}% {svj_k120:+7.2f}% | "
              f"{t_total:8.2f}")

    print(f"\nResults saved to {out}")
