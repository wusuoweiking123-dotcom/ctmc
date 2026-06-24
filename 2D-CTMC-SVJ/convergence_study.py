# -*- coding: utf-8 -*-
"""
文件名: convergence_study.py
功能描述: 系统性网格收敛性研究
          测试多种网格配置 (m × N) 的 CTMC 定价精度和计算时间
          对比 Heston/SVJ 模型下 CTMC vs 闭式解
作者: [Author]
创建日期: 2026-05-09
"""

import sys
import copy
import time
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from commonConfig import (
    HESTON_DEFAULT_PARAMS,
    SVJ_JUMP_DEFAULT_PARAMS,
    LAYER1_GRID_CONFIG,
    LAYER2_GRID_CONFIG,
    CTMC_COMPUTE_PARAMS,
    FILE_PATHS,
    LOG_CONFIG,
)
from utility.file_utils import create_directories, save_results
from src.grid_construction import (
    build_variance_grid,
    build_price_grid,
    build_adaptive_price_grid,
    build_strike_locked_price_grid,
)
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
from src.jump_generator import construct_all_regime_generators_svj
from src.option_pricing import price_european_fast, price_european_strang
from src.heston_analytical import compute_heston_price
from src.svj_analytical import compute_svj_price


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = "{}convergence_{}.log".format(FILE_PATHS['log_dir'], timestamp_str)
    create_directories([FILE_PATHS['log_dir']])
    logger.add(log_file_path, level="DEBUG", format=LOG_CONFIG['format'])
    return timestamp_str


GRID_CONFIGS = [
    {'m': 20, 'N': 100, 'label': '20x100'},
    {'m': 30, 'N': 150, 'label': '30x150'},
    {'m': 40, 'N': 200, 'label': '40x200'},
    {'m': 60, 'N': 300, 'label': '60x300'},
    {'m': 80, 'N': 400, 'label': '80x400'},
]

STRIKES = [85, 90, 95, 100, 105, 110, 115]
MATURITIES = [0.25, 0.5, 1.0]
N_STEPS_LIST = [200, 400, 800]


def run_single_config(model_params, jump_params, m, N, n_steps,
                       use_adaptive=False, use_strike_locked=False,
                       strikes_for_grid=None, smooth_payoff=False,
                       grid_config_overrides=None):
    """
    单组网格配置的定价测试

    参数:
        model_params: 模型参数
        jump_params: 跳跃参数
        m: 方差网格点数
        N: 价格网格点数
        n_steps: 时间步数
        use_adaptive: 是否使用自适应网格
        use_strike_locked: 是否使用行权价对齐网格
        strikes_for_grid: 行权价列表
        smooth_payoff: 是否使用平滑化支付函数
        grid_config_overrides: dict, 覆盖网格配置参数 (如 x_lower_factor, v_upper_abs)

    返回值:
        list of dict: 每个期权一行的结果
    """
    layer1_config = copy.deepcopy(LAYER1_GRID_CONFIG)
    layer1_config['m'] = m
    layer2_config = copy.deepcopy(LAYER2_GRID_CONFIG)
    layer2_config['N'] = N

    if grid_config_overrides:
        layer1_config.update({k: v for k, v in grid_config_overrides.items()
                              if k.startswith('v_')})
        layer2_config.update({k: v for k, v in grid_config_overrides.items()
                              if k.startswith('x_')})

    t0_build = time.time()

    variance_grid = build_variance_grid(layer1_config, model_params)

    if use_strike_locked and strikes_for_grid is not None:
        price_grid = build_strike_locked_price_grid(
            layer2_config, model_params, variance_grid, strikes_for_grid
        )
        N_actual = len(price_grid)
    elif use_adaptive and strikes_for_grid is not None:
        price_grid = build_adaptive_price_grid(
            layer2_config, model_params, variance_grid, strikes_for_grid
        )
        N_actual = len(price_grid)
    else:
        price_grid = build_price_grid(layer2_config, model_params, variance_grid)
        N_actual = N

    Q_variance = construct_variance_generator(variance_grid, model_params)
    G_sv_list = construct_all_regime_generators(
        price_grid, variance_grid, model_params, Q_variance=Q_variance
    )
    G_svj_list = construct_all_regime_generators_svj(
        price_grid, variance_grid, model_params, jump_params
    )

    t_build = time.time() - t0_build

    results = []

    for T in MATURITIES:
        for K in STRIKES:
            opt_p = {'K': K, 'T': T, 'option_type': 'put'}

            heston_price = compute_heston_price(
                S_0=model_params['S_0'], K=K, V_0=model_params['V_0'],
                r=model_params['r'], kappa=model_params['kappa'],
                theta=model_params['theta'], sigma_v=model_params['sigma_v'],
                rho=model_params['rho'], T=T, option_type='put',
            )

            svj_price = compute_svj_price(
                S0=model_params['S_0'], K=K, V0=model_params['V_0'],
                r=model_params['r'], kappa=model_params['kappa'],
                theta=model_params['theta'], sigma_v=model_params['sigma_v'],
                rho=model_params['rho'], T=T,
                lambda_jump=jump_params['lambda_jump'],
                mu_J=jump_params['mu_J'],
                sigma_J=jump_params['sigma_J'],
                option_type='put',
            )

            t0_sv = time.time()
            ctmc_sv = price_european_fast(
                Q_variance, G_sv_list, price_grid, variance_grid,
                model_params, opt_p, n_steps, smooth_payoff=smooth_payoff
            )
            t_sv = time.time() - t0_sv

            t0_svj = time.time()
            ctmc_svj = price_european_fast(
                Q_variance, G_svj_list, price_grid, variance_grid,
                model_params, opt_p, n_steps, smooth_payoff=smooth_payoff
            )
            t_svj = time.time() - t0_svj

            re_sv = (ctmc_sv - heston_price) / max(heston_price, 1e-10) * 100
            re_svj = (ctmc_svj - svj_price) / max(svj_price, 1e-10) * 100

            results.append({
                'm': m, 'N': N_actual, 'n_steps': n_steps,
                'grid_label': '{}x{}'.format(m, N),
                'grid_mode': 'strike_locked' if use_strike_locked else ('adaptive' if use_adaptive else 'standard'),
                'smooth_payoff': smooth_payoff,
                'adaptive': use_adaptive or use_strike_locked,
                'T': T, 'K': K,
                'Moneyness': K / model_params['S_0'],
                'CTMC_SV': round(ctmc_sv, 6),
                'Heston': round(heston_price, 6),
                'RE_SV_%': round(re_sv, 4),
                'CTMC_SVJ': round(ctmc_svj, 6),
                'SVJ_Analytical': round(svj_price, 6),
                'RE_SVJ_%': round(re_svj, 4),
                'Time_SV_s': round(t_sv, 3),
                'Time_SVJ_s': round(t_svj, 3),
                'Build_Time_s': round(t_build, 3),
                'Total_States': m * N_actual,
            })

            logger.info(
                "  {}x{} T={:.2f} K={:3d} | "
                "SV: {:.4f} vs {:.4f} ({:+.2f}%) | "
                "SVJ: {:.4f} vs {:.4f} ({:+.2f}%) | "
                "{:.2f}s".format(
                    m, N_actual, T, K,
                    ctmc_sv, heston_price, re_sv,
                    ctmc_svj, svj_price, re_svj,
                    t_sv + t_svj,
                )
            )

    return results


def run_convergence_study(model_params=None, jump_params=None,
                           use_adaptive=False, use_strike_locked=False,
                           smooth_payoff=False, grid_config_overrides=None):
    """
    运行完整收敛性研究

    参数:
        model_params (dict): 模型参数
        jump_params (dict): 跳跃参数
        use_adaptive (bool): 是否使用自适应网格
        use_strike_locked (bool): 是否使用行权价对齐网格
        smooth_payoff (bool): 是否使用平滑化支付函数
        grid_config_overrides (dict): 覆盖网格配置参数

    返回值:
        pd.DataFrame: 所有结果
    """
    timestamp_str = setup_logging()

    if model_params is None:
        model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
    if jump_params is None:
        jump_params = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)

    grid_mode = 'strike_locked' if use_strike_locked else ('adaptive' if use_adaptive else 'standard')
    logger.info("=" * 70)
    logger.info("Convergence Study: 2D-CTMC-SVJ")
    logger.info("Grid configs: {}".format(
        ', '.join(g['label'] for g in GRID_CONFIGS)))
    logger.info("Grid mode: {}".format(grid_mode))
    logger.info("Smooth payoff: {}".format(smooth_payoff))
    if grid_config_overrides:
        logger.info("Grid overrides: {}".format(grid_config_overrides))
    logger.info("=" * 70)

    all_results = []

    for grid_cfg in GRID_CONFIGS:
        m = grid_cfg['m']
        N = grid_cfg['N']
        n_steps = min(max(N, 200), 800)

        logger.info("\n--- Grid: {}x{}, n_steps={} ---".format(m, N, n_steps))

        try:
            rows = run_single_config(
                model_params, jump_params, m, N, n_steps,
                use_adaptive=use_adaptive,
                use_strike_locked=use_strike_locked,
                strikes_for_grid=STRIKES if (use_adaptive or use_strike_locked) else None,
                smooth_payoff=smooth_payoff,
                grid_config_overrides=grid_config_overrides,
            )
            all_results.extend(rows)
        except Exception as e:
            logger.error("Grid {}x{} FAILED: {}".format(m, N, e))
            import traceback
            logger.error(traceback.format_exc())

    df = pd.DataFrame(all_results)

    output_path = "{}convergence_study_2d_ctmc_{}.csv".format(
        FILE_PATHS['result_dir'], timestamp_str)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info("\nResults saved to {}".format(output_path))

    _print_convergence_summary(df)

    return df


def _print_convergence_summary(df):
    """
    打印收敛性汇总表
    """
    logger.info("\n" + "=" * 70)
    logger.info("CONVERGENCE SUMMARY")
    logger.info("=" * 70)

    for T in sorted(df['T'].unique()):
        sub_T = df[df['T'] == T]
        logger.info("\n--- T = {:.2f} ---".format(T))
        logger.info(
            "{:>10s} {:>10s} {:>8s} | {:>10s} {:>10s} | {:>8s}".format(
                'Grid', 'States', 'Steps',
                'Mean_SV_RE%', 'Max_SV_RE%', 'Time_s'))
        logger.info("-" * 70)

        for grid_label in sorted(sub_T['grid_label'].unique()):
            sub = sub_T[sub_T['grid_label'] == grid_label]
            if len(sub) == 0:
                continue
            mean_sv = sub['RE_SV_%'].mean()
            max_sv = sub['RE_SV_%'].abs().max()
            mean_svj = sub['RE_SVJ_%'].mean()
            max_svj = sub['RE_SVJ_%'].abs().max()
            total_time = sub['Time_SV_s'].mean() + sub['Time_SVJ_s'].mean()
            states = sub['Total_States'].iloc[0]

            logger.info(
                "{:>10s} {:>10d} {:>8d} | "
                "{:+10.2f} {:10.2f} | {:8.2f}".format(
                    grid_label, states, sub['n_steps'].iloc[0],
                    mean_sv, max_sv, total_time))

    logger.info("\n--- Wing Accuracy (K=85, T=1.0) ---")
    wing = df[(df['K'] == 85) & (df['T'] == 1.0)]
    if len(wing) > 0:
        for _, row in wing.sort_values('m').iterrows():
            logger.info(
                "  {:>10s}: SV_RE={:+.2f}%, SVJ_RE={:+.2f}%".format(
                    row['grid_label'], row['RE_SV_%'], row['RE_SVJ_%']))

    logger.info("\n--- ATM Accuracy (K=100, T=1.0) ---")
    atm = df[(df['K'] == 100) & (df['T'] == 1.0)]
    if len(atm) > 0:
        for _, row in atm.sort_values('m').iterrows():
            logger.info(
                "  {:>10s}: SV_RE={:+.2f}%, SVJ_RE={:+.2f}%".format(
                    row['grid_label'], row['RE_SV_%'], row['RE_SVJ_%']))


def run_nsteps_study(model_params=None, jump_params=None):
    """
    时间步数收敛性研究 (固定网格 40×200，变化 n_steps)
    """
    if model_params is None:
        model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
    if jump_params is None:
        jump_params = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)

    timestamp_str = setup_logging()
    logger.info("=" * 70)
    logger.info("Time Step Convergence Study (40x200 grid)")
    logger.info("=" * 70)

    m, N = 40, 200
    layer1_config = dict(LAYER1_GRID_CONFIG, m=m)
    layer2_config = dict(LAYER2_GRID_CONFIG, N=N)

    variance_grid = build_variance_grid(layer1_config, model_params)
    price_grid = build_price_grid(layer2_config, model_params, variance_grid)
    Q_variance = construct_variance_generator(variance_grid, model_params)
    G_sv_list = construct_all_regime_generators(
        price_grid, variance_grid, model_params, Q_variance=Q_variance
    )

    results = []
    K, T = 100, 1.0
    opt_p = {'K': K, 'T': T, 'option_type': 'put'}
    heston_price = compute_heston_price(
        S_0=model_params['S_0'], K=K, V_0=model_params['V_0'],
        r=model_params['r'], kappa=model_params['kappa'],
        theta=model_params['theta'], sigma_v=model_params['sigma_v'],
        rho=model_params['rho'], T=T, option_type='put',
    )

    for n_steps in N_STEPS_LIST:
        t0 = time.time()
        ctmc_lie = price_european_fast(
            Q_variance, G_sv_list, price_grid, variance_grid,
            model_params, opt_p, n_steps
        )
        t_lie = time.time() - t0

        t0 = time.time()
        ctmc_strang = price_european_strang(
            Q_variance, G_sv_list, price_grid, variance_grid,
            model_params, opt_p, n_steps
        )
        t_strang = time.time() - t0

        re_lie = (ctmc_lie - heston_price) / max(heston_price, 1e-10) * 100
        re_strang = (ctmc_strang - heston_price) / max(heston_price, 1e-10) * 100

        results.append({
            'n_steps': n_steps,
            'dt': T / n_steps,
            'CTMC_Lie': round(ctmc_lie, 8),
            'CTMC_Strang': round(ctmc_strang, 8),
            'Heston': round(heston_price, 8),
            'RE_Lie_%': round(re_lie, 6),
            'RE_Strang_%': round(re_strang, 6),
            'Lie_Time_s': round(t_lie, 3),
            'Strang_Time_s': round(t_strang, 3),
        })

        logger.info(
            "  n_steps={:4d} dt={:.5f} | "
            "Lie={:.6f} ({:+.4f}%) Strang={:.6f} ({:+.4f}%) | "
            "{:.2f}s + {:.2f}s".format(
                n_steps, T / n_steps,
                ctmc_lie, re_lie, ctmc_strang, re_strang,
                t_lie, t_strang))

    df = pd.DataFrame(results)
    output_path = "{}nsteps_convergence_{}.csv".format(
        FILE_PATHS['result_dir'], timestamp_str)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info("Results saved to {}".format(output_path))

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='2D-CTMC Convergence Study')
    parser.add_argument('--mode',
                        choices=['grid', 'nsteps', 'adaptive', 'strike_locked',
                                 'smooth', 'wide', 'all'],
                        default='all', help='Study mode')
    args = parser.parse_args()

    mp = copy.deepcopy(HESTON_DEFAULT_PARAMS)
    jp = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)

    wide_overrides = {
        'x_lower_factor': 5.0,
        'x_upper_factor': 5.0,
        'v_upper_abs': 0.30,
    }

    if args.mode in ('grid', 'all'):
        logger.info("\n>>> Standard Grid Convergence <<<")
        run_convergence_study(mp, jp, use_adaptive=False)

    if args.mode in ('adaptive', 'all'):
        logger.info("\n>>> Adaptive Grid Convergence <<<")
        run_convergence_study(mp, jp, use_adaptive=True)

    if args.mode in ('strike_locked', 'all'):
        logger.info("\n>>> Strike-Locked Grid Convergence <<<")
        run_convergence_study(mp, jp, use_strike_locked=True)

    if args.mode in ('smooth', 'all'):
        logger.info("\n>>> Smooth Payoff Convergence (40x200) <<<")
        run_convergence_study(mp, jp, use_adaptive=False, smooth_payoff=True)
        run_convergence_study(mp, jp, use_strike_locked=True, smooth_payoff=True)

    if args.mode in ('wide', 'all'):
        logger.info("\n>>> Wide Grid Range Convergence <<<")
        run_convergence_study(mp, jp, use_adaptive=False, grid_config_overrides=wide_overrides)
        run_convergence_study(mp, jp, use_strike_locked=True, grid_config_overrides=wide_overrides)

    if args.mode in ('nsteps', 'all'):
        logger.info("\n>>> Time Step Convergence <<<")
        run_nsteps_study(mp, jp)
