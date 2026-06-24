# -*- coding: utf-8 -*-
"""
文件名: main.py
功能描述: 2D-CTMC-SVJ 项目主入口
          Phase 0: 数据加载与校准 (真实数据 or 模拟数据)
          Phase 1: Heston 模型欧式期权验证
          Phase 2: SVJ 模型欧式期权验证
          Phase 3: 美式期权定价演示
作者: [Author]
创建日期: 2026-05-06
"""

import sys
import os
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
    validate_grid,
)
from src.layer1_variance import (
    construct_variance_generator,
    get_variance_state_index,
)
from src.layer2_price import (
    construct_all_regime_generators,
    compute_initial_auxiliary_value,
)
from src.combined_generator import construct_combined_generator
from src.heston_analytical import compute_heston_price
from src.svj_analytical import compute_svj_price
from src.jump_generator import construct_all_regime_generators_svj
from src.option_pricing import price_european_fast, price_european_strang
from src.american_pricing import price_american_fast
from src.data_loader import generate_mock_market_data, load_options_data, prepare_calibration_data
from src.calibration import calibrate_full


def setup_logging():
    """
    配置日志系统
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"{FILE_PATHS['log_dir']}run_2d_ctmc_{timestamp_str}.log"
    create_directories([FILE_PATHS['log_dir']])
    logger.add(log_file_path, level="DEBUG", format=LOG_CONFIG['format'])

    return timestamp_str


def build_ctmc_infrastructure(model_params, layer1_config, layer2_config,
                              stencil='auto'):
    """
    构建 CTMC 基础设施（网格 + 生成元）

    所有定价场景共享同一套网格和生成元。

    参数:
        model_params (dict): 模型参数
        layer1_config (dict): Layer 1 网格配置
        layer2_config (dict): Layer 2 网格配置
        stencil (str): 模板类型 '3pt', '5pt', 'auto'

    返回值:
        dict: 包含 variance_grid, price_grid, Q_variance, G_sv_list, G_svj_list
    """
    logger.info("=" * 60)
    logger.info("Building CTMC Infrastructure (stencil={})".format(stencil))
    logger.info("=" * 60)

    variance_grid = build_variance_grid(layer1_config, model_params)
    price_grid = build_price_grid(layer2_config, model_params, variance_grid)
    validate_grid(variance_grid)
    validate_grid(price_grid)

    m = len(variance_grid)
    N = len(price_grid)
    logger.info(f"State space: {m} variance × {N} price = {m * N} total")

    Q_variance = construct_variance_generator(variance_grid, model_params, stencil=stencil)
    G_sv_list = construct_all_regime_generators(
        price_grid, variance_grid, model_params, stencil=stencil,
        Q_variance=Q_variance
    )

    return {
        'variance_grid': variance_grid,
        'price_grid': price_grid,
        'Q_variance': Q_variance,
        'G_sv_list': G_sv_list,
        'm': m,
        'N': N,
        'stencil': stencil,
    }


# =============================================================================
# Phase 1: Heston European Validation
# =============================================================================

def run_heston_validation(infra, model_params, compute_params):
    """
    Phase 1: Heston 欧式期权验证 — CTMC vs 闭式解
             对比 Lie-Trotter 和 Strang splitting
    """
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Heston European Option Validation")
    logger.info("=" * 60)

    Q_variance = infra['Q_variance']
    G_sv_list = infra['G_sv_list']
    price_grid = infra['price_grid']
    variance_grid = infra['variance_grid']

    strikes = np.array([80, 90, 95, 100, 105, 110, 120])
    maturities = [0.25, 0.5, 1.0]
    n_steps = compute_params.get('n_time_steps', 300)

    results = []

    for T in maturities:
        for K in strikes:
            opt_p = {'K': K, 'T': T, 'option_type': 'put'}

            t0 = time.time()
            ctmc_price_lie = price_european_fast(
                Q_variance, G_sv_list, price_grid, variance_grid,
                model_params, opt_p, n_steps
            )
            lie_time = time.time() - t0

            t0 = time.time()
            ctmc_price_strang = price_european_strang(
                Q_variance, G_sv_list, price_grid, variance_grid,
                model_params, opt_p, n_steps
            )
            strang_time = time.time() - t0

            heston_price = compute_heston_price(
                S_0=model_params['S_0'], K=K, V_0=model_params['V_0'],
                r=model_params['r'], kappa=model_params['kappa'],
                theta=model_params['theta'], sigma_v=model_params['sigma_v'],
                rho=model_params['rho'], T=T, option_type='put',
            )

            abs_err_lie = abs(ctmc_price_lie - heston_price)
            rel_err_lie = abs_err_lie / max(heston_price, 1e-10) * 100
            abs_err_strang = abs(ctmc_price_strang - heston_price)
            rel_err_strang = abs_err_strang / max(heston_price, 1e-10) * 100

            results.append({
                'Phase': 'Heston',
                'T': T, 'K': K,
                'Moneyness': K / model_params['S_0'],
                'CTMC_Lie': round(ctmc_price_lie, 6),
                'CTMC_Strang': round(ctmc_price_strang, 6),
                'Analytical_Price': round(heston_price, 6),
                'RE_Lie_%': round(rel_err_lie, 4),
                'RE_Strang_%': round(rel_err_strang, 4),
                'Improvement_%': round(rel_err_lie - rel_err_strang, 4),
                'Lie_Time_s': round(lie_time, 4),
                'Strang_Time_s': round(strang_time, 4),
            })

            logger.info(
                f"  T={T:.2f} K={K:6.1f} | Lie={ctmc_price_lie:8.4f} "
                f"Strang={ctmc_price_strang:8.4f} Heston={heston_price:8.4f} | "
                f"RE: Lie={rel_err_lie:.2f}% Strang={rel_err_strang:.2f}% "
                f"({rel_err_lie - rel_err_strang:+.2f}%)"
            )

    df = pd.DataFrame(results)
    mean_lie = df['RE_Lie_%'].mean()
    mean_strang = df['RE_Strang_%'].mean()
    logger.info(f"  Mean RE: Lie={mean_lie:.4f}%, Strang={mean_strang:.4f}%, "
                f"Improvement={mean_lie - mean_strang:+.4f}%")

    return df


# =============================================================================
# Phase 2: SVJ European Validation
# =============================================================================

def run_svj_validation(infra, model_params, jump_params, compute_params):
    """
    Phase 2: SVJ 欧式期权验证 — CTMC-SVJ vs Bates 半闭式解
             对比 Lie-Trotter 和 Strang splitting
    """
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: SVJ European Option Validation")
    logger.info("=" * 60)

    price_grid = infra['price_grid']
    variance_grid = infra['variance_grid']
    Q_variance = infra['Q_variance']

    G_svj_list = construct_all_regime_generators_svj(
        price_grid, variance_grid, model_params, jump_params,
        stencil=infra.get('stencil', 'auto')
    )

    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    maturities = [0.5, 1.0]
    n_steps = compute_params.get('n_time_steps', 300)

    results = []

    for T in maturities:
        for K in strikes:
            opt_p = {'K': K, 'T': T, 'option_type': 'put'}

            ctmc_lie = price_european_fast(
                Q_variance, G_svj_list, price_grid, variance_grid,
                model_params, opt_p, n_steps
            )

            ctmc_strang = price_european_strang(
                Q_variance, G_svj_list, price_grid, variance_grid,
                model_params, opt_p, n_steps
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

            heston_price = compute_heston_price(
                S_0=model_params['S_0'], K=K, V_0=model_params['V_0'],
                r=model_params['r'], kappa=model_params['kappa'],
                theta=model_params['theta'], sigma_v=model_params['sigma_v'],
                rho=model_params['rho'], T=T, option_type='put',
            )

            re_lie = abs(ctmc_lie - svj_price) / max(svj_price, 1e-10) * 100
            re_strang = abs(ctmc_strang - svj_price) / max(svj_price, 1e-10) * 100

            results.append({
                'Phase': 'SVJ',
                'T': T, 'K': K,
                'CTMC_Lie': round(ctmc_lie, 6),
                'CTMC_Strang': round(ctmc_strang, 6),
                'SVJ_Analytical': round(svj_price, 6),
                'Heston_Analytical': round(heston_price, 6),
                'RE_Lie_%': round(re_lie, 4),
                'RE_Strang_%': round(re_strang, 4),
                'Improvement_%': round(re_lie - re_strang, 4),
            })

            logger.info(
                f"  T={T:.2f} K={K:6.1f} | Lie={ctmc_lie:8.4f} "
                f"Strang={ctmc_strang:8.4f} SVJ={svj_price:8.4f} | "
                f"RE: Lie={re_lie:.2f}% Strang={re_strang:.2f}% "
                f"({re_lie - re_strang:+.2f}%)"
            )

    df = pd.DataFrame(results)
    mean_lie = df['RE_Lie_%'].mean()
    mean_strang = df['RE_Strang_%'].mean()
    logger.info(f"  Mean RE: Lie={mean_lie:.4f}%, Strang={mean_strang:.4f}%, "
                f"Improvement={mean_lie - mean_strang:+.4f}%")

    return df


# =============================================================================
# Phase 3: American Option Pricing
# =============================================================================

def run_american_pricing(infra, model_params, jump_params, compute_params):
    """
    Phase 3: 美式期权定价演示 — SV 和 SVJ 模型
    """
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: American Option Pricing")
    logger.info("=" * 60)

    Q_variance = infra['Q_variance']
    G_sv_list = infra['G_sv_list']
    price_grid = infra['price_grid']
    variance_grid = infra['variance_grid']

    G_svj_list = construct_all_regime_generators_svj(
        price_grid, variance_grid, model_params, jump_params,
        stencil=infra.get('stencil', 'auto')
    )

    strikes = [95, 100, 105]
    maturities = [0.25, 0.5, 1.0]
    n_steps = compute_params.get('n_time_steps', 300)

    results = []

    for T in maturities:
        for K in strikes:
            opt_p = {'K': K, 'T': T, 'option_type': 'put'}

            am_sv_lie = price_american_fast(
                Q_variance, G_sv_list, price_grid, variance_grid,
                model_params, opt_p, n_steps, splitting='lie'
            )

            am_sv_strang = price_american_fast(
                Q_variance, G_sv_list, price_grid, variance_grid,
                model_params, opt_p, n_steps, splitting='strang'
            )

            am_svj_strang = price_american_fast(
                Q_variance, G_svj_list, price_grid, variance_grid,
                model_params, opt_p, n_steps, splitting='strang'
            )

            # European analytical for reference
            eu_svj = compute_svj_price(
                S0=model_params['S_0'], K=K, V0=model_params['V_0'],
                r=model_params['r'], kappa=model_params['kappa'],
                theta=model_params['theta'], sigma_v=model_params['sigma_v'],
                rho=model_params['rho'], T=T,
                lambda_jump=jump_params['lambda_jump'],
                mu_J=jump_params['mu_J'],
                sigma_J=jump_params['sigma_J'],
                option_type='put',
            )

            results.append({
                'Model': 'SV_Lie',
                'T': T, 'K': K,
                'American_CTMC': round(am_sv_lie['american_price'], 6),
                'European_CTMC': round(am_sv_lie['european_price'], 6),
                'EEP': round(am_sv_lie['early_exercise_premium'], 6),
                'EEP_%': round(
                    am_sv_lie['early_exercise_premium'] /
                    max(am_sv_lie['european_price'], 1e-10) * 100, 4
                ),
            })

            results.append({
                'Model': 'SV_Strang',
                'T': T, 'K': K,
                'American_CTMC': round(am_sv_strang['american_price'], 6),
                'European_CTMC': round(am_sv_strang['european_price'], 6),
                'EEP': round(am_sv_strang['early_exercise_premium'], 6),
                'EEP_%': round(
                    am_sv_strang['early_exercise_premium'] /
                    max(am_sv_strang['european_price'], 1e-10) * 100, 4
                ),
            })

            results.append({
                'Model': 'SVJ_Strang',
                'T': T, 'K': K,
                'American_CTMC': round(am_svj_strang['american_price'], 6),
                'European_CTMC': round(am_svj_strang['european_price'], 6),
                'EEP': round(am_svj_strang['early_exercise_premium'], 6),
                'EEP_%': round(
                    am_svj_strang['early_exercise_premium'] /
                    max(am_svj_strang['european_price'], 1e-10) * 100, 4
                ),
                'SVJ_Analytical_EU': round(eu_svj, 6),
            })

            logger.info(
                f"  T={T:.2f} K={K} | SV_Lie: Am={am_sv_lie['american_price']:.4f} "
                f"SV_Strang: Am={am_sv_strang['american_price']:.4f} | "
                f"SVJ_Strang: Am={am_svj_strang['american_price']:.4f} "
                f"Eu={am_svj_strang['european_price']:.4f}"
            )

    return pd.DataFrame(results)


# =============================================================================
# Phase 0: Data Loading & Calibration
# =============================================================================

def run_calibration(model_params, jump_params, compute_params, use_mock=True):
    """
    Phase 0: 加载数据 + 两阶段校准

    参数:
        model_params (dict): 初始模型参数 (作为校准起点)
        jump_params (dict): 初始跳跃参数
        use_mock (bool): True=用模拟数据, False=从 input/ 加载真实数据

    返回值:
        tuple: (calibrated_model_params, calibrated_jump_params, cal_data)
    """
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0: Data Loading & Calibration")
    logger.info("=" * 60)

    if use_mock:
        logger.info("Using MOCK data (synthetic market prices)")
        cal_data = generate_mock_market_data(
            model_params, jump_params,
            strikes=[85, 90, 95, 100, 105, 110, 115],
            maturities=[0.25, 0.5, 1.0],
            noise_level=0.02,
        )
    else:
        logger.info("Loading REAL market data from input/")
        options_path = FILE_PATHS.get('options_data', 'input/Options_au2412.csv')
        options_df = load_options_data(options_path)
        S_0 = model_params['S_0']
        r = model_params['r']
        cal_data = prepare_calibration_data(options_df, S_0, r)

    cal_config = {
        'method': compute_params.get('calibration_method', 'L-BFGS-B'),
        'n_multistart': compute_params.get('n_multistart', 3),
        'use_de': compute_params.get('use_de', False),
    }

    result = calibrate_full(
        cal_data,
        initial_model_params=model_params,
        initial_jump_params=jump_params,
        config=cal_config,
    )

    return (
        result['model_params'],
        result['jump_params'],
        cal_data,
        result,
    )


# =============================================================================
# Summary Report
# =============================================================================

def print_summary_report(all_results):
    """
    打印汇总报告
    """
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 60)

    for phase_name, df in all_results.items():
        logger.info(f"\n--- {phase_name} ---")
        if 'Rel_Error_%' in df.columns:
            mean_re = df['Rel_Error_%'].mean()
            max_re = df['Rel_Error_%'].max()
            logger.info(f"  Mean RE: {mean_re:.4f}%, Max RE: {max_re:.4f}%")
        if 'EEP' in df.columns:
            for model in df['Model'].unique():
                sub = df[df['Model'] == model]
                mean_eep = sub['EEP'].mean()
                logger.info(f"  {model} Mean EEP: {mean_eep:.4f}")

    logger.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    """
    主函数：Phase 0 (校准) + Phase 1-3 (验证)
    """
    timestamp_str = setup_logging()

    logger.info("2D-CTMC-SVJ Pricing System")
    logger.info(f"Start time: {timestamp_str}")

    model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
    jump_params = copy.deepcopy(SVJ_JUMP_DEFAULT_PARAMS)
    layer1_config = copy.deepcopy(LAYER1_GRID_CONFIG)
    layer2_config = copy.deepcopy(LAYER2_GRID_CONFIG)
    compute_params = copy.deepcopy(CTMC_COMPUTE_PARAMS)

    use_mock = not os.path.exists(
        FILE_PATHS.get('options_data', 'input/Options_au2412.csv')
    )

    try:
        # Phase 0: Calibration
        cal_model, cal_jumps, cal_data, cal_result = run_calibration(
            model_params, jump_params, compute_params, use_mock=use_mock
        )
        save_results(
            pd.DataFrame([{
                'param': k, 'value': v,
                'stage': 'model',
            } for k, v in cal_model.items()] + [{
                'param': k, 'value': v,
                'stage': 'jump',
            } for k, v in cal_jumps.items()]),
            "phase0_calibration", FILE_PATHS['result_dir'], '2d_ctmc'
        )

        # Use calibrated params for subsequent phases
        model_params = cal_model
        jump_params = cal_jumps

        logger.info(f"\nCalibrated model: {model_params}")
        logger.info(f"Calibrated jumps: {jump_params}")

        # Phase 1-3: Validation with calibrated parameters
        infra = build_ctmc_infrastructure(model_params, layer1_config, layer2_config)

        logger.info(f"Grid:  {layer1_config['m']} variance x {layer2_config['N']} price, G_sv tensor {infra['G_sv_list'].shape}")

        df_heston = run_heston_validation(infra, model_params, compute_params)
        save_results(df_heston, "phase1_heston", FILE_PATHS['result_dir'], '2d_ctmc')

        df_svj = run_svj_validation(infra, model_params, jump_params, compute_params)
        save_results(df_svj, "phase2_svj", FILE_PATHS['result_dir'], '2d_ctmc')

        df_american = run_american_pricing(infra, model_params, jump_params, compute_params)
        save_results(df_american, "phase3_american", FILE_PATHS['result_dir'], '2d_ctmc')

        print_summary_report({
            'Phase 0 - Calibration': cal_result,
            'Phase 1 - Heston': df_heston,
            'Phase 2 - SVJ': df_svj,
            'Phase 3 - American': df_american,
        })

        logger.info("\nAll phases completed successfully.")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
