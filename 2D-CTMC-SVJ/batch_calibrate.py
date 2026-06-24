# -*- coding: utf-8 -*-
"""
文件名: batch_calibrate.py
功能描述: au2412 期权多日批量校准
          - 逐日两阶段校准 (Heston → SVJ)
          - 热启动: 前一日参数作为下一日初始猜测
          - 输出参数时间序列 CSV
作者: [Author]
创建日期: 2026-05-08
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from commonConfig import (
    HESTON_DEFAULT_PARAMS,
    SVJ_JUMP_DEFAULT_PARAMS,
    FILE_PATHS,
    LOG_CONFIG,
)
from utility.file_utils import create_directories
from src.calibration import calibrate_full


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = "{}batch_cal_{}.log".format(FILE_PATHS['log_dir'], timestamp_str)
    create_directories([FILE_PATHS['log_dir']])
    logger.add(log_file_path, level="DEBUG", format=LOG_CONFIG['format'])
    return timestamp_str


def load_au2412_options(file_path=None):
    if file_path is None:
        file_path = FILE_PATHS['options_data']
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = df['Date'].astype(str)
    df['T'] = df['TAU']
    df['S'] = df['F_t']
    df['r'] = df['Rf']
    df['option_type'] = 'put'
    df = df[df['market_price'] > 0].copy()
    df = df[df['T'] > 0.01].copy()
    df = df[df['K'] <= df['S']].copy()
    df['weight'] = 1.0 / np.maximum(df['market_price'], 0.01)
    df = df.reset_index(drop=True)
    return df


def batch_calibrate(options_file=None, min_options=4,
                    method='L-BFGS-B', n_multistart=1):
    timestamp_str = setup_logging()
    logger.info("Batch Calibration: au2412 put options (futures)")
    logger.info("Config: method={}, n_multistart={}, min_options={}".format(
        method, n_multistart, min_options))

    options_df = load_au2412_options(options_file)
    unique_dates = sorted(options_df['date'].unique())
    logger.info("Total dates: {}, Total OTM options: {}".format(
        len(unique_dates), len(options_df)))

    config = {
        'method': method,
        'n_multistart': n_multistart,
        'use_de': False,
    }

    results = []
    prev_model = None
    prev_jump = None

    for i, date_val in enumerate(unique_dates):
        date_data = options_df[options_df['date'] == date_val].copy()

        if len(date_data) < min_options:
            logger.warning("[{}/{}] {} skipped: {} options < {}".format(
                i + 1, len(unique_dates), date_val, len(date_data), min_options))
            continue

        S_0 = date_data['S'].iloc[0]
        r_val = date_data['r'].iloc[0]

        cal_data = date_data[['S', 'K', 'T', 'r', 'market_price',
                              'option_type', 'weight']].copy()

        logger.info("[{}/{}] {} | {} options | S={:.2f} r={:.5f}".format(
            i + 1, len(unique_dates), date_val, len(cal_data), S_0, r_val))

        if prev_model is not None:
            model_init = prev_model.copy()
            model_init['S_0'] = S_0
            model_init['r'] = r_val
        else:
            model_init = HESTON_DEFAULT_PARAMS.copy()
            model_init['S_0'] = S_0
            model_init['r'] = r_val
            model_init['underlying_type'] = 'futures'

        jump_init = prev_jump.copy() if prev_jump is not None else SVJ_JUMP_DEFAULT_PARAMS.copy()

        t0 = time.time()
        try:
            result = calibrate_full(cal_data, model_init, jump_init, config)
            elapsed = time.time() - t0

            mp = result['model_params']
            jp = result['jump_params']
            feller = 2 * mp['kappa'] * mp['theta'] / max(mp['sigma_v'] ** 2, 1e-10)

            row = {
                'date': date_val,
                'S_0': S_0,
                'r': r_val,
                'n_options': len(cal_data),
                'V_0': mp['V_0'],
                'kappa': mp['kappa'],
                'theta': mp['theta'],
                'sigma_v': mp['sigma_v'],
                'rho': mp['rho'],
                'lambda_jump': jp['lambda_jump'],
                'mu_J': jp['mu_J'],
                'sigma_J': jp['sigma_J'],
                'feller_ratio': feller,
                'heston_rmse': result['stage1']['rmse'],
                'svj_rmse': result['stage2']['rmse'],
                'time_s': round(elapsed, 2),
            }
            results.append(row)

            prev_model = mp.copy()
            prev_jump = jp.copy()

            logger.info(
                "  V0={:.5f} k={:.3f} th={:.5f} sv={:.4f} rho={:.4f} | "
                "SVJ RMSE={:.4f} | {:.1f}s".format(
                    mp['V_0'], mp['kappa'], mp['theta'],
                    mp['sigma_v'], mp['rho'],
                    result['stage2']['rmse'], elapsed))
        except Exception as e:
            logger.error("  {} FAILED: {}".format(date_val, e))

    df_results = pd.DataFrame(results)

    output_path = "{}batch_calibration_{}.csv".format(
        FILE_PATHS['result_dir'], timestamp_str)
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info("Results saved to {}".format(output_path))

    if len(df_results) > 0:
        logger.info("Calibrated: {}/{} dates".format(
            len(results), len(unique_dates)))
        logger.info("Parameter statistics:")
        for col in ['V_0', 'kappa', 'theta', 'sigma_v', 'rho',
                     'lambda_jump', 'mu_J', 'sigma_J']:
            logger.info("  {:12s}: mean={:.5f} std={:.5f} "
                        "min={:.5f} max={:.5f}".format(
                            col,
                            df_results[col].mean(),
                            df_results[col].std(),
                            df_results[col].min(),
                            df_results[col].max()))
        logger.info("  SVJ RMSE   : mean={:.4f} median={:.4f}".format(
            df_results['svj_rmse'].mean(),
            df_results['svj_rmse'].median()))
        logger.info("  Total time : {:.1f}s".format(
            df_results['time_s'].sum()))

    return df_results


if __name__ == "__main__":
    batch_calibrate()
