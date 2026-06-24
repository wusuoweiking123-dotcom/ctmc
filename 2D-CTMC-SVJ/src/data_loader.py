# -*- coding: utf-8 -*-
"""
文件名: src/data_loader.py
功能描述: 市场数据加载与预处理模块
          支持从 Wind 导出的 CSV 文件加载期货、期权、利率数据
          支持生成模拟市场数据用于测试校准流程
作者: [Author]
创建日期: 2026-05-07
"""

import os
import numpy as np
import pandas as pd
from loguru import logger


def load_futures_data(file_path):
    """
    加载期货价格数据

    参数:
        file_path (str): CSV文件路径

    返回值:
        pd.DataFrame: 标准化后的期货数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    col_map = _build_column_map(df.columns)
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f"Futures data loaded: {len(df)} records from {file_path}")
    return df


def load_options_data(file_path):
    """
    加载期权行情数据

    参数:
        file_path (str): CSV文件路径

    返回值:
        pd.DataFrame: 标准化后期权数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    col_map = _build_column_map(df.columns)
    df = df.rename(columns=col_map)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'expiry' in df.columns:
        df['expiry'] = pd.to_datetime(df['expiry'])

    if 'settle_price' in df.columns:
        df['market_price'] = df['settle_price']
    elif 'close' in df.columns:
        df['market_price'] = df['close']

    if 'T' not in df.columns and 'expiry' in df.columns and 'date' in df.columns:
        df['T'] = (df['expiry'] - df['date']).dt.days / 365.0
        df = df[df['T'] > 0.01].copy()

    df = df.reset_index(drop=True)
    logger.info(f"Options data loaded: {len(df)} records from {file_path}")
    return df


def load_risk_free_rate(file_path):
    """
    加载无风险利率数据

    参数:
        file_path (str): CSV文件路径

    返回值:
        pd.Series: 以日期为索引的利率序列
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    col_map = _build_column_map(df.columns)
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    rate_col = 'rate' if 'rate' in df.columns else df.columns[0]
    rates = df[rate_col] / 100.0 if df[rate_col].mean() > 1 else df[rate_col]

    logger.info(f"Risk-free rate loaded: {len(rates)} records")
    return rates


def prepare_calibration_data(options_df, S_0, r):
    """
    将期权数据整理为校准所需的标准化格式

    校准惯例: 使用 OTM 期权 (虚值期权)
    - K < S: 用 put
    - K > S: 用 call
    - K ≈ S: 两者都可用

    参数:
        options_df (pd.DataFrame): 期权数据
        S_0 (float): 标的资产当前价格
        r (float): 无风险利率

    返回值:
        pd.DataFrame: 校准用数据，包含 S, K, T, r, market_price, option_type, weight
    """
    cal = options_df.copy()

    required = ['K', 'T', 'market_price']
    for col in required:
        if col not in cal.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'option_type' not in cal.columns:
        cal['option_type'] = 'put'

    if 'S' not in cal.columns:
        cal['S'] = S_0
    if 'r' not in cal.columns:
        cal['r'] = r

    cal = cal[cal['market_price'] > 0].copy()
    cal = cal[cal['T'] > 0.01].copy()

    otm_mask = (
        ((cal['option_type'] == 'put') & (cal['K'] <= cal['S'])) |
        ((cal['option_type'] == 'call') & (cal['K'] >= cal['S']))
    )
    cal = cal[otm_mask].copy()

    cal['weight'] = 1.0 / np.maximum(cal['market_price'], 0.01)

    cal = cal.reset_index(drop=True)
    logger.info(
        f"Calibration data prepared: {len(cal)} OTM options, "
        f"K range=[{cal['K'].min():.1f}, {cal['K'].max():.1f}], "
        f"T range=[{cal['T'].min():.3f}, {cal['T'].max():.3f}]"
    )
    return cal


def generate_mock_market_data(
    model_params, jump_params,
    strikes=None, maturities=None,
    noise_level=0.02, seed=42
):
    """
    生成模拟市场数据用于测试校准流程

    使用已知的 Heston-SVJ 参数生成期权价格，加入随机噪声模拟买卖价差。

    参数:
        model_params (dict): Heston 模型参数
        jump_params (dict): SVJ 跳跃参数
        strikes (list): 行权价列表，默认 [85, 90, 95, 100, 105, 110, 115]
        maturities (list): 到期时间列表(年)，默认 [0.25, 0.5, 1.0]
        noise_level (float): 相对噪声水平，默认 2%
        seed (int): 随机种子

    返回值:
        pd.DataFrame: 模拟市场数据
    """
    from src.svj_analytical import compute_svj_price

    rng = np.random.default_rng(seed)

    if strikes is None:
        strikes = [85, 90, 95, 100, 105, 110, 115]
    if maturities is None:
        maturities = [0.25, 0.5, 1.0]

    S_0 = model_params['S_0']
    r = model_params['r']

    records = []
    for T in maturities:
        for K in strikes:
            if K <= S_0:
                otype = 'put'
            else:
                otype = 'call'

            true_price = compute_svj_price(
                S0=S_0, K=K, V0=model_params['V_0'],
                r=r, kappa=model_params['kappa'],
                theta=model_params['theta'],
                sigma_v=model_params['sigma_v'],
                rho=model_params['rho'], T=T,
                lambda_jump=jump_params['lambda_jump'],
                mu_J=jump_params['mu_J'],
                sigma_J=jump_params['sigma_J'],
                option_type=otype,
            )

            noise = rng.normal(0, noise_level * max(true_price, 0.01))
            market_price = max(true_price + noise, 0.01)

            records.append({
                'S': S_0, 'K': K, 'T': T, 'r': r,
                'option_type': otype,
                'market_price': round(market_price, 4),
                'true_price': round(true_price, 4),
            })

    df = pd.DataFrame(records)
    df['weight'] = 1.0 / np.maximum(df['market_price'], 0.01)

    logger.info(
        f"Mock data generated: {len(df)} options, "
        f"noise={noise_level*100:.1f}%, "
        f"K={strikes}, T={maturities}"
    )
    return df


def _build_column_map(columns):
    """
    构建中文/英文列名到标准列名的映射

    参数:
        columns: 原始列名列表

    返回值:
        dict: 列名映射
    """
    mapping = {}

    cn_map = {
        'date': ['date', 'Date', '日期', '交易日期', 'trade_date', 'Trddate'],
        'close': ['close', 'Close', '收盘价', 'close_price', 'settle', '结算价'],
        'open': ['open', 'Open', '开盘价', 'open_price'],
        'high': ['high', 'High', '最高价', 'high_price'],
        'low': ['low', 'Low', '最低价', 'low_price'],
        'volume': ['volume', 'Volume', '成交量', 'vol', 'VOL'],
        'open_interest': ['open_interest', 'OI', '持仓量', 'oi', 'Openinterest'],
        'strike': ['strike', 'Strike', '行权价', '执行价', 'exercise_price', 'K'],
        'expiry': ['expiry', 'Expiry', '到期日', '到期时间', 'maturity', 'expire_date', 'last_trade_date'],
        'option_type': ['option_type', 'Option_Type', '类型', 'option_type_code', 'call_put', 'CP'],
        'bid_price': ['bid_price', 'Bid', '买价', 'bid', 'buy_price'],
        'ask_price': ['ask_price', 'Ask', '卖价', 'ask', 'sell_price'],
        'settle_price': ['settle_price', 'Settle', '结算价', 'settlement_price'],
        'contract_code': ['contract_code', 'code', '合约代码', 'contract', 'ticker', 'instrument_id'],
        'rate': ['rate', 'Rate', '利率', 'yield', 'shibor', 'close'],
        'implied_vol': ['implied_vol', 'IV', '隐含波动率', 'volatility', 'impl_vol'],
    }

    col_set = set(columns)
    for std_name, variants in cn_map.items():
        for v in variants:
            if v in col_set and std_name not in mapping:
                mapping[v] = std_name

    if 'K' in col_set and 'strike' not in mapping.get('K', ''):
        if 'K' not in mapping:
            pass
    if 'strike' in col_set:
        mapping['strike'] = 'K'

    return mapping
