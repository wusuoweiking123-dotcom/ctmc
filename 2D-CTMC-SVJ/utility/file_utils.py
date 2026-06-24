# -*- coding: utf-8 -*-
"""
文件名: utility/file_utils.py
功能描述: 文件I/O工具函数，提供目录创建、数据读写、结果保存功能
作者: [Author]
创建日期: 2026-05-06
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger


def create_directories(dir_list):
    """
    批量创建目录

    参数:
        dir_list (list): 需要创建的目录路径列表
    """
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Directory created: {dir_path}")


def read_futures_data(file_path):
    """
    读取期货价格数据

    参数:
        file_path (str): 期货数据CSV文件路径

    返回值:
        pd.DataFrame: 包含 'Date' 和 'close' 列的DataFrame

    异常:
        FileNotFoundError: 文件不存在时
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Futures data file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Futures data loaded: {len(df)} records from {file_path}")
    return df


def read_option_data(file_path):
    """
    读取期权市场数据

    参数:
        file_path (str): 期权数据CSV文件路径

    返回值:
        pd.DataFrame: 期权数据DataFrame

    异常:
        FileNotFoundError: 文件不存在时
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Options data file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Options data loaded: {len(df)} records from {file_path}")
    return df


def save_results(data, filename, result_dir='result/', model_type='heston',
                 timestamp=True):
    """
    保存结果数据到CSV文件

    参数:
        data (pd.DataFrame): 要保存的结果数据
        filename (str): 文件名前缀
        result_dir (str): 结果目录路径
        model_type (str): 模型类型标识
        timestamp (bool): 是否在文件名中添加时间戳
    """
    os.makedirs(result_dir, exist_ok=True)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ''
    full_name = f"{filename}_{model_type}_{ts_str}.csv" if timestamp else f"{filename}_{model_type}.csv"
    file_path = os.path.join(result_dir, full_name)

    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False, encoding='utf-8-sig')
    elif isinstance(data, np.ndarray):
        np.savetxt(file_path, data, delimiter=',')
    elif isinstance(data, dict):
        pd.DataFrame([data]).to_csv(file_path, index=False, encoding='utf-8-sig')

    logger.info(f"Results saved to {file_path}")
