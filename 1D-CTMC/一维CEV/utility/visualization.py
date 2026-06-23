import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger


def plot_rs_exercise_boundaries(boundaries_r1, boundaries_r2, time_to_maturity, result_dir='result'):
    """
    绘制 CEV-RS 模型下两个体制的美式期权早偿边界 (对应论文 Figure 2)

    :param boundaries_r1: Regime 1 的早偿边界价格数组 (长度为 N_t)
    :param boundaries_r2: Regime 2 的早偿边界价格数组 (长度为 N_t)
    :param time_to_maturity: 期权总的剩余到期时间 (年化，例如 68/365)
    """
    try:
        # 设置字体和图表样式以贴近学术论文风格
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'

        # 将时间步转换为“距离到期日的天数” (Days to expiration)
        # 假设数组索引 0 是今天(距离到期天数最大)，索引 -1 是到期日(距离到期天数为0)
        n_steps = len(boundaries_r1)
        total_days = time_to_maturity * 365

        # 构造 X 轴：从 total_days 递减到 0
        days_to_expiration = np.linspace(total_days, 0, n_steps)

        plt.figure(figsize=(8, 6))

        # 绘制两条边界线，对应论文里的深蓝色实线和红色虚线
        plt.plot(days_to_expiration, boundaries_r1, color='navy', linestyle='-', linewidth=2, label='Regime 1')
        plt.plot(days_to_expiration, boundaries_r2, color='crimson', linestyle='-.', linewidth=2, label='Regime 2')

        # 设置坐标轴标签
        plt.xlabel('Days to expiration', fontsize=12)
        plt.ylabel('Futures Price', fontsize=12)

        # 坐标轴刻度朝内 (经典的学术图表风格)
        plt.tick_params(direction='in', top=True, right=True)

        # 添加图例
        plt.legend(loc='upper right', frameon=True, fontsize=11)

        # 调整布局并保存
        plt.tight_layout()

        # 确保输出目录存在
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        save_path = os.path.join(result_dir, 'Figure2_Early_Exercise_Boundary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Early exercise boundary plot successfully saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot exercise boundary: {e}")