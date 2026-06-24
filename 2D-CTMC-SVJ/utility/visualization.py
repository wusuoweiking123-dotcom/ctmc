# -*- coding: utf-8 -*-
"""
文件名: utility/visualization.py
功能描述: 可视化工具模块，提供网格分布、定价误差、收敛性分析的绘图功能
作者: [Author]
创建日期: 2026-05-06
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger


def plot_grid_distribution(grid, title='Grid Distribution', save_path=None):
    """
    绘制网格点分布图

    参数:
        grid (np.ndarray): 网格点数组        title (str): 图表标题
        save_path (str): 保存路径（None 则不保存）    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(grid, np.zeros_like(grid), '|', markersize=10)
    ax1.set_ylabel('Grid Points')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    spacings = np.diff(grid)
    ax2.plot(grid[:-1], spacings, 'o-', markersize=3)
    ax2.set_xlabel('Grid Value')
    ax2.set_ylabel('Spacing')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grid plot saved to {save_path}")

    plt.close()


def plot_pricing_comparison(results_df, save_path=None):
    """
    绘制 CTMC vs Heston 定价对比图
    参数:
        results_df (pd.DataFrame): 定价结果 DataFrame
        save_path (str): 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, T in enumerate(sorted(results_df['T'].unique())):
        ax = axes[idx]
        sub = results_df[results_df['T'] == T].sort_values('K')

        ax.plot(sub['K'], sub['CTMC_Price'], 'bo-', label='CTMC', markersize=4)
        ax.plot(sub['K'], sub['Heston_Price'], 'r--', label='Heston', markersize=4)
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Option Price')
        ax.set_title(f'T = {T:.2f} year')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('CTMC vs Heston Analytical Prices', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")

    plt.close()


def plot_convergence(convergence_df, save_path=None):
    """
    绘制网格收敛性图

    参数:
        convergence_df (pd.DataFrame): 收敛性结果        save_path (str): 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for m in sorted(convergence_df['m'].unique()):
        sub = convergence_df[convergence_df['m'] == m].sort_values('N')
        ax1.semilogy(sub['N'], sub['Abs_Error'], 'o-', label=f'm={m}')

    ax1.set_xlabel('N (price grid points)')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Convergence vs Price Grid Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for N in sorted(convergence_df['N'].unique()):
        sub = convergence_df[convergence_df['N'] == N].sort_values('m')
        ax2.semilogy(sub['m'], sub['Abs_Error'], 'o-', label=f'N={N}')

    ax2.set_xlabel('m (variance grid points)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Convergence vs Variance Grid Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Grid Convergence Study', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Convergence plot saved to {save_path}")

    plt.close()


def plot_error_heatmap(results_df, save_path=None):
    """
    绘制定价误差热力图
    参数:
        results_df (pd.DataFrame): 定价结果
        save_path (str): 保存路径
    """
    pivoted = results_df.pivot_table(
        index='K', columns='T', values='Rel_Error_%'
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivoted.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivoted.columns)))
    ax.set_xticklabels([f'{t:.2f}' for t in pivoted.columns])
    ax.set_yticks(range(len(pivoted.index)))
    ax.set_yticklabels(pivoted.index)
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Strike Price')
    ax.set_title('Relative Pricing Error (%)')

    plt.colorbar(im, ax=ax, label='Rel Error (%)')

    for i in range(len(pivoted.index)):
        for j in range(len(pivoted.columns)):
            val = pivoted.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if val < 1 else 'white')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error heatmap saved to {save_path}")

    plt.close()
