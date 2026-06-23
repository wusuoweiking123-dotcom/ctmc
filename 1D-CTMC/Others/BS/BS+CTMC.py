import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# 构造对数均匀的状态空间网格
def construct_log_grid(S_min, S_max, num_bins):
    log_grid = np.linspace(np.log(S_min), np.log(S_max), num_bins)
    grid = np.exp(log_grid)
    return grid


# 构造生成元矩阵 Q
def theoretical_Q(grid, sigma, r):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))
    mu = r - 0.5 * sigma**2
    D = 0.5 * sigma**2

    for i in range(N):
        if i > 0:
            Q[i, i - 1] = D / h**2 - mu / (2 * h)
        if i < N - 1:
            Q[i, i + 1] = D / h**2 + mu / (2 * h)
        Q[i, i] = -np.sum(Q[i])  # 保证每行和为 0

    return Q


# 用 CTMC 定价欧式期权
def price_European_option(Q, centers, K, r, T, option_type='put'):
    P_T = expm(Q * T)

    if option_type == 'put':
        payoff = np.maximum(K - centers, 0)
    else:
        payoff = np.maximum(centers - K, 0)

    # 正确的矩阵乘法：@ 表示矩阵乘向量
    option_values = np.exp(-r * T) * P_T @ payoff
    return option_values, payoff


def main():
    # 参数设定
    sigma = 0.3
    r = 0.05
    T = 1
    S0 = 100
    K = 100
    num_bins = 1000

    # 构造 log 状态网格
    grid = construct_log_grid(40, 160, num_bins)  # 网格从50到150

    # 构造 Q 矩阵
    Q = theoretical_Q(grid, sigma, r)

    # CTMC 期权定价
    option_values, payoff = price_European_option(Q, grid, K, r, T, option_type='call')

    print("Payoff at Maturity:")
    print(payoff)
    print("\nOption Value at t = 0:")
    print(np.mean(option_values))


if __name__ == "__main__":
    main()
